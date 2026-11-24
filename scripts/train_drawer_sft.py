"""Supervised PI05 LoRA on seeded SO-101 drawer demonstrations."""

import argparse
from bisect import bisect_right
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from eval_drawer import TASK
from evaluate_so101_mujoco_lora import BASE_SHA
from pi05_rl_runtime import load_ira_lora, make_processors, sha256

SOURCE_SHA = "c4b65f3134cd065f45dc7c7e40f2d9adb3291116e3cc0eec49f1aa60a49a8fd6"
VALIDATION_WINDOWS = 12
CHUNK = 50


class EpisodeFrames(Dataset):
    def __init__(self, root, rows, validation=False):
        self.data = []
        self.ends = []
        self.length = 0
        for row in rows:
            ep = root / row["directory"]
            with np.load(ep / "motion.npz") as motion:
                state = motion["state"]
                action = motion["action"]
                rgb_frames = motion["rgb_frames"]
            n = row["frames"]
            if (state.shape != (n, 6) or action.shape != (n, 6)
                    or not np.array_equal(rgb_frames, np.arange(0, n, 3))
                    or len(rgb_frames) < VALIDATION_WINDOWS or row["rgb_frames"] != len(rgb_frames)
                    or not np.isfinite(state).all() or not np.isfinite(action).all()):
                raise ValueError(f"Invalid demonstration {ep}")
            offsets = rgb_frames[np.linspace(0, len(rgb_frames)-1, VALIDATION_WINDOWS, dtype=int)] if validation else rgb_frames
            self.data.append((state, action, ep, offsets))
            self.length += len(offsets)
            self.ends.append(self.length)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        episode = bisect_right(self.ends, index)
        state, action, directory, offsets = self.data[episode]
        frame = offsets[index - (self.ends[episode - 1] if episode else 0)]
        indices = np.minimum(frame + np.arange(CHUNK), len(action) - 1)
        images = {}
        for name in ("front", "wrist"):
            with Image.open(directory / f"{name}-{frame:04d}.png") as image:
                rgb = np.array(image)
            if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Invalid {name} image at {directory}/{frame}")
            images[f"observation.images.{name}"] = torch.from_numpy(rgb).permute(2, 0, 1)
        return {
            **images,
            "observation.state": torch.from_numpy(state[frame]),
            "action": torch.from_numpy(action[indices]),
            "task": TASK,
        }


def write_json(path, data):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(data, indent=2) + "\n")
    temp.replace(path)


def prepare(pre, batch):
    for view in ("front", "wrist"):
        key = f"observation.images.{view}"
        batch[key] = batch[key].float().div_(255)
    return pre(batch)


def validate(policy, pre, batches):
    policy.eval()
    losses = []
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
        torch.manual_seed(701)
        for batch in batches:
            loss, _ = policy(prepare(pre, batch), reduction="none")
            losses.extend(loss.detach().cpu().tolist())
    policy.train()
    if not np.isfinite(losses).all():
        raise ValueError("Non-finite validation loss")
    return float(np.mean(losses)), len(losses)


def save_checkpoint(policy, optimizer, source, path, meta):
    staging = path.with_name(path.name + ".staging")
    staging.mkdir(parents=True)
    policy.save_pretrained(staging, safe_serialization=True, save_embedding_layers=False)
    shutil.copy2(source / "config.json", staging / "config.json")
    for file in source.glob("policy_*"):
        shutil.copy2(file, staging / file.name)
    torch.save({
        "optimizer": optimizer.state_dict(),
        "cpu_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state(),
    }, staging / "training_state.pt")
    write_json(staging / "training.json", meta)
    staging.rename(path)
    return sha256(path / "adapter_model.safetensors")


def main(args):
    if args.steps % args.accumulate or args.save_freq % args.accumulate:
        raise ValueError("steps and save-freq must end at an optimizer update")
    torch.set_num_threads(1)
    spec_data = (args.data / "manifest.json").read_bytes()
    manifest = json.loads(spec_data)
    if manifest["task"] != TASK or manifest["rgb_stride"] != 3:
        raise ValueError("Unexpected task or observation stride")
    rows = {row["episode"]: row for row in manifest["episodes"]}
    train_ids, val_ids = manifest["train"], manifest["validation"]
    if (len(train_ids) != args.train_episodes or len(val_ids) != args.validation_episodes
            or len(rows) != args.train_episodes + args.validation_episodes
            or set(train_ids) & set(val_ids) or set(train_ids + val_ids) != set(rows)
            or not all(row["success"] for row in rows.values())):
        raise ValueError("Expected disjoint successful drawer episodes and exact split counts")

    train = EpisodeFrames(args.data, [rows[i] for i in train_ids])
    val = EpisodeFrames(args.data, [rows[i] for i in val_ids], validation=True)
    metadata = {
        "method": "supervised_flow_matching", "starting_adapter": str(args.adapter),
        "starting_adapter_sha256": args.adapter_sha256, "base_sha256": BASE_SHA,
        "manifest_sha256": hashlib.sha256(spec_data).hexdigest(),
        "train_episodes": train_ids, "validation_episodes": val_ids,
        "train_frames": len(train), "validation_frames_per_episode": VALIDATION_WINDOWS,
        "batch_size": args.batch_size, "gradient_accumulation_steps": args.accumulate,
        "num_workers": args.num_workers, "steps": args.steps, "save_freq": args.save_freq,
        "learning_rate": args.lr, "offload_text_tables": args.offload_text_tables,
    }
    if args.continue_optimizer:
        prior = json.loads((args.adapter / "training.json").read_text())
        for key in ("learning_rate", "batch_size", "gradient_accumulation_steps"):
            if prior["config"][key] != metadata[key]:
                raise ValueError(f"Cannot continue optimizer with a different {key}")
        metadata["previous_steps"] = prior["step"] + prior["config"].get("previous_steps", 0)
        metadata["previous_optimizer_updates"] = (
            prior["optimizer_updates"] + prior["config"].get("previous_optimizer_updates", 0)
        )

    start_step = 0
    if args.resume:
        saved = json.loads((args.resume / "training.json").read_text())
        if saved["config"] != metadata:
            raise ValueError("Cannot resume with changed training data or settings")
        start_step = saved["step"]
    else:
        args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "config.json", metadata)

    # Draw the full sample order once so the next sample is stable after resume.
    rng = np.random.default_rng(20261121)
    count = args.steps * args.batch_size
    draws = np.concatenate([rng.permutation(len(train)) for _ in range(
        (count + len(train) - 1) // len(train))])[:count]
    train_loader = DataLoader(train, batch_size=args.batch_size,
                              sampler=draws[start_step * args.batch_size:].tolist(),
                              num_workers=args.num_workers, pin_memory=True,
                              generator=torch.Generator().manual_seed(902))
    val_loader = DataLoader(val, batch_size=args.batch_size, pin_memory=True)

    torch.manual_seed(500)
    source = args.resume or args.adapter
    policy, provenance = load_ira_lora(args.base, source, device="cuda",
                                       offload_text_tables=args.offload_text_tables)
    if provenance["base_sha256"] != BASE_SHA or (
        not args.resume and provenance["adapter_sha256"] != args.adapter_sha256
    ):
        raise ValueError("Unexpected base or starting adapter")
    pre, _ = make_processors(policy, source, args.tokenizer, "cuda")
    policy.get_base_model().model.gradient_checkpointing_enable()
    policy.train()
    trainable = [(name, param) for name, param in policy.named_parameters() if param.requires_grad]
    if not trainable or any("lora_" not in name for name, _ in trainable):
        raise ValueError("Expected only LoRA tensors to be trainable")
    params = [param for _, param in trainable]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    if args.resume or args.continue_optimizer:
        state = torch.load(source / "training_state.pt", map_location="cpu", weights_only=True)
        optimizer.load_state_dict(state["optimizer"])
        expected = metadata.get("previous_optimizer_updates", 0) + start_step // args.accumulate
        restored = max(int(slot["step"].item()) for slot in optimizer.state.values())
        if restored != expected:
            raise ValueError(f"Expected optimizer step {expected}, restored {restored}")
        torch.random.set_rng_state(state["cpu_rng"])
        torch.cuda.set_rng_state(state["cuda_rng"])

    def emit(tag, **details):
        print(tag, json.dumps(details), flush=True)

    emit("TRAIN_READY", **metadata, parameters=sum(p.numel() for p in params))
    if not args.resume:
        value, frames = validate(policy, pre, val_loader)
        write_json(args.output / "validation-000000.json", {"step": 0, "loss": value, "frames": frames})
        emit("VALIDATION", step=0, loss=value, frames=frames)

    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    begin = perf_counter()
    for step, batch in enumerate(train_loader, start=start_step + 1):
        loss, _ = policy(prepare(pre, batch))
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step}")
        (loss / args.accumulate).backward()
        running_loss += loss.detach().item()
        if step % args.accumulate == 0:
            grad = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        if step in (1, args.accumulate) or step % 100 == 0:
            emit("TRAIN_STEP", step=step, updates=step // args.accumulate,
                 loss=running_loss / (step - start_step),
                 grad_norm=grad.item() if step % args.accumulate == 0 else None,
                 max_gpu_mib=round(torch.cuda.max_memory_allocated() / 2**20),
                 seconds_per_step=(perf_counter() - begin) / (step - start_step))
        if step % args.save_freq == 0 or step == args.steps:
            value, frames = validate(policy, pre, val_loader)
            checkpoint = args.output / "checkpoints" / f"{step:06d}"
            checkpoint.parent.mkdir(exist_ok=True)
            info = {"config": metadata, "step": step, "optimizer_updates": step // args.accumulate,
                    "validation_loss": value, "validation_frames": frames}
            checksum = save_checkpoint(policy, optimizer, args.adapter, checkpoint, info)
            emit("CHECKPOINT", path=str(checkpoint), sha256=checksum, **info)
    emit("TRAIN_COMPLETE", step=args.steps, updates=args.steps // args.accumulate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "adapter", "tokenizer", "data", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--save-freq", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--offload-text-tables", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=190)
    parser.add_argument("--validation-episodes", type=int, default=10)
    parser.add_argument("--adapter-sha256", default=SOURCE_SHA)
    parser.add_argument("--continue-optimizer", action="store_true")
    parser.add_argument("--resume", type=Path)
    main(parser.parse_args())
