"""Train an episode batch, then export a separate LoRA/optimizer checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import uuid

import torch

from pi05_rl import FlowRLConfig, FlowTrace, LeRobotFlowRL, grpo_loss
from pi05_rl_episodes import training_set, write_json
from pi05_rl_runtime import load_ira_lora, sha256
from pi05_rl_service import pipeline_identity


def to_device_record(record, device):
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in record["batch"].items()}
    values = {k: v.to(device) if torch.is_tensor(v) else v for k, v in record["trace"].items()}
    return batch, FlowTrace(**values)


def batch_digest(episodes):
    manifests = sorted((m for m,_,_ in episodes),key=lambda m:m["episode_id"])
    return hashlib.sha256(json.dumps(manifests,sort_keys=True).encode()).hexdigest()


def update_episodes(adapter, episodes, optimizer, *, device="cpu", check_replay=True):
    """One accumulated update. Episodes equal weight; chunks equal weight within each episode."""
    optimizer.zero_grad(set_to_none=True)
    total, max_error, max_ratio_error, chunks = 0., 0., 0., 0
    for manifest, rows, advantage in episodes:
        for row, record in rows:
            batch, trace = to_device_record(record, device)
            new = adapter.recompute(batch, trace)
            start = row.get("execution_start", 0)
            end = start + row["executed_steps"]
            if check_replay:
                delta = new.detach()[:, start:end]-trace.old_logprobs[:, start:end]
                error = float(delta.abs().max())
                ratio_error = float((delta.sum((1,2)).exp()-1).abs().max())
                max_error, max_ratio_error = max(max_error, error), max(max_ratio_error, ratio_error)
                if error > 1e-5 or ratio_error > 1e-3:
                    optimizer.zero_grad(set_to_none=True)
                    raise ValueError("Behavior logprobs fail replay; check runtime, model and preprocessing")
            # RTC skips already elapsed positions; the terminal partial chunk also drops its tail.
            positions = torch.arange(adapter.cfg.action_chunk, device=device)[None,:,None]
            mask = (positions >= start) & (positions < end)
            loss, _ = grpo_loss(new*mask, trace.old_logprobs*mask,
                                torch.tensor([advantage], device=device),
                                torch.ones(1, dtype=torch.bool, device=device), adapter.cfg)
            weighted = loss / (len(episodes)*len(rows))
            weighted.backward()
            total += float(weighted.detach()); chunks += 1
    params = [p for p in adapter.policy.parameters() if p.requires_grad]
    gradients = [p.grad for p in params if p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise ValueError("Missing/non-finite gradients")
    norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
    if not norm > 0: raise ValueError("Zero gradient; no update exported")
    optimizer.step()
    if not all(torch.isfinite(p).all() for p in params): raise ValueError("Non-finite updated parameters")
    return {"loss": total, "grad_norm": float(norm), "episodes": len(episodes), "chunks": chunks,
            "replay_checked": check_replay,
            "replay_max_abs_error": max_error if check_replay else None,
            "replay_max_ratio_error": max_ratio_error if check_replay else None}


def export_checkpoint(policy, optimizer, source_adapter, output, training):
    """Publish a new directory only after all weights/metadata have been saved."""
    output, source_adapter = Path(output), Path(source_adapter)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(output.name+".staging-"+uuid.uuid4().hex)
    staging.mkdir()
    policy.save_pretrained(str(staging), safe_serialization=True, save_embedding_layers=False)
    base_policy = policy.get_base_model()
    base_policy.config.device = "cpu"
    base_policy.config.use_peft = True
    base_policy.config.save_pretrained(staging)
    for file in source_adapter.glob("policy_*"):
        if file.is_file(): shutil.copy2(file, staging/file.name)
    torch.save(optimizer.state_dict(), staging/"optimizer.pt")
    write_json(staging/"rl_training.json", training)
    staging.rename(output)
    return sha256(output/"adapter_model.safetensors")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=Path, required=True)
    p.add_argument("--base", type=Path, required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--adapter", type=Path)
    inputs.add_argument("--resume", type=Path, help="Continue updates on the exact same recorded behavior batch")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--max-steps", type=int, help="Stop at this total optimizer step, including resumed steps")
    p.add_argument("--save-every", type=int, help="Save checkpoints under --output at this step interval and at the final step")
    p.add_argument("--allow-test-data", action="store_true")
    p.add_argument("--allow-sim-data", action="store_true", help="Accept MuJoCo rollout groups")
    args = p.parse_args(); torch.set_num_threads(args.threads)
    if args.max_steps is not None and args.max_steps < 1:
        p.error("--max-steps must be positive")
    if args.save_every is not None and args.save_every < 1:
        p.error("--save-every must be positive")
    if args.output.exists(): raise FileExistsError(args.output)
    episodes = training_set(args.episodes, args.group_size, allow_test=args.allow_test_data,
                            allow_sim=args.allow_sim_data)
    health = episodes[0][0]["health"]
    cfg = FlowRLConfig(**health["config"])
    source_adapter = args.resume or args.adapter
    policy, provenance = load_ira_lora(args.base, source_adapter, device=args.device,
                                      allow_test_checkpoint=args.allow_test_data)
    previous = None
    digest = batch_digest(episodes)
    if args.resume:
        previous = json.loads((args.resume/"rl_training.json").read_text(encoding="utf-8"))
        if (previous["batch_digest"] != digest or
            previous["pipeline_id"] != health["pipeline_id"] or previous["config"] != cfg.serialized() or
            previous["parent"]["base_sha256"] != provenance["base_sha256"]):
            raise ValueError("Resume checkpoint does not belong to this exact behavior batch/base/config")
    behavior = previous["parent"] if previous else provenance
    if behavior["policy_version"] != health["policy_version"]:
        raise ValueError("Training input weights are not the sampled behavior policy")
    pipeline_id, _ = pipeline_identity(source_adapter, behavior, cfg)
    if pipeline_id != health["pipeline_id"]: raise ValueError("Processor/sampler pipeline differs")
    adapter = LeRobotFlowRL(policy, cfg, policy_version=behavior["policy_version"])
    optimizer = torch.optim.AdamW([p for p in policy.parameters() if p.requires_grad], lr=args.lr, weight_decay=0)
    if previous:
        optimizer.load_state_dict(torch.load(args.resume/"optimizer.pt",map_location=args.device,weights_only=True))
        for group in optimizer.param_groups: group["lr"] = args.lr
    completed = previous["updates"] if previous else 0
    max_steps = args.max_steps if args.max_steps is not None else completed+1
    if max_steps <= completed:
        p.error("--max-steps must exceed the resumed optimizer step")
    if args.save_every is not None:
        args.output.mkdir(parents=True, exist_ok=False)
    training = {"schema": "pi05-flow-update-v1", "parent": behavior, "pipeline_id": pipeline_id,
                "config": cfg.serialized(), "group_size": args.group_size,
                "test_only": any(m["source"] not in ("real", "sim") for m, _, _ in episodes),
                "data_sources": sorted({m["source"] for m, _, _ in episodes}),
                "episodes": [m["episode_id"] for m, _, _ in episodes], "lr": args.lr,
                "batch_digest": digest, "resumed_from":str(args.resume) if args.resume else None,
                "max_steps": max_steps, "save_every": args.save_every,
                "claim": "episode-weighted GRPO updates; no deployment evaluation"}
    for step in range(completed+1, max_steps+1):
        stats = update_episodes(adapter, episodes, optimizer, device=args.device, check_replay=step==1)
        training.update(updates=step, stats=stats)
        print(json.dumps({"step": step, **stats}), flush=True)
        if step == max_steps or (args.save_every is not None and step % args.save_every == 0):
            output = args.output if args.save_every is None else args.output/f"{step:06d}"
            adapter_digest = export_checkpoint(policy, optimizer, source_adapter, output, training)
            print(json.dumps({"checkpoint": str(output), "adapter_sha256": adapter_digest, **training}, indent=2), flush=True)


if __name__ == "__main__": main()
