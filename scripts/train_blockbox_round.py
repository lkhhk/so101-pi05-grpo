"""Collect fixed eight-lane MuJoCo rollouts and train with B8 and three CUDA streams."""

import argparse
from dataclasses import asdict
import fcntl
import gzip
import hashlib
import io
import json
from pathlib import Path
import shutil
from time import perf_counter

import mujoco
import numpy as np
import torch

from evaluate_so101_mujoco_lora import BASE_SHA, JointMapping, TASK
from pi05_grpo_streams import accumulate
from pi05_rl import FlowRLConfig, FlowTrace, LeRobotFlowRL, group_advantages, observation_digest
from pi05_rl_episodes import cpu_tree, write_json
from pi05_rl_runtime import load_ira_lora, make_processors
from pi05_rl_service import pipeline_identity
from so101_mujoco_blockbox import CAMERAS, SO101BlockBox
from train_so101_mujoco_grpo_streams import GROUPS_PER_UPDATE, save, update_schedule

LANES = 8
VALIDATION_GROUPS = 10
SCHEMA = "mujoco-flow-b8-v1"


def pack(path, record):
    buffer = io.BytesIO()
    torch.save(cpu_tree(record), buffer)
    payload = gzip.compress(buffer.getvalue(), compresslevel=1)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def read_record(path, checksum, cfg, behavior):
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != checksum:
        raise ValueError(f"Rollout hash mismatch: {path}")
    with gzip.GzipFile(fileobj=io.BytesIO(payload)) as stream:
        raw = stream.read(64 * 1024 * 1024 + 1)
    if len(raw) > 64 * 1024 * 1024:
        raise ValueError(f"Oversized rollout: {path}")
    r = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    trace = FlowTrace(**r["trace"])
    trace.validate(cfg, behavior)
    counts = torch.as_tensor(r["counts"])
    actions, actual = r["actions"], r["actual_rad"]
    if (r["schema"] != SCHEMA or trace.chains.shape[0] != LANES
            or trace.observation_digest != observation_digest(r["batch"])
            or any(x.shape[0] != LANES for x in r["batch"].values())
            or counts.shape != (LANES,) or bool(((counts < 0) | (counts > cfg.action_chunk)).any())
            or not bool(counts.any()) or actions.shape != (LANES, cfg.action_chunk, cfg.action_dim)
            or actual.shape != actions.shape or not torch.isfinite(actions).all()
            or not torch.isfinite(actual).all()):
        raise ValueError(f"Invalid fixed-batch record: {path}")
    return r


def physical_batch(envs, renderer, mapping, pre):
    frames = {camera: [] for camera in CAMERAS}
    states = []
    for env in envs:
        states.append(mapping.observation(env.data.qpos[env.position]))
        for camera in CAMERAS:
            renderer.update_scene(env.data, camera=camera, scene_option=env.view)
            frames[camera].append(renderer.render())
    batch = {"observation.state": torch.tensor(np.stack(states)), "task": [TASK] * LANES}
    for camera, rgb in frames.items():
        batch[f"observation.images.{camera}"] = (
            torch.from_numpy(np.stack(rgb)).permute(0, 3, 1, 2).float() / 255
        )
    return pre(batch)


@torch.no_grad()
def collect_one(folder, candidate, envs, renderer, mapping, adapter, pre, post):
    folder.mkdir()
    seed = SCENE_OFFSET + candidate
    generator = torch.Generator(device="cuda").manual_seed(POLICY_OFFSET + candidate)
    resets = [env.reset(seed=seed)[0] for env in envs]
    for reset in resets[1:]:
        np.testing.assert_array_equal(reset, resets[0])
    rewards = [0] * LANES
    done = [False] * LANES
    manifest = []
    max_replay = max_ratio_error = 0.0
    while not all(done):
        batch = physical_batch(envs, renderer, mapping, pre)
        context = adapter.prefix(batch)
        actions, trace = adapter.sample_from_prefix(batch, context, generator=generator)
        replay = adapter.recompute_from_prefix(batch, trace, context)
        delta = replay - trace.old_logprobs
        error = delta.abs().max().item()
        ratio_error = (delta.sum((1, 2)).exp() - 1).abs().max().item()
        if error > 1e-5 or ratio_error > 1e-3:
            raise RuntimeError(f"B8 old-policy replay mismatch {error}, {ratio_error}")
        max_replay = max(max_replay, error)
        max_ratio_error = max(max_ratio_error, ratio_error)
        actions = post(actions.clone()).detach().cpu()
        if actions.shape != (LANES, 50, 6) or not torch.isfinite(actions).all():
            raise ValueError("Invalid action batch")
        actual = np.zeros(actions.shape, dtype=np.float64)
        counts = [0] * LANES
        for lane, env in enumerate(envs):
            if done[lane]:
                continue
            for frame, command in enumerate(actions[lane].numpy()):
                control = np.clip(mapping.radians(command), env.low, env.high)
                _, _, success, timeout, _ = env.step(env.action_for_targets(control))
                actual[lane, frame] = control
                counts[lane] += 1
                if success or timeout:
                    rewards[lane] = int(success)
                    done[lane] = True
                    break
        path = folder / f"chunk-{len(manifest):03d}.pt.gz"
        record = {"schema": SCHEMA, "batch": {k: v for k, v in batch.items() if k.startswith("observation.")},
                  "trace": asdict(trace), "actions": actions, "actual_rad": torch.from_numpy(actual),
                  "counts": counts}
        checksum = pack(path, record)
        manifest.append({"file": path.name, "sha256": checksum, "counts": counts})
    steps = [e.steps for e in envs]
    assert steps == [sum(x["counts"][lane] for x in manifest) for lane in range(LANES)]
    return {"candidate": candidate, "scene_seed": seed, "policy_seed": POLICY_OFFSET + candidate,
            "rewards": rewards, "steps": steps, "reset_state": resets[0].tolist(),
            "accepted": 0 < sum(rewards) < LANES, "records": manifest,
            "old_logprob_max_error": max_replay, "old_ratio_max_error": max_ratio_error}


def host_group(path, adapter):
    group = json.loads((path / "group.json").read_bytes())
    rewards = torch.tensor(group["rewards"])
    advantages, keep = group_advantages(rewards, torch.zeros(LANES, dtype=torch.long), LANES)
    if not keep.all():
        raise ValueError("Invalid accepted reward group")
    steps = [sum(r["counts"][i] for r in group["records"]) for i in range(LANES)]
    if steps != group["steps"]:
        raise ValueError(f"Executed coverage changed for {path}")
    visits = torch.tensor([sum(r["counts"][i] > 0 for r in group["records"]) for i in range(LANES)])
    if not bool((visits > 0).all()):
        raise ValueError("An episode has no active actions")
    finished = torch.zeros(LANES, dtype=torch.bool)
    used = []
    for item in group["records"]:
        record = read_record(path / item["file"], item["sha256"], adapter.cfg, adapter.policy_version)
        if record["counts"] != item["counts"]:
            raise ValueError("Execution counts differ from manifest")
        count = torch.tensor(record["counts"])
        if bool((finished & (count > 0)).any()):
            raise ValueError("A finished lane became active again")
        finished |= count < adapter.cfg.action_chunk
        obs = {key: value.pin_memory() for key, value in record["batch"].items()}
        trace = {key: value.pin_memory() if torch.is_tensor(value) else value
                 for key, value in record["trace"].items()}
        weights = (count > 0).float() / (LANES * visits)
        mask = torch.arange(adapter.cfg.action_chunk)[None, :] < count[:, None]
        used.append((obs, trace, advantages.pin_memory(), weights.pin_memory(),
                     mask.pin_memory(), int(count.count_nonzero())))
    return used


def replay_gate(adapter, path, streams):
    stats = accumulate(adapter, host_group(path, adapter), streams)
    grads = [p.grad for p in adapter.parameters() if p.requires_grad and p.grad is not None]
    if (stats["old_logprob_max_error"] > 1e-5 or stats["old_ratio_max_error"] > 1e-3
            or not grads or not all(torch.isfinite(g).all() for g in grads)
            or not any(torch.count_nonzero(g) for g in grads)):
        raise RuntimeError(f"Full group B8/three-stream replay gate failed: {stats}")
    adapter.zero_grad(set_to_none=True)
    return stats


def collect(args, root, adapter, pre, post, streams):
    accept, reject, pending = (root / name for name in ("accepted", "rejected", "pending"))
    for path in (accept, reject, pending):
        path.mkdir(exist_ok=True)
    logs = sorted([*accept.glob("group-*/group.json"), *reject.glob("group-*.json")])
    ledger = [json.loads(path.read_bytes()) for path in logs]
    candidate = max((g["candidate"] for g in ledger), default=-1) + 1
    kept = sum(g["accepted"] for g in ledger)
    if kept >= args.groups:
        return accept
    envs = [SO101BlockBox(args.assets) for _ in range(LANES)]
    for env in envs[1:]:
        env.model = envs[0].model
        env.data = mujoco.MjData(env.model)
    renderer = mujoco.Renderer(envs[0].model, height=240, width=320)
    mapping = JointMapping(json.loads(args.calibration.read_bytes()))
    try:
        while kept < args.groups:
            if candidate >= 1000:
                raise RuntimeError("Round exceeded its unique scene-seed allocation")
            if shutil.disk_usage(root).free < 4 * 2**30:
                raise RuntimeError("Insufficient disk space to continue collection")
            name = f"group-{candidate:06d}"
            work = pending / name
            if work.exists():
                shutil.rmtree(work)
            t = perf_counter()
            row = collect_one(work, candidate, envs, renderer, mapping, adapter, pre, post)
            if row["accepted"]:
                write_json(work / "group.json", row)
                if kept == 0:
                    gate = replay_gate(adapter, work, streams)
                    write_json(root / "pilot-replay.json", gate)
                    print("PILOT_REPLAY_OK", json.dumps(gate), flush=True)
                work.replace(accept / name)
                kept += 1
            else:
                write_json(reject / f"{name}.json", {k: v for k, v in row.items() if k != "records"})
                shutil.rmtree(work)
            ledger.append(row)
            progress = {"phase": "collecting", "accepted_groups": kept, "groups_target": args.groups,
                        "accepted_episodes": kept * LANES, "candidate_groups": len(ledger),
                        "rejected_groups": len(ledger) - kept, "last_rewards": row["rewards"],
                        "last_group_seconds": perf_counter() - t, "training_updates": 0}
            write_json(root / "progress.json", progress)
            print("GROUP", json.dumps(progress), flush=True)
            candidate += 1
    finally:
        renderer.close()
        for env in envs:
            env.close()
    return accept


def train(args, root, accepted, behavior, adapter, streams):
    groups = sorted(p.parent for p in accepted.glob("group-*/group.json"))
    if len(groups) != args.groups:
        raise ValueError("Unexpected number of fresh rollout groups")
    schedule = update_schedule([p.name for p in groups], args.steps, SCENE_OFFSET)
    write_json(root / "schedule.json", schedule)
    params = [p for p in adapter.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    checkpoints = root / "checkpoints"
    existing = sorted(p.parent for p in checkpoints.glob("[0-9]*/rl_training.json"))
    source = existing[-1] if existing else args.adapter
    state = json.loads((source / "rl_training.json").read_bytes())
    if existing and state["run_config"] != behavior:
        raise ValueError("Resume checkpoint differs from B8 run")
    completed = state["updates"] - START_STEP
    if not 0 <= completed <= args.steps or (not existing and completed != 0):
        raise ValueError("Invalid B8 continuation checkpoint step")
    optimizer.load_state_dict(torch.load(source / "optimizer.pt", map_location="cpu", weights_only=True))
    counters = {int(item["step"].item()) for item in optimizer.state.values()}
    if counters != {START_STEP + completed} or len(optimizer.state) != len(params):
        raise ValueError(f"B8 optimizer state does not match checkpoint: {counters}")
    if any(group["lr"] != args.lr for group in optimizer.param_groups):
        raise ValueError("B8 optimizer learning rate differs from requested learning rate")
    groups_by_name = {p.name: p for p in groups}
    # Cache verified CPU records for the most recently read groups. Pinned
    # buffers are bounded independently of the total 100-group dataset.
    from functools import lru_cache
    get_group = lru_cache(maxsize=12)(lambda name: host_group(groups_by_name[name], adapter))
    print("B8_TRAIN_READY", START_STEP + completed, START_STEP + args.steps,
          "train_groups", len(groups), "validation_groups", VALIDATION_GROUPS, flush=True)
    for step, keys in enumerate(schedule[completed:], start=completed + 1):
        rows = []
        for name in keys:
            for obs, trace, adv, weight, mask, count in get_group(name):
                rows.append((obs, trace, adv, (weight / GROUPS_PER_UPDATE).pin_memory(), mask, count))
        stats = accumulate(adapter, rows, streams)
        if step == 1 and (stats["old_logprob_max_error"] > 1e-5
                          or stats["old_ratio_max_error"] > 1e-3):
            raise RuntimeError(f"B8 initial replay verification failed: {stats}")
        norm = torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True).item()
        if norm == 0:
            raise RuntimeError("B8 update has zero gradient")
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in params):
            raise ValueError("Non-finite LoRA parameter")
        progress = {"phase": "training" if step < args.steps else "complete", "training_updates": step,
                    "training_target": args.steps, "total_updates": START_STEP + step,
                    "round_end_step": START_STEP + args.steps, "training_groups": len(groups),
                    "validation_groups": VALIDATION_GROUPS, "grad_norm": norm, **stats}
        if step % args.save_every == 0 or step == args.steps:
            total_updates = START_STEP + step
            checkpoint = checkpoints / f"{total_updates:06d}"
            metadata = {"schema": SCHEMA, "run_config": behavior, "updates": total_updates,
                        "continuation_updates": step, "data_sources": ["sim"],
                        "stats": progress, "test_only": False}
            progress["checkpoint_sha256"] = save(adapter.policy, optimizer, args.adapter, checkpoint, metadata)
            print("CHECKPOINT", total_updates, progress["checkpoint_sha256"], flush=True)
        write_json(root / "progress.json", progress)
        print("B8_GRPO_STEP", json.dumps(progress), flush=True)


def main(args):
    torch.set_num_threads(1)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free, total = torch.cuda.mem_get_info()
        torch.cuda.set_per_process_memory_fraction((free - 1536 * 2**20) / total)
        cfg = FlowRLConfig(num_steps=10, noise_level=0.5)
        # Sample from the frozen checkpoint. A resume loads updated weights only
        # once the collection is complete and its binding has been verified.
        existing = sorted(p.parent for p in (root / "checkpoints").glob("[0-9]*/rl_training.json"))
        source = existing[-1] if existing else args.adapter
        policy, provenance = load_ira_lora(args.base, source, device="cuda")
        if provenance["base_sha256"] != BASE_SHA:
            raise ValueError("Unexpected base model")
        if existing:
            meta = json.loads((source / "rl_training.json").read_bytes())
            initial = meta["run_config"]["provenance"]
        else:
            initial = provenance
        if initial["adapter_sha256"] != args.adapter_sha256:
            raise ValueError("Unexpected frozen sampling adapter")
        identity, _ = pipeline_identity(args.adapter, initial, cfg)
        behavior = {"schema": SCHEMA, "layout": "fixed 8 lanes, terminal lanes masked",
                    "provenance": initial, "pipeline_id": identity, "sampler": cfg.serialized(),
                    "groups": args.groups, "steps": args.steps, "save_every": args.save_every,
                    "lr": args.lr, "streams": 3, "groups_per_update": GROUPS_PER_UPDATE,
                    "scene_seed_offset": SCENE_OFFSET, "policy_seed_offset": POLICY_OFFSET,
                    "start_step": START_STEP, "validation_groups": VALIDATION_GROUPS,
                    "validation_manifest": str(args.validation_manifest.resolve()),
                    "validation_manifest_sha256": hashlib.sha256(args.validation_manifest.read_bytes()).hexdigest(),
                    "starting_optimizer_sha256": hashlib.sha256((args.adapter / "optimizer.pt").read_bytes()).hexdigest(),
                    "calibration_sha256": hashlib.sha256(args.calibration.read_bytes()).hexdigest(),
                    "environment_sha256": hashlib.sha256(
                        Path(__file__).with_name("so101_mujoco_blockbox.py").read_bytes()).hexdigest()}
        config = root / "config.json"
        if config.exists() and json.loads(config.read_bytes()) != behavior:
            raise ValueError("B8 run configuration changed")
        write_json(config, behavior)
        adapter = LeRobotFlowRL(policy, cfg, policy_version=initial["policy_version"])
        streams = (torch.cuda.Stream(), torch.cuda.Stream())
        print("B8_READY", identity, flush=True)
        if existing:
            accepted = root / "accepted"
        else:
            pre, post = make_processors(policy, args.adapter, args.tokenizer, "cuda")
            accepted = collect(args, root, adapter, pre, post, streams)
        train(args, root, accepted, behavior, adapter, streams)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("base", "adapter", "tokenizer", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--assets", type=Path, default=Path("menagerie"))
    parser.add_argument("--calibration", type=Path, default=Path("official_block_follower.json"))
    parser.add_argument("--groups", type=int, default=100)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--scene-offset", type=int, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    args = parser.parse_args()
    START_STEP = args.start_step
    SCENE_OFFSET = args.scene_offset
    POLICY_OFFSET = SCENE_OFFSET * 100
    if args.groups < GROUPS_PER_UPDATE or args.steps < 1 or args.save_every < 1 or args.lr <= 0:
        parser.error("Expected at least four groups and positive training settings")
    main(args)
