"""Run grouped, episode-weighted GRPO with singleton chunks and three CUDA streams."""

import argparse
from collections import deque
import fcntl
import hashlib
import json
from pathlib import Path
import random

import torch

from pi05_grpo_streams import accumulate, host_batches
from pi05_rl import FlowRLConfig, LeRobotFlowRL
from pi05_rl_episodes import training_set, write_json
from pi05_rl_runtime import load_ira_lora
from pi05_rl_service import pipeline_identity
from train_pi05_rl import export_checkpoint

GROUP_SIZE = 8
GROUPS_PER_UPDATE = 4


def choose_groups(dataset, count, seed):
    candidates = sorted(p.parent for p in dataset.glob("group-*/group.json"))
    if len(candidates) < count:
        raise ValueError("Not enough committed groups")
    return sorted(random.Random(seed).sample(candidates, count))


def update_schedule(groups, updates, seed):
    rng = random.Random(seed)
    pending = deque()
    batches = []
    for _ in range(updates):
        if len(pending) < GROUPS_PER_UPDATE:
            order = list(groups)
            rng.shuffle(order)
            # Fill an update across an epoch boundary without repeating a group in it.
            pending.extend([name for name in order if name not in pending] +
                           [name for name in order if name in pending])
        batches.append([pending.popleft() for _ in range(GROUPS_PER_UPDATE)])
    return batches


def inspect_index(groups):
    digest = hashlib.sha256()
    identity = None
    episode_ids = set()
    chunks = successes = 0
    for group in groups:
        paths = sorted(group.glob("episode-*/episode.json"))
        if len(paths) != GROUP_SIZE:
            raise ValueError(f"Incomplete group {group}")
        group_rows = []
        for path in paths:
            data = path.read_bytes()
            digest.update(data)
            row = json.loads(data)
            if row["episode_id"] in episode_ids or row["group_id"] != group.name:
                raise ValueError(f"Episode identity mismatch: {path}")
            if (row["source"] != "sim" or row["phase"] != "complete"
                    or row.get("trace_integrity") != "complete" or row["reward"] not in (0, 1)
                    or row["reward"] != int(row["outcome"] == "success")
                    or not row["chunks"] or sum(b["executed_steps"] for b in row["chunks"])
                    != row["simulated_commands"]):
                raise ValueError(f"Invalid simulator episode: {path}")
            episode_ids.add(row["episode_id"])
            identity = identity or row["health"]
            if identity != row["health"]:
                raise ValueError(f"Mixed behavior policy: {path}")
            group_rows.append(row)
            chunks += len(row["chunks"])
        if (len({json.dumps((r["scene_seed"], r["reset_state"])) for r in group_rows}) != 1
                or len({r["policy_seed"] for r in group_rows}) != GROUP_SIZE):
            raise ValueError(f"Inconsistent group reset or policy seeds: {group}")
        count = sum(r["reward"] for r in group_rows)
        if not 0 < count < GROUP_SIZE:
            raise ValueError(f"Homogeneous rewards in accepted group {group}")
        successes += count
    return digest.hexdigest(), identity, chunks, successes


def save(policy, optimizer, source, path, state):
    cfg = policy.get_base_model().config
    device, use_peft = cfg.device, cfg.use_peft
    try:
        return export_checkpoint(policy, optimizer, source, path, state)
    finally:
        cfg.device, cfg.use_peft = device, use_peft


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "training.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        train(args)


def train(args):
    torch.set_num_threads(1)
    groups = choose_groups(args.episodes, args.groups, args.seed)
    names = [path.name for path in groups]
    digest, behavior, chunks, successes = inspect_index(groups)
    cfg = FlowRLConfig(**behavior["config"])
    if cfg.rtc:
        raise ValueError("Three-stream trainer requires synchronous simulator rollouts")
    schedule = update_schedule(names, args.steps, args.seed + 1)
    meta = {
        "schema": "pi05-mujoco-grpo-streams-v1", "group_size": GROUP_SIZE,
        "groups_per_update": GROUPS_PER_UPDATE, "selected_groups": names,
        "selected_episodes": len(names) * GROUP_SIZE, "selected_chunks": chunks,
        "selected_successes": successes, "episode_manifest_sha256": digest,
        "pipeline_id": behavior["pipeline_id"], "policy_version": behavior["policy_version"],
        "sampler_config": behavior["config"], "seed": args.seed,
        "steps": args.steps, "save_every": args.save_every, "lr": args.lr,
        "chunk_batch": 1, "streams": 3, "data_sources": ["sim"],
    }
    migration_step = 0
    if args.continue_from:
        prior = json.loads((args.continue_from / "rl_training.json").read_bytes())
        previous_meta = prior["run_config"]
        changed = {key for key in meta if meta[key] != previous_meta.get(key)}
        if changed - {"steps", "save_every"}:
            raise ValueError(f"Cannot continue with changed behavior or training configuration: {changed}")
        migration_step = prior["updates"]
        if not 0 < migration_step < args.steps:
            raise ValueError("Continuation checkpoint is at or beyond the requested target")
        original_schedule = json.loads(
            (args.continue_from.parent.parent / "schedule.json").read_bytes()
        )
        if schedule[:migration_step] != original_schedule[:migration_step]:
            raise ValueError("Continuation did not preserve update order")
        meta["continued_from"] = str(args.continue_from.resolve())
        meta["starting_update"] = migration_step

    config = args.output / "config.json"
    if config.exists():
        if not args.resume or json.loads(config.read_bytes()) != meta:
            raise ValueError("Existing run requires --resume with its original settings")
    else:
        if args.resume:
            raise ValueError("Cannot resume without run metadata")
        write_json(config, meta)
        write_json(args.output / "schedule.json", schedule)

    checkpoints = args.output / "checkpoints"
    complete = ordered_checkpoints(checkpoints)
    source = args.continue_from or args.adapter
    completed = migration_step
    if complete:
        source = complete[-1]
        saved = json.loads((source / "rl_training.json").read_bytes())
        if saved["run_config"] != meta:
            raise ValueError("Checkpoint run metadata mismatch")
        completed = saved["updates"]
        if completed == args.steps:
            print("TRAIN_COMPLETE", completed, flush=True)
            return

    policy, provenance = load_ira_lora(args.base, source, device="cuda")
    parent = behavior["provenance"]
    if provenance["base_sha256"] != parent["base_sha256"]:
        raise ValueError("Base checkpoint mismatch")
    if not completed and provenance["policy_version"] != behavior["policy_version"]:
        raise ValueError("Start from the exact behavior checkpoint")
    identity, _ = pipeline_identity(source, parent, cfg)
    if identity != behavior["pipeline_id"]:
        raise ValueError("Processor or sampler mismatch")
    adapter = LeRobotFlowRL(policy, cfg, policy_version=behavior["policy_version"])
    params = [param for param in policy.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=0)
    if completed:
        optimizer.load_state_dict(torch.load(source / "optimizer.pt", map_location="cuda",
                                             weights_only=True))
        restored = max(int(item["step"].item()) for item in optimizer.state.values())
        if restored != completed:
            raise ValueError(f"Optimizer counters restored to {restored}, expected {completed}")
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    groups_by_name = {path.name: path for path in groups}
    cached = {}
    print("TRAIN_READY", json.dumps(meta), flush=True)
    for update, names_for_update in enumerate(schedule[completed:], start=completed + 1):
        inputs = []
        for name in names_for_update:
            if name not in cached:
                group = training_set(groups_by_name[name], GROUP_SIZE, allow_sim=True)
                if len(group) != GROUP_SIZE:
                    raise ValueError(f"Incorrect group length: {name}")
                cached[name] = group
            for _, rows, advantage in cached[name]:
                weight = 1 / (GROUPS_PER_UPDATE * GROUP_SIZE * len(rows))
                inputs.extend((row, record, advantage, weight) for row, record in rows)
        stats = accumulate(adapter, host_batches(inputs, 1), streams)
        if update == 1 and (stats["old_logprob_max_error"] > 1e-5
                            or stats["old_ratio_max_error"] > 1e-3):
            raise ValueError("Initial old-policy replay failed")
        if not any(param.grad is not None for param in params):
            raise ValueError("GRPO produced no gradients")
        norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
        if norm.item() == 0:
            raise ValueError("Zero gradient")
        optimizer.step()
        if not all(torch.isfinite(param).all() for param in params):
            raise ValueError("Non-finite updated LoRA parameter")
        stat = {"update": update, "target": args.steps, "groups": names_for_update,
                "initial_replay_check": update == 1, "grad_norm": norm.item(), **stats}
        print("GRPO_STEP", json.dumps(stat), flush=True)

        if update % args.save_every == 0 or update == args.steps:
            checkpoint = checkpoints / f"{update:06d}"
            state = {"run_config": meta, "parent": parent, "updates": update,
                     "stats": stat, "test_only": False, "data_sources": ["sim"]}
            checksum = save(policy, optimizer, args.adapter, checkpoint, state)
            stat["latest_checkpoint"] = str(checkpoint)
            stat["checkpoint_adapter_sha256"] = checksum
            print("CHECKPOINT", update, checksum, flush=True)
        write_json(args.output / "progress.json", stat)
    print("TRAIN_COMPLETE", args.steps, flush=True)


def ordered_checkpoints(path):
    return sorted(p for p in path.glob("[0-9]*") if (p / "rl_training.json").is_file())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "adapter", "episodes", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--groups", type=int, default=50)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20261122)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-from", type=Path)
    args = parser.parse_args()
    if args.groups < GROUPS_PER_UPDATE or args.steps < 1 or args.save_every < 1 or args.lr <= 0:
        parser.error("Expected at least four groups and positive steps, save interval and learning rate")
    run(args)
