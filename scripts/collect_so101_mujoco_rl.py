"""Collect mixed-reward groups from a frozen PI05 policy in MuJoCo."""

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
from time import perf_counter
import uuid

import numpy as np
import torch

from evaluate_so101_mujoco_lora import BASE_SHA, JointMapping
from pi05_rl import FlowRLConfig, FlowTrace
from pi05_rl_episodes import EpisodeWriter, decode_record, label_episode, write_json
from pi05_rl_service import SamplerEngine
from so101_mujoco_blockbox import BOX_WALL_CONTACT_TOLERANCE, CAMERAS, SO101BlockBox

GROUP_SIZE = 8
SCENE_OFFSET = 31000
POLICY_OFFSET = 3100000


def independent_replay(engine, path):
    record = decode_record(path.read_bytes())
    batch = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in record["batch"].items()}
    trace = FlowTrace(**{
        k: v.to("cuda") if torch.is_tensor(v) else v for k, v in record["trace"].items()
    })
    with torch.no_grad():
        delta = engine.adapter.recompute(batch, trace) - trace.old_logprobs
        error = delta.abs().max().item()
        ratio_error = (delta.sum(dim=(1, 2)).exp() - 1).abs().max().item()
    if error > 1e-5 or ratio_error > 1e-3:
        raise RuntimeError(f"Serialized old logprob replay failed: {error}, {ratio_error}")
    return error, ratio_error


def rollout(env, engine, mapping, candidate, index, directory, reset_reference):
    scene_seed = SCENE_OFFSET + candidate
    policy_seed = POLICY_OFFSET + candidate * GROUP_SIZE + index
    torch.manual_seed(policy_seed)
    reset, _ = env.reset(seed=scene_seed)
    np.testing.assert_array_equal(reset, reset_reference)
    writer = EpisodeWriter(directory, group_id=f"group-{candidate:06d}",
                           scene_id=f"mujoco-{scene_seed}", health=engine.health,
                           source="sim", reset_state=reset.tolist())
    cube = env.data.qpos[env.cube_pos:env.cube_pos + 3]
    cube_track = [cube.copy()]
    actual = []
    terminated = truncated = False
    max_error = max_ratio = 0.0
    while not (terminated or truncated):
        q = env.data.qpos[env.position]
        rgb = [env.render_camera(cam) for cam in CAMERAS]
        record = engine.sample(mapping.observation(q), *rgb, uuid.uuid4().hex)
        check = record["replay_check"]
        max_error = max(max_error, check["max_abs_error"])
        max_ratio = max(max_ratio, check["max_ratio_error"])
        executed = clipped_steps = 0
        for command in record["physical_actions"][0].numpy():
            targets = mapping.radians(command)
            clipped = np.clip(targets, env.low, env.high)
            clipped_steps += int(np.any(np.abs(targets - clipped) > 1e-7))
            _, reward, terminated, truncated, info = env.step(env.action_for_targets(clipped))
            actual.append(clipped)
            cube_track.append(cube.copy())
            executed += 1
            if terminated or truncated:
                break
        writer.append(record, executed_steps=executed, clipped_steps=clipped_steps)
    assert env.steps == len(actual) == sum(c["executed_steps"] for c in writer.manifest["chunks"])
    assert reward == float(info["success"])
    error, ratio = independent_replay(engine, directory / writer.manifest["chunks"][0]["file"])
    np.savez_compressed(directory / "simulation.npz", sim_targets_rad=np.asarray(actual),
                        cube_xyz_m=np.asarray(cube_track), box_xyz_m=env.data.mocap_pos[env.box_mocap])
    writer.finish(execution_mode="mujoco-flow-sde", scene_seed=scene_seed,
                  policy_seed=policy_seed, simulated_commands=env.steps, trace_integrity="complete",
                  truncated=truncated, motor_commands_sent=0,
                  replay_max_abs_error=max(error, max_error), replay_max_ratio_error=max(ratio, max_ratio))
    label_episode(directory, "success" if info["success"] else "failure")
    return int(info["success"]), env.steps, len(writer.manifest["chunks"])


def collect(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "collector.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        collect_locked(args)


def collect_locked(args):
    accept = args.output / "accepted"
    reject = args.output / "rejected"
    pending = args.output / "pending"
    for path in (accept, reject, pending):
        path.mkdir(exist_ok=True)
    cfg = FlowRLConfig(num_steps=10, noise_level=0.5)
    torch.set_num_threads(1)
    engine = SamplerEngine(args.base, args.adapter, args.tokenizer, "cuda", cfg,
                           verify_replay=True, reuse_replay_prefix=True)
    provenance = engine.health["provenance"]
    if (provenance["base_sha256"], provenance["adapter_sha256"]) != (BASE_SHA, args.adapter_sha256):
        raise ValueError("Unexpected rollout weights")
    # Bind the dynamics, reward, asset and fixed calibration to this dataset.
    files = list(args.assets.rglob("*")) + [args.calibration,
                                            Path(__file__).with_name("so101_mujoco_blockbox.py")]
    digest = hashlib.sha256()
    for file in sorted(p for p in files if p.is_file()):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    protocol = {"groups_target": args.groups, "episodes_per_group": GROUP_SIZE,
                "horizon": 900, "physics_scene_sha256": digest.hexdigest(),
                "wall_tolerance_mm": BOX_WALL_CONTACT_TOLERANCE * 1000,
                "paper_sha256": "41580d8dea64f8ba34e0a6a3246a37464271d34e87e8b1432770f83cda0761fd",
                "dynamic_sampling": "retain groups with 1 through 7 successes",
                "temperature_adaptation": "continuous Flow-SDE noise_level=0.5",
                "scene_seed_offset": SCENE_OFFSET, "policy_seed_offset": POLICY_OFFSET,
                "sampler": engine.health}
    manifest = args.output / "config.json"
    if manifest.exists():
        if json.loads(manifest.read_bytes()) != protocol:
            raise ValueError("Cannot continue under a different rollout protocol")
    else:
        write_json(manifest, protocol)
    mapping = JointMapping(json.loads(args.calibration.read_bytes()))
    print("COLLECTOR_READY", engine.health["pipeline_id"], flush=True)
    past = sorted([*accept.glob("group-*/group.json"), *reject.glob("group-*.json")])
    ledger = [json.loads(path.read_bytes()) for path in past]
    candidate = max((row["candidate"] for row in ledger), default=-1) + 1
    kept = sum(row["accepted"] for row in ledger)
    start = perf_counter()

    def update_status():
        accepted_chunks = sum(sum(row["chunks"]) for row in ledger if row["accepted"])
        row = {
            "status": "complete" if kept == args.groups else "running",
            "groups_target": args.groups, "accepted_groups": kept, "accepted_episodes": kept * GROUP_SIZE,
            "accepted_chunks": accepted_chunks, "candidate_groups": len(ledger),
            "rejected_groups": len(ledger) - kept,
            "all_fail_groups": sum(sum(g["rewards"]) == 0 for g in ledger),
            "all_success_groups": sum(sum(g["rewards"]) == GROUP_SIZE for g in ledger),
            "sampled_episodes": len(ledger) * GROUP_SIZE,
            "last_group": ledger[-1] if ledger else None,
        }
        write_json(args.output / "progress.json", row)

    update_status()
    env = SO101BlockBox(args.assets, max_steps=protocol["horizon"])
    try:
        while kept < args.groups:
            number = f"group-{candidate:06d}"
            work = pending / number
            # An interrupted attempt is recomputed with its original deterministic seeds.
            if work.exists():
                shutil.rmtree(work)
            work.mkdir()
            reset, _ = env.reset(seed=SCENE_OFFSET + candidate)
            outcomes = []
            for index in range(GROUP_SIZE):
                value = rollout(env, engine, mapping, candidate, index,
                                work / f"episode-{index:02d}", reset)
                outcomes.append(value)
                print("EPISODE", candidate, index, *value, flush=True)
            rewards, steps, chunks = (list(column) for column in zip(*outcomes))
            accepted = 0 < sum(rewards) < GROUP_SIZE
            row = {"candidate": candidate, "scene_seed": SCENE_OFFSET + candidate,
                   "rewards": rewards, "steps": steps, "chunks": chunks, "accepted": accepted}
            if accepted:
                write_json(work / "group.json", row)
                work.replace(accept / number)
                kept += 1
            else:
                write_json(reject / f"{number}.json", row)
                shutil.rmtree(work)
            ledger.append(row)
            update_status()
            print("GROUP", kept, args.groups, json.dumps(row), "elapsed_s",
                  round(perf_counter()-start, 1), flush=True)
            candidate += 1
    finally:
        env.close()
    print("COLLECTION_COMPLETE", kept, kept * GROUP_SIZE, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("base", "adapter", "tokenizer", "output"):
        parser.add_argument(f"--{key}", required=True, type=Path)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--assets", type=Path, default=Path("menagerie"))
    parser.add_argument("--calibration", type=Path, default=Path("official_block_follower.json"))
    parser.add_argument("--groups", type=int, default=100)
    args = parser.parse_args()
    if args.groups < 1:
        parser.error("--groups must be positive")
    collect(args)
