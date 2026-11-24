"""Collect successful scripted SO-101 MuJoCo episodes and a fixed episode split."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from demo_so101_mujoco_blockbox import arm_targets, radial_direction
from evaluate_so101_mujoco_lora import JointMapping, REFERENCE, TASK
from so101_mujoco_blockbox import CAMERAS, JOINTS, SO101BlockBox


def attempt(env, kin, seed, jaw_closed):
    env.reset(seed=seed)
    cube = env.data.qpos[env.cube_pos:env.cube_pos + 3].copy()
    box = env.data.mocap_pos[env.box_mocap].copy()
    start = env.data.qpos[env.position].copy()
    grasp = np.r_[cube[:2], 0.024]
    grasp -= 0.005 * radial_direction(grasp, env.model.body("shoulder").pos)
    plan = (
        (grasp + [0, 0, 0.06], 0, 0.005, 0.60, 40),
        (grasp, 0, 0.005, 0.60, 40),
        (grasp, 0, 0.005, jaw_closed, 30),
        (grasp + [0, 0, 0.08], 0, 0.005, jaw_closed, 40),
        (box + [0, 0, 0.124], 30, 0, jaw_closed, 55),
        (box + [0, 0, 0.124], 30, 0, 0.60, 50),
    )
    targets = []
    peak = cube[2]
    previous = start
    for point, pitch, depth, jaw, count in plan:
        target = np.r_[arm_targets(env, kin, point, np.deg2rad(pitch),
                                   depth, previous[:5]), jaw]
        for alpha in np.linspace(0, 1, count):
            command = (1 - alpha) * previous + alpha * target
            targets.append(command)
            _, _, success, _, _ = env.step(env.action_for_targets(command))
            peak = max(peak, env.data.qpos[env.cube_pos + 2])
            if success:
                return np.array(targets), cube, box, float(peak)
        previous = target
    return None


def save_episode(env, mapping, output, seed, targets):
    env.reset(seed=seed)
    images = np.lib.format.open_memmap(output / "rgb.npy", mode="w+", dtype=np.uint8,
                                       shape=(len(targets), len(CAMERAS), 240, 320, 3))
    states, actions = [], []
    for index, target in enumerate(targets):
        states.append(mapping.observation(env.data.qpos[env.position]))
        action = mapping.observation(target)
        actions.append(action)
        for camera_id, name in enumerate(CAMERAS):
            images[index, camera_id] = env.render_camera(name)
        np.testing.assert_allclose(mapping.radians(action), target, atol=2e-6)
        _, _, success, _, _ = env.step(env.action_for_targets(target))
        if success != (index == len(targets) - 1):
            raise RuntimeError(f"Seed {seed}: replay did not match accepted trajectory")
    images.flush()
    np.savez(output / "motion.npz", state=np.array(states), action=np.array(actions))


def main(args):
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = JointMapping(json.loads(args.calibration.read_text()))
    closed_state = REFERENCE.copy()
    closed_state[-1] = 0
    jaw_closed = mapping.radians(closed_state)[-1]
    env = SO101BlockBox(args.assets)
    env.reset(seed=11)
    kin = mujoco.MjData(env.model)
    episodes, failed = [], []
    try:
        for seed in range(args.seed_start, args.seed_start + 600):
            if len(episodes) == args.count:
                break
            try:
                result = attempt(env, kin, seed, jaw_closed)
            except RuntimeError as error:
                if "out of reach" not in str(error):
                    raise
                failed.append(seed)
                continue
            if result is None or result[3] < 0.045:
                failed.append(seed)
                continue
            targets, cube, box, peak = result
            episode = len(episodes)
            name = f"episode-{episode:03d}"
            directory = args.output / name
            directory.mkdir()
            save_episode(env, mapping, directory, seed, targets)
            row = {"episode": episode, "directory": name, "seed": seed, "frames": len(targets),
                   "cube_start_m": cube.tolist(), "box_center_m": box.tolist(),
                   "cube_peak_z_m": peak, "success": True}
            episodes.append(row)
            print(json.dumps(row), flush=True)
        if len(episodes) != args.count:
            raise RuntimeError(f"Collected {len(episodes)} successful episodes, expected {args.count}")
        order = np.random.default_rng(20261121).permutation(len(episodes))
        validation = sorted(order[:args.validation].tolist())
        train = sorted(order[args.validation:].tolist())
        manifest = {"fps": 30, "task": TASK, "joints": list(JOINTS), "cameras": list(CAMERAS),
                    "episodes": episodes, "train": train, "validation": validation,
                    "failed_seeds": failed, "split_seed": 20261121}
        temp = args.output / "manifest.tmp"
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(args.output / "manifest.json")
        print(f"DATASET_READY episodes={len(episodes)} train={len(train)} validation={len(validation)}",
              flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("menagerie"))
    parser.add_argument("--calibration", type=Path, default=Path("official_block_follower.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    parser.add_argument("--validation", type=int, default=8)
    parser.add_argument("--seed-start", type=int, default=11)
    args = parser.parse_args()
    if not 0 <= args.validation < args.count:
        parser.error("validation count must be nonnegative and smaller than total")
    main(args)
