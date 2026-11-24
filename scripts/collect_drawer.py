"""Collect successful seeded drawer demonstrations with 30 Hz action labels."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from drawer_env import DrawerTask, OBS_STRIDE, SCENE_SEED_BASE
from eval_drawer import CAMERAS, JOINTS, TASK, evaluation
from scripted_drawer import rollout
from train_so101_mujoco_sft import write_json

TOTAL = 200
VALIDATION = 10
SPLIT_SEED = 20261124


def expert(env, seed, closed):
    for offsets in ((0, 0, 0, 0, .20), (0, 0, -.003, 0, .20),
                    (-.003, 0, 0, 0, .20), (.003, 0, -.003, 0, .24)):
        try:
            report, targets = rollout(env, offsets, seed=seed, jaw_closed=closed)
        except RuntimeError as error:
            if "out of reach" not in str(error):
                raise
            continue
        if report["success"]:
            return targets
    return None


def replay(env, mapping, seed, targets, directory=None):
    env.reset(seed=seed)
    initial = float(env.data.qpos[env.slide])
    handle = env.data.geom_xpos[env.handle_id].tolist()
    states, actions, rgb_frames = [], [], []
    for index, target in enumerate(targets):
        state = mapping.observation(env.data.qpos[env.position])
        action = mapping.observation(target)
        actual = np.clip(mapping.radians(action), env.low, env.high)
        np.testing.assert_allclose(actual, target, atol=2e-6)
        states.append(state)
        actions.append(action)
        if index % OBS_STRIDE == 0:
            rgb_frames.append(index)
            if directory is not None:
                for camera, rgb in zip(CAMERAS, env.images()):
                    Image.fromarray(rgb).save(directory / f"{camera}-{index:04d}.png",
                                              compress_level=1)
        env.step(actual)
        if env.success:
            break
    if not env.success:
        raise RuntimeError(f"Seed {seed} failed when replaying mapped actions")
    if directory is not None:
        np.savez(directory / "motion.npz", state=np.stack(states), action=np.stack(actions),
                 rgb_frames=np.array(rgb_frames))
    return {"frames": len(actions), "rgb_frames": len(rgb_frames), "success": True,
            "drawer_start_mm": initial * 1000, "handle_start_m": handle,
            "drawer_end_mm": 1000 * float(env.data.qpos[env.slide]),
            "handle_contact": env.touched_handle}


def collect(args):
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    if manifest.is_file():
        print("DATASET_EXISTS", manifest, flush=True)
        return
    mapping = evaluation.JointMapping(json.loads(args.calibration.read_bytes()))
    closed_state = evaluation.REFERENCE.copy()
    closed_state[-1] = 0
    closed = mapping.radians(closed_state)[-1]
    env = DrawerTask(args.assets)
    try:
        if args.preflight:
            ok = 0
            for seed in range(SCENE_SEED_BASE, SCENE_SEED_BASE + 10):
                targets = expert(env, seed, closed)
                if targets is not None:
                    try:
                        result = replay(env, mapping, seed, targets)
                    except RuntimeError:
                        continue
                    print("PREFLIGHT", seed, result, flush=True)
                    ok += 1
            if ok < 5:
                raise RuntimeError(f"Only {ok}/10 scripted layouts passed")
            return
        rows = [json.loads(meta.read_bytes()) for meta in sorted(root.glob("episode-*/episode.json"))]
        tried = {row["seed"] for row in rows}
        assert [r["episode"] for r in rows] == list(range(len(rows)))
        for seed in range(SCENE_SEED_BASE, SCENE_SEED_BASE + 1500):
            if len(rows) == TOTAL:
                break
            if seed in tried:
                continue
            if shutil.disk_usage(root).free < 6 * 2**30:
                raise RuntimeError("Need at least 6 GiB free for demonstration collection")
            targets = expert(env, seed, closed)
            if targets is None:
                print("REJECT", seed, flush=True)
                continue
            try:
                replay(env, mapping, seed, targets)
            except RuntimeError as error:
                if "failed when replaying" not in str(error):
                    raise
                print("REJECT_REPLAY", seed, flush=True)
                continue
            name = f"episode-{len(rows):03d}"
            pending = root / (name + ".pending")
            if pending.exists():
                shutil.rmtree(pending)
            pending.mkdir()
            row = {"episode": len(rows), "directory": name, "seed": seed,
                   **replay(env, mapping, seed, targets, pending)}
            write_json(pending / "episode.json", row)
            pending.rename(root / name)
            rows.append(row)
            print("DEMO", json.dumps(row), flush=True)
        if len(rows) != TOTAL:
            raise RuntimeError(f"Collected {len(rows)}/{TOTAL} accepted episodes")
        shuffled = np.random.default_rng(SPLIT_SEED).permutation(TOTAL).tolist()
        write_json(manifest, {"fps": 30, "rgb_stride": OBS_STRIDE, "task": TASK,
                              "cameras": list(CAMERAS), "joints": list(JOINTS), "episodes": rows,
                              "train": sorted(shuffled[VALIDATION:]),
                              "validation": sorted(shuffled[:VALIDATION]), "split_seed": SPLIT_SEED})
        print("DATASET_READY", TOTAL, TOTAL-VALIDATION, VALIDATION, flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("menagerie"))
    parser.add_argument("--calibration", type=Path, default=Path("official_block_follower.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    collect(parser.parse_args())
