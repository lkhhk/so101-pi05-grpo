"""Scripted grasp-and-place acceptance test for the SO-101 MuJoCo scene."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from so101_mujoco_blockbox import HOME, SO101BlockBox


def radial_direction(point, shoulder):
    radial = np.r_[point[:2] - shoulder[:2], 0.]
    return radial / np.linalg.norm(radial)


def arm_targets(env, kin, point, pitch, depth, initial):
    radial = radial_direction(point, env.model.body("shoulder").pos)
    side = np.cross([0, 0, 1], radial)
    down = np.sin(pitch) * radial + np.array([0, 0, -np.cos(pitch)])
    target = point + depth * down

    def residual(angles):
        kin.qpos[env.position[:5]] = angles
        mujoco.mj_kinematics(env.model, kin)
        site = kin.site("gripperframe")
        axes = site.xmat.reshape(3, 3)
        return np.r_[site.xpos - target, 0.035 * (axes[:, 0] - down),
                     0.04 * (axes[:, 1] - side)]

    bounds = env.low[:5] + 1e-5, env.high[:5] - 1e-5
    result = least_squares(residual, np.clip(initial, *bounds), bounds=bounds,
                           max_nfev=140, diff_step=1e-4)
    error = np.linalg.norm(result.fun[:3])
    if error > 0.01:
        raise RuntimeError(f"gripper target is out of reach (residual {error:.3f} m)")
    return result.x


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    env = SO101BlockBox(args.assets)
    try:
        env.reset(seed=args.seed)
        kin = mujoco.MjData(env.model)
        box = env.data.mocap_pos[env.box_mocap].copy()
        cube = env.data.qpos[env.cube_pos:env.cube_pos + 3]
        grasp = np.r_[cube[:2], 0.024]
        grasp -= 0.005 * radial_direction(grasp, env.model.body("shoulder").pos)
        plan = (
            ("approach", grasp + [0, 0, 0.06], 0, 0.005, 0.60, 40),
            ("lower", grasp, 0, 0.005, 0.60, 40),
            ("close", grasp, 0, 0.005, -0.12, 30),
            ("lift", grasp + [0, 0, 0.08], 0, 0.005, -0.12, 40),
            ("transfer", box + [0, 0, 0.124], 30, 0, -0.12, 55),
            ("release", box + [0, 0, 0.124], 30, 0, 0.60, 50),
        )
        frames = []

        def capture():
            rgb = np.hstack([env.render_camera("front"), env.render_camera("wrist")])
            frames.append(Image.fromarray(rgb))

        previous = np.deg2rad(HOME)
        phases = []
        capture()
        for phase, point, pitch, depth, jaw, count in plan:
            target = np.r_[arm_targets(env, kin, point, np.deg2rad(pitch),
                                       depth, previous[:5]), jaw]
            for fraction in np.linspace(0, 1, count):
                positions = (1 - fraction) * previous + fraction * target
                _, reward, terminated, truncated, info = env.step(env.action_for_targets(positions))
                if env.steps % 3 == 0 or terminated:
                    capture()
                if terminated or truncated:
                    break
            phases.append({"phase": phase, "step": env.steps, "cube_xyz_m": cube.tolist()})
            if terminated or truncated:
                break
            previous = target

        report = {
            "seed": args.seed, "success": info["success"], "reward": reward,
            "steps": env.steps, "simulated_seconds": env.steps / 30, "phases": phases,
        }
        frames[0].save(args.output / "scripted_grasp.gif", save_all=True,
                       append_images=frames[1:], duration=100, loop=0)
        frames[-1].save(args.output / "scripted_final.png")
        text = json.dumps(report, indent=2)
        (args.output / "scripted_grasp.json").write_text(text + "\n")
        print(text)
        if not info["success"] or not any(
            stage["phase"] == "lift" and stage["cube_xyz_m"][2] > 0.045 for stage in phases
        ):
            raise RuntimeError("the scripted arm did not lift and place the cube")
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("menagerie"))
    parser.add_argument("--output", type=Path, default=Path("smoke"))
    parser.add_argument("--seed", type=int, default=11)
    run(parser.parse_args())
