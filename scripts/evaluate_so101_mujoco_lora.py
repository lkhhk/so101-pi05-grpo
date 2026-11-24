"""Evaluate a verified pi0.5 SO-101 adapter in MuJoCo."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageDraw
import torch

from pi05_rl_runtime import load_ira_lora, make_processors
from so101_mujoco_blockbox import CAMERAS, HOME, JOINTS, SO101BlockBox

TASK = "Pick up the block and place it inside the box."
BASE_SHA = "7f8056e0cebbd8767a93c0f51bf5e20751baad455cf4b9965c07efca2df0fdef"
ADAPTER_SHA = "34152040b57910ffe2b3ed3d80b8fa09fd4e19de73bffcb4892860922e1498cb"
REFERENCE = np.array([-12.6593406593, -68.5714285714, 79.0329670330,
                      53.8461538462, 0.3076923077, 1.3413816231])


class JointMapping:
    """Approximate motor-to-sim mapping anchored at the shared starting pose."""

    def __init__(self, calibration):
        if any(calibration[name]["drive_mode"] != 0 for name in JOINTS):
            raise ValueError("This mapping expects increasing encoder counts for every joint")
        jaw = calibration["gripper"]
        jaw_rad_per_percent = (jaw["range_max"] - jaw["range_min"]) * (2 * np.pi / 4095) / 100
        self.scale = np.r_[np.repeat(np.pi / 180, 5), jaw_rad_per_percent]
        self.home = np.deg2rad(HOME)

    def radians(self, state):
        return self.home + (state - REFERENCE) * self.scale

    def observation(self, radians):
        state = (radians - self.home) / self.scale + REFERENCE
        state[-1] = np.clip(state[-1], 0, 100)
        return state.astype(np.float32)


def render_frame(images, label, seed, step, phase):
    frame = Image.new("RGB", (640, 265), "#202328")
    for index, rgb in enumerate(images):
        frame.paste(Image.fromarray(rgb), (index * 320, 25))
    draw = ImageDraw.Draw(frame)
    draw.text((8, 7), f"Front | {label} | seed {seed} | {step / 30:04.1f} s",
              fill="white")
    draw.text((328, 7), f"Wrist{phase}", fill="white")
    return frame


@torch.no_grad()
def predict(policy, pre, post, state, rgb):
    batch = {"observation.state": torch.from_numpy(state), "task": TASK}
    for name, image in zip(CAMERAS, rgb):
        batch[f"observation.images.{name}"] = torch.from_numpy(image).permute(2, 0, 1).float() / 255
    actions = post(policy.predict_action_chunk(pre(batch))).detach().float().cpu().numpy()[0]
    if actions.shape != (50, 6) or not np.isfinite(actions).all():
        raise ValueError(f"Expected 50 finite six-joint actions, got {actions.shape}")
    return actions


def run_episode(args, seed, policy, pre, post, mapping, output):
    torch.manual_seed(seed)
    env = SO101BlockBox(args.assets)
    try:
        env.reset(seed=seed)
        np.testing.assert_allclose(mapping.observation(env.data.qpos[env.position]),
                                   REFERENCE, atol=1e-5)
        cube = env.data.qpos[env.cube_pos:env.cube_pos + 3]
        cube_track = [cube.copy()]
        frames = []
        predicted, targets, states, timings, chunk_steps = [], [], [], [], []
        terminated = truncated = False
        total_reward = 0.0

        def images():
            return [env.render_camera(name) for name in CAMERAS]

        while not (terminated or truncated):
            obs_rgb = images()
            if env.steps == 0:
                frames.append(render_frame(obs_rgb, args.label, seed, 0, ""))
            states.append(mapping.observation(env.data.qpos[env.position]))
            chunk_steps.append(env.steps)
            start = perf_counter()
            action_chunk = predict(policy, pre, post, states[-1], obs_rgb)
            timings.append(perf_counter() - start)
            for command in action_chunk[:args.execute_steps]:
                clipped = np.clip(mapping.radians(command), env.low, env.high)
                predicted.append(command)
                targets.append(clipped)
                _, reward, terminated, truncated, info = env.step(env.action_for_targets(clipped))
                cube_track.append(cube.copy())
                total_reward += reward
                if env.steps % 6 == 0 or terminated or truncated:
                    phase = " | placed" if terminated else " | timeout" if truncated else ""
                    frames.append(render_frame(images(), args.label, seed, env.steps, phase))
                if terminated or truncated:
                    break

        gif = output / f"seed-{seed}.gif"
        frames[0].save(gif, save_all=True, append_images=frames[1:],
                       duration=[200] * (len(frames) - 1) + [1000], loop=0)
        cube_track = np.array(cube_track)
        predicted = np.array(predicted)
        targets = np.array(targets)
        excess = np.abs(mapping.radians(predicted) - targets) > 1e-7
        box = env.data.mocap_pos[env.box_mocap].copy()
        np.savez_compressed(output / f"seed-{seed}.npz", model_actions=predicted,
                            sim_targets_rad=targets, cube_xyz_m=cube_track,
                            box_xyz_m=box, chunk_states=np.array(states),
                            chunk_steps=np.array(chunk_steps))
        return {
            "seed": seed, "success": info["success"], "reward": total_reward, "steps": env.steps,
            "simulated_seconds": env.steps / 30, "timeout": bool(truncated),
            "predicted_chunks": len(timings), "mean_inference_seconds": float(np.mean(timings)),
            "clipped_steps": int(excess.any(axis=1).sum()),
            "clipped_steps_by_joint": dict(zip(JOINTS, excess.sum(axis=0).tolist())),
            "cube_start_m": cube_track[0].tolist(), "cube_final_m": cube_track[-1].tolist(),
            "box_center_m": box.tolist(),
            "cube_peak_rise_mm": float(1000 * (cube_track[:, 2].max() - cube_track[0, 2])),
            "cube_max_horizontal_movement_mm": float(1000 * np.linalg.norm(
                cube_track[:, :2] - cube_track[0, :2], axis=1).max()),
            "video": gif.name,
        }
    finally:
        env.close()


def main(args):
    if not 1 <= args.execute_steps <= 50:
        raise ValueError("execute-steps must be between 1 and 50")
    args.output.mkdir(parents=True, exist_ok=False)
    mapping = JointMapping(json.loads(args.calibration.read_text(encoding="utf-8")))
    placement = {"offload_text_tables": True} if args.offload_text_tables else {}
    policy, provenance = load_ira_lora(args.base, args.adapter, device="cuda", **placement)
    if (provenance["base_sha256"], provenance["adapter_sha256"]) != (BASE_SHA, args.adapter_sha256):
        raise ValueError("Loaded weights differ from the pinned base and adapter hashes")
    policy.requires_grad_(False)
    pre, post = make_processors(policy, args.adapter, args.tokenizer, "cuda")
    report = {"checkpoint_label": args.label, "weights": provenance, "task": TASK, "camera_names": list(CAMERAS),
              "control_hz": 30, "action_chunk": 50, "execute_steps": args.execute_steps,
              "denoising_steps": policy.config.num_inference_steps,
              "mapping": "approximate forward and inverse mapping at the shared start pose",
              "real_reference": REFERENCE.tolist(), "home_rad": mapping.home.tolist(),
              "rad_per_unit": mapping.scale.tolist(), "episodes": [], "status": "running"}

    def write_report():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    write_report()
    print("PINNED_LORA_READY", provenance["adapter_sha256"], flush=True)
    for seed in args.seeds:
        row = run_episode(args, seed, policy, pre, post, mapping, args.output)
        report["episodes"].append(row)
        write_report()
        print(json.dumps(row), flush=True)
    report["successes"] = sum(row["success"] for row in report["episodes"])
    report["status"] = "complete"
    write_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("menagerie"))
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-sha256", default=ADAPTER_SHA)
    parser.add_argument("--label", default="pi0.5 LoRA 020000")
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-steps", type=int, default=50)
    parser.add_argument("--offload-text-tables", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 101, 102, 103, 104])
    main(parser.parse_args())
