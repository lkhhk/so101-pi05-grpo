"""Generate a contact-driven scripted drawer demonstration."""
import argparse
import itertools
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from eval_drawer import DrawerScene, HOLD_STEPS, HOME, TARGET_M, arm_targets, evaluation


def rollout(env, offsets, save=None, *, seed=None, jaw_closed=-.1745):
    env.reset() if seed is None else env.reset(seed=seed)
    kin = mujoco.MjData(env.model)
    handle = env.data.geom_xpos[env.handle_id].copy()
    dx, dy, dz, pitch, jaw_open = offsets
    pinch = handle + [dx, dy, dz]
    prior = np.deg2rad(HOME)
    angles = prior[:5]
    phases = []
    frames, trace = [], []
    steps = stable = 0

    def render():
        frames.append(evaluation.render_frame(env.images(), "scripted drawer", 11, steps,
                                              f" | slide {1000 * env.data.qpos[env.slide]:.0f} mm"))

    def move(target, count):
        nonlocal prior, steps, stable
        for mix in np.linspace(0, 1, count + 1)[1:]:
            command = np.clip(prior + mix * (target - prior), env.low, env.high)
            env.step(command)
            steps += 1
            drawer, handle_contact = env.contacts()
            slide = float(env.data.qpos[env.slide])
            stable = stable + 1 if slide >= TARGET_M else 0
            trace.append((command.copy(), env.data.qpos[env.position].copy(), slide,
                          drawer, handle_contact, stable))
            if save is not None and steps % 3 == 0:
                render()
        prior = target

    def waypoint(point, jaw, n):
        nonlocal angles
        angles = arm_targets(env, kin, point, np.deg2rad(pitch), 0, angles)
        move(np.r_[angles, jaw], n)

    def phase(name):
        phases.append({"phase": name, "step": steps, "open_mm": 1000 * trace[-1][2],
                       "handle_contact": trace[-1][4]})

    if save is not None:
        render()
    waypoint(pinch + [0, 0, .063], jaw_open, 75)
    phase("hover")
    for lift in np.linspace(.057, 0, 16):
        waypoint(pinch + [0, 0, lift], jaw_open, 5)
    phase("approach")
    move(np.r_[angles, jaw_closed], 45)
    phase("grasp")
    for opening in np.linspace(.004, .072, 19):
        waypoint(pinch - [opening, 0, 0], jaw_closed, 7)
    phase("pull")
    move(prior, 45)
    phase("hold")
    displacement = np.array([row[2] for row in trace])
    contacts = np.array([row[4] for row in trace], dtype=bool)
    streaks = np.array([row[5] for row in trace])
    result = {"success": bool(streaks.max() >= HOLD_STEPS and contacts.any()),
              "max_open_mm": 1000 * float(displacement.max()),
              "final_open_mm": 1000 * float(displacement[-1]),
              "handle_contact_steps": int(contacts.sum()),
              "longest_open_hold_steps": int(streaks.max()),
              "success_threshold_mm": 1000 * TARGET_M,
              "required_control_steps": HOLD_STEPS,
              "steps": steps, "simulated_seconds": steps / 30,
              "offsets_m_pitch_deg_jaw_rad": list(offsets), "phases": phases,
              "drawer_force_override": False}
    if save is not None:
        render()
        frames[0].save(save / "scripted-open-drawer.gif", save_all=True, append_images=frames[1:],
                       duration=[100] * (len(frames)-1) + [750], loop=0)
        frames[-1].save(save / "scripted-final.png")
        np.savez_compressed(save / "scripted-trajectory.npz",
                            target_rad=np.stack([r[0] for r in trace]),
                            joints_rad=np.stack([r[1] for r in trace]), slide_m=displacement,
                            drawer_contact=np.array([r[3] for r in trace], dtype=bool),
                            handle_contact=contacts)
        (save / "scripted-report.json").write_text(json.dumps(result, indent=2) + "\n")
    return (result, np.stack([row[0] for row in trace])) if seed is not None else result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    env = DrawerScene(args.assets)
    try:
        candidates = itertools.product((0, -.004, .004, -.008), (0, .007, -.007),
                                       (0, .004, -.004), (0, 15, 35, 65), (.20, .32))
        best = None
        for candidate in candidates:
            try:
                result = rollout(env, candidate)
            except RuntimeError:
                continue
            if best is None or result["max_open_mm"] > best["max_open_mm"]:
                best = result
                print("BEST", json.dumps(result), flush=True)
            if result["success"]:
                result = rollout(env, candidate, save=args.output)
                assert result["success"], "Replay diverged"
                print("SCRIPTED_SUCCESS", json.dumps(result), flush=True)
                return
        raise RuntimeError(f"No candidate passed; best {best}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
