"""One-episode SO-101 drawer probe using the existing PI05 policy interface."""
import argparse
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
import torch

import evaluate_so101_mujoco_lora as evaluation
from demo_so101_mujoco_blockbox import arm_targets
from pi05_rl_runtime import load_ira_lora, make_processors
from so101_mujoco_blockbox import CAMERAS, HOME, JOINTS, SO101BlockBox, build_scene, vector

TASK = "Open the drawer."
TARGET_M = 0.060
TRAVEL_M = 0.075
HOLD_STEPS = 10
SEED = 11
EXECUTE_STEPS = 50
MAX_STEPS = 900
ADAPTER_SHA = "c4b65f3134cd065f45dc7c7e40f2d9adb3291116e3cc0eec49f1aa60a49a8fd6"


def drawer_scene(assets):
    blockbox = build_scene(assets)
    with tempfile.TemporaryDirectory() as folder:
        compiled = Path(folder) / "blockbox.xml"
        mujoco.mj_saveLastXML(str(compiled), blockbox)
        root = ET.parse(compiled).getroot()
    root.find("compiler").set("meshdir", str((assets / "assets").resolve()))
    world = root.find("worldbody")
    for name in ("cube", "box"):
        world.remove(world.find(f"body[@name='{name}']"))

    # Face the drawer front, with the opening direction toward the arm.
    camera = world.find("camera[@name='front']")
    eye = np.array([0.12, -0.25, 0.29])
    back = eye - np.array([0.25, -0.035, 0.036])
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    up = np.cross(back / np.linalg.norm(back), right)
    for name in ("quat", "axisangle", "euler", "zaxis", "xyaxes"):
        camera.attrib.pop(name, None)
    camera.set("pos", vector(eye))
    camera.set("xyaxes", vector(np.r_[right, up]))

    cabinet = ET.SubElement(world, "body", name="cabinet", pos="0.313 -0.045 0")
    drawer = ET.SubElement(world, "body", name="drawer", pos="0.313 -0.045 0")
    ET.SubElement(drawer, "joint", name="drawer_slide", type="slide", axis="-1 0 0",
                  limited="true", range=f"0 {TRAVEL_M}", damping="1.2", frictionloss="0.3")

    def box(parent, name, pos, size, color, mass=None):
        attrib = dict(name=name, type="box", pos=pos, size=size, rgba=color,
                      friction="1.2 0.02 0.002")
        if mass is not None:
            attrib["mass"] = mass
        ET.SubElement(parent, "geom", **attrib)

    for sign, side in ((-1, "near"), (1, "far")):
        box(cabinet, f"cabinet_{side}", f"0 {0.055*sign} 0.031",
            "0.058 0.004 0.031", "0.38 0.44 0.49 1")
        box(drawer, f"drawer_{side}", f"0 {0.045*sign} 0.024",
            "0.049 0.003 0.011", "0.66 0.48 0.30 1", "0.025")
    box(cabinet, "cabinet_top", "0 0 0.064", "0.058 0.059 0.003", "0.38 0.44 0.49 1")
    box(cabinet, "cabinet_back", "0.055 0 0.031", "0.003 0.051 0.031", "0.38 0.44 0.49 1")
    box(drawer, "drawer_floor", "0 0 0.010", "0.049 0.048 0.003", "0.66 0.48 0.30 1", "0.07")
    box(drawer, "drawer_face", "-0.052 0 0.031", "0.004 0.048 0.028", "0.66 0.48 0.30 1", "0.035")
    for sign, side in ((-1, "near"), (1, "far")):
        box(drawer, f"handle_post_{side}", f"-0.064 {0.023*sign} 0.039",
            "0.009 0.003 0.003", "0.22 0.27 0.31 1", "0.008")
    box(drawer, "drawer_handle", "-0.075 0 0.039", "0.004 0.026 0.005",
        "0.16 0.23 0.28 1", "0.018")
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class DrawerScene:
    render_camera = SO101BlockBox.render_camera
    close = SO101BlockBox.close

    def __init__(self, assets):
        self.model = drawer_scene(assets)
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.view = mujoco.MjvOption()
        self.view.geomgroup[3:5] = 0
        joints = [self.model.joint(name).id for name in JOINTS]
        self.position = self.model.jnt_qposadr[joints]
        self.actuators = [self.model.actuator(name).id for name in JOINTS]
        self.low, self.high = self.model.actuator_ctrlrange[self.actuators].T
        self.slide = self.model.joint("drawer_slide").qposadr.item()
        self.slide_dof = self.model.joint("drawer_slide").dofadr.item()
        self.handle_id = self.model.geom("drawer_handle").id
        self.drawer_id = self.model.body("drawer").id
        self.robot_geoms = set()
        robot = self.model.body("base").id
        for geom_id, body_id in enumerate(self.model.geom_bodyid):
            while body_id:
                if body_id == robot:
                    self.robot_geoms.add(geom_id)
                    break
                body_id = self.model.body_parentid[body_id]
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.position] = np.deg2rad(HOME)
        self.data.ctrl[self.actuators] = np.deg2rad(HOME)
        mujoco.mj_forward(self.model, self.data)

    def step(self, targets):
        self.data.ctrl[self.actuators] = targets
        mujoco.mj_step(self.model, self.data, nstep=8)
        mujoco.mj_forward(self.model, self.data)

    def contacts(self):
        drawer, handle = False, False
        for contact in self.data.contact:
            first, second = map(int, contact.geom)
            if second in self.robot_geoms:
                first, second = second, first
            if first in self.robot_geoms and self.model.geom_bodyid[second] == self.drawer_id:
                drawer = True
                handle |= second == self.handle_id
        return drawer, handle

    def images(self):
        return [self.render_camera(camera) for camera in CAMERAS]


def preflight(env, output):
    # Verify kinematic reach at both ends, resting closure, and slide motion.
    kin = mujoco.MjData(env.model)
    handle = env.data.geom_xpos[env.handle_id].copy()
    for point in (handle, handle - [TARGET_M, 0, 0]):
        arm_targets(env, kin, point, np.deg2rad(65), 0, np.deg2rad(HOME[:5]))
    for _ in range(30):
        env.step(np.deg2rad(HOME))
    idle_m = float(env.data.qpos[env.slide])
    assert abs(idle_m) < 0.002, f"Drawer moved during idle: {idle_m}"
    env.data.qfrc_applied[env.slide_dof] = 1.5
    for _ in range(30):
        env.step(np.deg2rad(HOME))
    moved_m = float(env.data.qpos[env.slide])
    assert moved_m >= TARGET_M, f"Slide does not reach target: {moved_m}"
    env.reset()
    preview = np.hstack(env.images())
    Image.fromarray(preview).save(output / "initial.png")
    return {"idle_mm": 1000 * idle_m, "external_force_preflight_mm": 1000 * moved_m,
            "handle_reachable_by_ik": True}


def evaluate(env, args):
    policy, weights = load_ira_lora(args.base, args.adapter, device=args.device, offload_text_tables=args.device == "cuda")
    assert weights["base_sha256"] == evaluation.BASE_SHA and weights["adapter_sha256"] == ADAPTER_SHA
    policy.requires_grad_(False)
    if args.device == "cpu":
        policy.float()
    pre, post = make_processors(policy, args.adapter, args.tokenizer, args.device)
    print("MODEL_READY", args.device, flush=True)
    calibration = json.loads(args.calibration.read_bytes())
    mapping = evaluation.JointMapping(calibration)
    np.testing.assert_allclose(mapping.observation(env.data.qpos[env.position]),
                               evaluation.REFERENCE, atol=1e-5)
    evaluation.TASK = TASK
    torch.manual_seed(SEED)
    initial = env.images()
    frames = [evaluation.render_frame(initial, "002520 drawer", SEED, 0, "")]
    trace = []
    steps = stable = 0
    success = False
    while not success and steps < MAX_STEPS:
        rgb = initial if steps == 0 else env.images()
        state = mapping.observation(env.data.qpos[env.position])
        commands = evaluation.predict(policy, pre, post, state, rgb)
        for cmd in commands[:EXECUTE_STEPS]:
            env.step(np.clip(mapping.radians(cmd), env.low, env.high))
            steps += 1
            position = float(env.data.qpos[env.slide])
            drawer, handle = env.contacts()
            distance = float(np.linalg.norm(env.data.site("gripperframe").xpos -
                                            env.data.geom_xpos[env.handle_id]))
            trace.append((position, drawer, handle, distance))
            stable = stable + 1 if position >= TARGET_M else 0
            success = stable >= HOLD_STEPS
            if steps % 6 == 0 or success or steps == MAX_STEPS:
                frames.append(evaluation.render_frame(env.images(), "002520 drawer", SEED, steps,
                                                      f" | slide {position*1000:.0f} mm"))
            if success or steps == MAX_STEPS:
                break
    frames[0].save(args.output / "trial.gif", save_all=True, append_images=frames[1:],
                   duration=[200] * (len(frames)-1) + [1000], loop=0)
    frames[-1].save(args.output / "final.png")
    data = np.array(trace)
    np.savez_compressed(args.output / "trajectory.npz", slide_m=data[:, 0],
                        robot_drawer_contact=data[:, 1].astype(bool),
                        robot_handle_contact=data[:, 2].astype(bool),
                        gripper_to_handle_m=data[:, 3])
    return {"mode": "MuJoCo zero-shot ODE", "checkpoint": "002520", "weights": weights,
            "task": TASK, "inference_dtype": str(next(policy.parameters()).dtype), "seed": SEED, "execute_steps": EXECUTE_STEPS,
            "denoising_steps": policy.config.num_inference_steps, "success": success,
            "steps": steps, "simulated_seconds": steps / 30,
            "success_threshold_mm": 1000 * TARGET_M, "required_control_steps": HOLD_STEPS,
            "drawer_travel_mm": 1000 * TRAVEL_M, "max_open_mm": 1000 * data[:, 0].max(),
            "final_open_mm": 1000 * data[-1, 0],
            "drawer_contact_steps": int(data[:, 1].sum()),
            "handle_contact_steps": int(data[:, 2].sum()),
            "minimum_gripper_to_handle_mm": 1000 * data[:, 3].min()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("assets", "base", "adapter", "tokenizer", "calibration", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--preflight-only", action="store_true")
    opts = parser.parse_args()
    env = DrawerScene(opts.assets)
    try:
        info = preflight(env, opts.output)
        print(json.dumps(info), flush=True)
        if not opts.preflight_only:
            report = evaluate(env, opts)
            report["fixture_preflight"] = info
            (opts.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
    finally:
        env.close()
