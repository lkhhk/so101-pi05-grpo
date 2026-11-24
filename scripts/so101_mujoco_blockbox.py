"""Headless MuJoCo environment for SO-101 cube placement."""

import argparse
import json
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import gymnasium as gym
import mujoco
import numpy as np

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper")
CAMERAS = ("front", "wrist")
CUBE_HALF = 0.014
BOX_HALF = np.array([0.055, 0.050])
BOX_FLOOR = 0.006
BOX_RIM = 0.056
BOX_WALL_CONTACT_TOLERANCE = 0.001
WALL = 0.004
HOME = np.array([-12.66, -68.57, 79.03, 53.85, 0.31, 0.0])
CORNERS = np.array(np.meshgrid(*[[-CUBE_HALF, CUBE_HALF]] * 3)).reshape(3, -1).T


def vector(values):
    return " ".join(str(value) for value in values)


def build_scene(asset_dir):
    xml = ET.parse(asset_dir / "so101.xml")
    root = xml.getroot()
    root.find("compiler").set("meshdir", str((asset_dir / "assets").resolve()))
    root.find("option").set("timestep", str(1 / 240))
    for material in root.findall("asset/material"):
        if material.get("rgba") == "1 0.82 0.12 1":
            material.set("rgba", "0.91 0.91 0.88 1")
        if any(name in material.get("name") for name in ("moving_jaw", "wrist_roll_follower")):
            material.set("rgba", "0.10 0.11 0.12 1")
    world = root.find("worldbody")
    ET.SubElement(world, "light", pos="0.1 -0.3 1.1", dir="0 0 -1")
    ET.SubElement(world, "geom", name="table", type="box", pos="0.12 0 -0.03",
                  size="0.58 0.45 0.03", rgba="0.67 0.52 0.37 1")

    eye = np.array([0.36, -0.175, 0.33])
    back = eye - np.array([0.22, 0.012, 0.038])
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    up = np.cross(back / np.linalg.norm(back), right)
    ET.SubElement(world, "camera", name="front", pos=vector(eye),
                  xyaxes=vector(np.r_[right, up]), fovy="58")
    wrist = world.find(".//camera[@name='wrist_cam']")
    wrist.set("name", "wrist")
    del wrist.attrib["quat"]
    # Retain the Menagerie mount and aim it toward the nominal cube at HOME.
    wrist.set("xyaxes", "0.912349 0.230202 0.333856 0.142906 0.595878 -0.790258")
    wrist.set("fovy", "73")

    cube = ET.SubElement(world, "body", name="cube", pos="0.24 0.01 0.014")
    ET.SubElement(cube, "freejoint", name="cube_free")
    ET.SubElement(cube, "geom", name="cube", type="box", size=vector([CUBE_HALF] * 3),
                  mass="0.015", rgba="0.91 0.20 0.09 1", friction="1.1 0.01 0.001")
    box = ET.SubElement(world, "body", name="box", mocap="true", pos="0.25 -0.105 0")

    def box_geom(name, pos, size):
        ET.SubElement(box, "geom", name=name, type="box", pos=vector(pos),
                      size=vector(size), rgba="0.63 0.40 0.24 1")

    x, y = BOX_HALF
    box_geom("box_floor", [0, 0, BOX_FLOOR / 2], [x + WALL, y + WALL, BOX_FLOOR / 2])
    for sign in (-1, 1):
        box_geom(f"box_x_{sign}", [sign * (x + WALL / 2), 0, BOX_RIM / 2],
                 [WALL / 2, y + WALL, BOX_RIM / 2])
        box_geom(f"box_y_{sign}", [0, sign * (y + WALL / 2), BOX_RIM / 2],
                 [x, WALL / 2, BOX_RIM / 2])
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))


class SO101BlockBox(gym.Env):
    """30 Hz joint-position control with sparse placement reward.

    Actions are actuator targets scaled to [-1, 1] in JOINTS order.
    State-only observations support fast RL. Set pixels=True for RGB cameras.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, asset_dir, *, pixels=False, max_steps=900, render_mode=None):
        self.model = build_scene(Path(asset_dir))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.pixels = pixels
        self.max_steps = max_steps
        self.renderer = None
        self.view = mujoco.MjvOption()
        self.view.geomgroup[3:5] = 0
        self.joints = np.array([self.model.joint(name).id for name in JOINTS])
        self.position = self.model.jnt_qposadr[self.joints]
        self.velocity = self.model.jnt_dofadr[self.joints]
        self.actuators = np.array([self.model.actuator(name).id for name in JOINTS])
        self.low, self.high = self.model.actuator_ctrlrange[self.actuators].T
        self.cube_pos = self.model.joint("cube_free").qposadr.item()
        self.cube_vel = self.model.joint("cube_free").dofadr.item()
        self.cube_geom = self.model.geom("cube").id
        self.box_floor = self.model.geom("box_floor").id
        self.box_mocap = self.model.body("box").mocapid.item()
        self.action_space = gym.spaces.Box(-1, 1, (6,), np.float32)
        self.state_space = gym.spaces.Box(-np.inf, np.inf, (28,), np.float32)
        if pixels:
            image = gym.spaces.Box(0, 255, (240, 320, 3), np.uint8)
            self.observation_space = gym.spaces.Dict(
                {**dict.fromkeys(CAMERAS, image), "state": self.state_space}
            )
        else:
            self.observation_space = self.state_space
        self.steps = 0
        self.stable_steps = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.steps = 0
        self.stable_steps = 0
        home = np.deg2rad(HOME)
        self.data.qpos[self.position] = home
        self.data.ctrl[self.actuators] = home
        cube_xy = np.array([0.24, -0.012]) + self.np_random.uniform(-0.012, 0.012, 2)
        box_xy = np.array([0.25, -0.115]) + self.np_random.uniform(-0.008, 0.008, 2)
        self.data.qpos[self.cube_pos:self.cube_pos + 7] = [*cube_xy, CUBE_HALF, 1, 0, 0, 0]
        self.data.mocap_pos[self.box_mocap] = [*box_xy, 0]
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), {"success": False}

    def observation(self):
        state = np.r_[self.data.qpos[self.position], self.data.qvel[self.velocity],
                      self.data.qpos[self.cube_pos:self.cube_pos + 7],
                      self.data.qvel[self.cube_vel:self.cube_vel + 6],
                      self.data.mocap_pos[self.box_mocap]].astype(np.float32)
        if self.pixels:
            return {"state": state, **{name: self.render_camera(name) for name in CAMERAS}}
        return state

    def action_for_targets(self, targets):
        return (2 * (targets - self.low) / (self.high - self.low) - 1).astype(np.float32)

    def cube_status(self):
        center = self.data.geom_xpos[self.cube_geom]
        rotation = self.data.geom_xmat[self.cube_geom].reshape(3, 3)
        corners = CORNERS @ rotation.T + center - self.data.mocap_pos[self.box_mocap]
        # Soft contacts can leave a resting corner slightly inside a wall geom.
        inside = (np.all(np.abs(corners[:, :2]) <= BOX_HALF + BOX_WALL_CONTACT_TOLERANCE)
                  and corners[:, 2].max() < BOX_RIM
                  and corners[:, 2].min() >= BOX_FLOOR - 0.002)
        supported = False
        touching_arm = False
        for contact in self.data.contact:
            first, second = contact.geom
            if self.cube_geom not in (first, second):
                continue
            other = second if first == self.cube_geom else first
            supported |= other == self.box_floor
            body = self.model.geom_bodyid[other]
            while body != 0:
                if body == self.model.body("base").id:
                    touching_arm = True
                    break
                body = self.model.body_parentid[body]
        speed = np.linalg.norm(self.data.qvel[self.cube_vel:self.cube_vel + 3])
        return bool(inside and supported and not touching_arm and speed < 0.025)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (6,) or not np.isfinite(action).all():
            raise ValueError("expected six finite actuator targets")
        targets = self.low + (np.clip(action, -1, 1) + 1) * (self.high - self.low) / 2
        self.data.ctrl[self.actuators] = targets
        mujoco.mj_step(self.model, self.data, nstep=8)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        self.stable_steps = self.stable_steps + 1 if self.cube_status() else 0
        success = self.stable_steps >= 10
        return self.observation(), float(success), success, self.steps >= self.max_steps, {
            "success": success
        }

    def render_camera(self, camera):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=240, width=320)
        self.renderer.update_scene(self.data, camera=camera, scene_option=self.view)
        return self.renderer.render()

    def render(self):
        return self.render_camera("front")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def smoke(args):
    from gymnasium.utils.env_checker import check_env
    from PIL import Image

    args.output.mkdir(parents=True, exist_ok=True)
    env = SO101BlockBox(args.assets, pixels=True, max_steps=40, render_mode="rgb_array")
    state_env = SO101BlockBox(args.assets)
    try:
        check_env(state_env, skip_render_check=True)
        start, _ = env.reset(seed=11)
        for camera in CAMERAS:
            assert start[camera].shape == (240, 320, 3)
            assert np.std(start[camera]) > 5
            Image.fromarray(start[camera]).save(args.output / f"{camera}.png")

        hold = env.action_for_targets(np.deg2rad(HOME))
        begin = perf_counter()
        for _ in range(env.max_steps):
            _, reward, terminated, truncated, _ = env.step(hold)
            assert reward == 0 and not terminated
        assert truncated
        fps = env.max_steps / (perf_counter() - begin)
        repeated, _ = env.reset(seed=11)
        assert np.array_equal(start["state"], repeated["state"])
        # GPU rasterization may round an edge pixel differently after a reset.
        pixel_delta = {key: int(np.max(np.abs(start[key].astype(np.int16) - repeated[key])))
                       for key in CAMERAS}
        assert max(pixel_delta.values()) <= 2, pixel_delta

        # Start the cube above the box to test gravity, contact and reward in isolation.
        x, y, _ = env.data.mocap_pos[env.box_mocap]
        env.data.qpos[env.cube_pos:env.cube_pos + 3] = [x, y, BOX_RIM + 0.025]
        mujoco.mj_forward(env.model, env.data)
        for step in range(1, env.max_steps + 1):
            obs, reward, terminated, _, info = env.step(hold)
            if terminated:
                break
        assert terminated and reward == 1 and info["success"]
        Image.fromarray(obs["front"]).save(args.output / "box_success.png")
        report = {
            "mujoco": mujoco.__version__, "cameras": list(CAMERAS),
            "rgb_shape": [240, 320, 3], "seeded_state_reset_exact": True,
            "seeded_reset_rgb_max_delta": pixel_delta,
            "negative_hold_reward": 0, "negative_hold_truncated": True,
            "positive_test": "gravity drop above box", "positive_reward": 1,
            "positive_steps": step, "dual_camera_steps_per_second": round(fps, 1),
        }
        (args.output / "smoke.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    finally:
        state_env.close()
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets", type=Path, default=Path("menagerie"))
    parser.add_argument("--output", type=Path, default=Path("smoke"))
    smoke(parser.parse_args())
