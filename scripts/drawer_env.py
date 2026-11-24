"""Seeded SO-101 drawer task shared by demonstrations and GRPO."""
import mujoco
import numpy as np

from eval_drawer import DrawerScene, HOLD_STEPS, TARGET_M

SCENE_SEED_BASE = 125000
RL_SCENE_SEED_BASE = 180000
OBS_STRIDE = 3


class DrawerTask(DrawerScene):
    def reset(self, *, seed=None):
        super().reset()
        self.steps = self.stable = 0
        self.touched_handle = False
        self.success = False
        if seed is not None:
            rng = np.random.default_rng(seed)
            offset = np.r_[rng.uniform(-0.009, 0.009, size=2), 0.]
            nominal = [0.313, -0.045, 0.]
            for name in ("cabinet", "drawer"):
                self.model.body(name).pos[:] = nominal + offset
            self.data.qpos[self.slide] = rng.uniform(0.0, 0.008)
            mujoco.mj_forward(self.model, self.data)
        return self.observation(), {"success": False}

    def observation(self):
        return np.r_[self.data.qpos[self.position],
                     self.data.qpos[self.slide], self.data.qvel[self.slide_dof],
                     self.data.geom_xpos[self.handle_id]]

    def step(self, targets):
        super().step(targets)
        self.steps += 1
        self.touched_handle |= self.contacts()[1]
        opening = float(self.data.qpos[self.slide])
        self.stable = self.stable + 1 if opening >= TARGET_M and self.touched_handle else 0
        self.success = self.stable >= HOLD_STEPS


class DrawerB8(DrawerTask):
    def __init__(self, assets):
        super().__init__(assets)
        self.max_steps = 900

    def action_for_targets(self, targets):
        return (2 * (targets - self.low) / (self.high - self.low) - 1).astype(np.float32)

    def step(self, action):
        targets = self.low + (np.clip(action, -1, 1) + 1) * (self.high - self.low) / 2
        super().step(targets)
        return self.observation(), float(self.success), self.success, self.steps >= self.max_steps, {
            "success": self.success,
        }
