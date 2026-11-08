"""Headless planar pushing with real MuJoCo contact dynamics."""

from pathlib import Path

import mujoco
import numpy as np

OBS_DIM = 12
ACTION_DIM = 2
HORIZON = 120
SUCCESS_RADIUS = 0.025


class PushEnv:
    """Two prismatic arm joints push a block constrained to the table plane.

    Observations: arm xy, block xy, target xy, block minus arm xy,
    target minus block xy, arm velocity xy. Actions are normalized velocities.
    """

    def __init__(self, horizon: int = HORIZON, lateral_offset: float = 0.0):
        if horizon < 1:
            raise ValueError("horizon must be positive")
        if not np.isfinite(lateral_offset) or abs(lateral_offset) > 0.15:
            raise ValueError("lateral_offset must be finite and within [-0.15, 0.15] m")
        self.model = mujoco.MjModel.from_xml_path(str(Path(__file__).parent / "models/push.xml"))
        self.data = mujoco.MjData(self.model)
        self.horizon = horizon
        self.lateral_offset = lateral_offset
        self.target = np.zeros(2)
        self.steps = 0
        self.stable_steps = 0
        self.done = True

    def reset(self, seed: int = 0) -> np.ndarray:
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        block = np.array([rng.uniform(-0.1, -0.03), rng.uniform(-0.08, 0.08)])
        self.target = np.array([rng.uniform(0.17, 0.27), block[1] + rng.uniform(-0.02, 0.02)])
        direction = self.target - block
        direction /= np.linalg.norm(direction)
        self.data.qpos[:2] = block - 0.09 * direction + rng.uniform(-0.008, 0.008, 2)
        self.data.qpos[1] += self.lateral_offset
        self.data.qpos[2:4] = block
        self.model.site_pos[0, :2] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.steps = self.stable_steps = 0
        self.done = False
        return self.observation()

    def observation(self) -> np.ndarray:
        arm, block = self.data.qpos[:2], self.data.qpos[2:4]
        return np.concatenate(
            [arm, block, self.target, block - arm, self.target - block, self.data.qvel[:2]]
        ).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        if self.done:
            raise RuntimeError("reset before stepping a finished episode")
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError("action must contain two finite values")
        if np.any(np.abs(action) > 1.000001):
            raise ValueError("action must be in [-1, 1]")
        self.data.ctrl[:] = np.clip(action, -1, 1) * 0.2
        mujoco.mj_step(self.model, self.data, nstep=20)
        mujoco.mj_forward(self.model, self.data)
        self.steps += 1
        close = self.distance < SUCCESS_RADIUS and np.linalg.norm(self.data.qvel[2:4]) < 0.05
        self.stable_steps = self.stable_steps + 1 if close else 0
        success = self.stable_steps >= 3
        self.done = success or self.steps >= self.horizon
        return self.observation(), success, self.done

    @property
    def distance(self) -> float:
        return float(np.linalg.norm(self.data.qpos[2:4] - self.target))


def expert_action(obs: np.ndarray) -> np.ndarray:
    """Push the rear face along x and slide toward the target's lateral coordinate.

    The task's lateral target offset is smaller than the success radius. A box
    contact should stay on its rear face rather than rotate around a point goal.
    """
    arm, block, target = obs[:2], obs[2:4], obs[4:6]
    desired_tip = np.array([block[0] - 0.044 + 0.5 * (target[0] - block[0]), target[1]])
    return np.clip(12.0 * (desired_tip - arm), -1, 1).astype(np.float32)
