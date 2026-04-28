import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from env import MiniMetroFrameStack


class CounterEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        return np.array([1, 2, 3], dtype=np.float32), {}

    def step(self, action):
        self.step_count += 1
        obs = np.array([self.step_count] * 3, dtype=np.float32)
        return obs, 0.0, False, False, {}

    def action_masks(self):
        return np.array([True, False], dtype=bool)


def test_frame_stack_preserves_action_masks_and_flattens_observation():
    base = DummyVecEnv([lambda: CounterEnv()])
    base.action_masks = lambda: [base.envs[0].action_masks()]
    env = MiniMetroFrameStack(base, n_stack=4)

    obs = env.reset()
    assert obs.shape == (1, 12)
    assert np.array_equal(obs[0], np.array([1, 2, 3] * 4, dtype=np.float32))
    assert env.action_masks()[0].tolist() == [True, False]

    env.step_async(np.array([0]))
    obs, _, _, _ = env.step_wait()
    assert obs.shape == (1, 12)
    assert np.array_equal(obs[0, -3:], np.array([1, 1, 1], dtype=np.float32))
