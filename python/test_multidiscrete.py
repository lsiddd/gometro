import numpy as np
import gymnasium as gym
from gymnasium import spaces
from sb3_contrib import MaskablePPO

class TestEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,))
        self.action_space = spaces.MultiDiscrete([2, 3, 4])
    def reset(self, seed=None):
        return np.ones(4), {}
    def step(self, action):
        return np.ones(4), 1.0, False, False, {}
    def action_masks(self):
        # A single flat 1D array of sum(dims)
        return np.array([1, 1,   1, 1, 0,   1, 0, 1, 1], dtype=bool)

env = TestEnv()
model = MaskablePPO("MlpPolicy", env, n_steps=16)
model.learn(total_timesteps=16)
print("Finished learning with MultiDiscrete!")
