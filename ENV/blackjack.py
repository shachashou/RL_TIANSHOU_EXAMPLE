import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class BlackjackEnvWrapper(gym.Wrapper):
    reward_threshold = 0.1

    def __init__(self, render_mode=None):
        env = gym.make("Blackjack-v1", render_mode=render_mode)
        super().__init__(env)

        self.observation_space = Box(
            low=np.array([0, 0, 0], dtype=np.float32),
            high=np.array([31, 10, 1], dtype=np.float32),
            shape=(3,),
            dtype=np.float32
        )

    def _flatten_obs(self, obs_tuple: tuple) -> np.ndarray:
        return np.array(obs_tuple, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._flatten_obs(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._flatten_obs(obs), reward, terminated, truncated, info
