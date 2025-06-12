from collections import deque

import ale_py
import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

gym.register_envs(ale_py)


class AtariWrapper(gym.Wrapper):
    def __init__(self, env_id="", render_mode=None, frame_stack=4, resize_height=64, resize_width=64):
        env = gym.make(env_id, obs_type="grayscale", render_mode=render_mode)
        super().__init__(env)

        self.frame_stack = frame_stack

        self.frames = deque([], maxlen=frame_stack)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(frame_stack, resize_height, resize_width),
            dtype=np.uint8
        )

    def _preprocess(self, frame):
        frame = cv2.resize(
            frame,
            (
                self.observation_space.shape[1],
                self.observation_space.shape[2],
            ),
            interpolation=cv2.INTER_AREA
        )
        return frame

    def _get_observation(self) -> np.ndarray:
        return np.array(self.frames, dtype=np.uint8)

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        self.frames.clear()
        obs = self._preprocess(obs)
        for _ in range(self.frame_stack):
            self.frames.append(obs)

        return self._get_observation(), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self._preprocess(obs)
        self.frames.append(obs)

        return self._get_observation(), reward, terminated, truncated, info
