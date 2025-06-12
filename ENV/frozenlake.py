import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box


class FrozenLakeWrapper(gym.Wrapper):
    def __init__(self, render_mode=None):
        env = gym.make("FrozenLake-v1", render_mode=render_mode)
        super().__init__(env)
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(self.env.observation_space.n,),
            dtype=np.float32
        )

    def _one_hot(self, obs_int: int) -> np.ndarray:
        """将整数观测值转换为独热编码的numpy数组。"""
        # 创建一个全零向量
        one_hot_vector = np.zeros(self.observation_space.shape, dtype=np.float32)
        # 将对应位置的元素设为1
        one_hot_vector[obs_int] = 1.0
        return one_hot_vector

    def reset(self, **kwargs):
        """重写 reset 方法，返回独热编码的观测值。"""
        obs, info = self.env.reset(**kwargs)
        return self._one_hot(obs), info

    def step(self, action):
        """重写 step 方法，返回独热编码的观测值。"""
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._one_hot(obs), reward, terminated, truncated, info
