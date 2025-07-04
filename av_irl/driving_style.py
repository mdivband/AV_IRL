import gymnasium as gym
import numpy as np
from .safe_distance import calculate_safe_distance

class DrivingStyleRewardWrapper(gym.Wrapper):
    """Reward = a * speed - b * (1 / distance to closest vehicle)."""
    def __init__(self, env: gym.Env, a: float = 1.0, b: float = 1.0):
        super().__init__(env)
        self.a = a
        self.b = b

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        speed = getattr(self.env.unwrapped, "vehicle", None)
        speed = getattr(speed, "speed", 0.0)
        penalty = calculate_safe_distance(info, env=self.env)
        reward = self.a * float(speed) - self.b * penalty
        return obs, reward, done, truncated, info