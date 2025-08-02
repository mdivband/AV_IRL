import gymnasium as gym
import numpy as np
from .safe_distance import calculate_safe_distance

class DrivingStyleRewardWrapper(gym.Wrapper):
    """
    - speed (log-scaled)
    - distance (exp-based safety shaping)
    - proximity penalty (hard threshold)
    """
    def __init__(self, env: gym.Env,
                 a: float = 1.0,
                 b: float = 1.0,
                 c: float = 1.0,
                 speed_max: float = 30.0,
                 distance_threshold: float = 2.0,
                 epsilon: float = 1e-6):
        super().__init__(env)
        self.a = a
        self.b = b
        self.c = c
        self.speed_max = speed_max
        self.distance_threshold = distance_threshold
        self.epsilon = epsilon

        self.reward_min = -self.c
        self.reward_max = self.a + self.b

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)

        vehicle = getattr(self.env.unwrapped, "vehicle", None)
        speed = getattr(vehicle, "speed", 0.0) if vehicle else 0.0
        reward_speed = np.log(1 + speed) / np.log(1 + self.speed_max)

        min_distance = calculate_safe_distance(info, env=self.env)

        reward_distance = np.exp(-2 / (min_distance + self.epsilon))

        penalty = self.c if min_distance < self.distance_threshold else 0.0

        raw_reward = self.a * reward_speed + self.b * reward_distance - penalty

        normalized_reward = (raw_reward - self.reward_min) / (self.reward_max - self.reward_min)

        return obs, normalized_reward, done, truncated, info
