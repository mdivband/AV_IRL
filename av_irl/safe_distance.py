import gymnasium as gym
import numpy as np
from typing import Optional

def _distance_to_lead_from_env(env: gym.Env) -> Optional[float]:
    try:
        front, _ = env.road.neighbour_vehicles(env.vehicle)
        if front is None:
            return None
        return env.vehicle.lane_distance_to(front)
    except Exception:
        return None

def _nearest_vehicle_distance(env: gym.Env) -> Optional[float]:
    try:
        ego = env.vehicle
        near = env.road.close_vehicles_to(ego, distance=float("inf"), count=1)
        if not near:
            return None
        other = near[0]
        return np.linalg.norm(np.array(other.position) - np.array(ego.position))
    except Exception:
        return None


def calculate_safe_distance(info: dict, env: Optional[gym.Env] = None) -> float:

    if info.get("crashed"):
        return 1.0

    if env is None:
        return 0.0

    distance_lead = _distance_to_lead_from_env(env)
    distance_side = _nearest_vehicle_distance(env)

    distances = [d for d in (distance_lead, distance_side) if d is not None]
    if not distances:
        return 0.0

    distance = min(distances)
    return 1.0 / (distance + 1e-6)

class SafeDistanceRewardWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, weight: float = 1.0):
        super().__init__(env)
        self.weight = weight

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        penalty = calculate_safe_distance(info, env=self.env)
        reward -= self.weight * penalty
        return obs, reward, done, truncated, info


class RewardScaleWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward *= self.scale
        return obs, reward, done, truncated, info


class TimePenaltyWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, time_penalty: float = 0.01):
        super().__init__(env)
        self.time_penalty = time_penalty

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward -= self.time_penalty
        return obs, reward, done, truncated, info