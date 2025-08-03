import gymnasium as gym
import numpy as np
from typing import Optional


def _distance_to_lead_from_env(env: gym.Env) -> Optional[float]:
    try:
        base = env.unwrapped
        front, _ = base.road.neighbour_vehicles(base.vehicle)
        if front is None:
            return None
        return base.vehicle.lane_distance_to(front)
    except Exception:
        return None

def _nearest_vehicle_distance(env: gym.Env) -> Optional[float]:
    try:
        base = env.unwrapped
        ego = base.vehicle
        near = base.road.close_vehicles_to(ego, distance=float("inf"), count=1)
        if not near:
            return None
        other = near[0]
        return np.linalg.norm(np.array(other.position) - np.array(ego.position))
    except Exception:
        return None

def calculate_safe_distance(info: dict, env: Optional[gym.Env] = None) -> Optional[float]:
    if info.get("crashed"):
        return 0.0
    if env is None:
        return None
    d_lead = _distance_to_lead_from_env(env)
    d_side = _nearest_vehicle_distance(env)
    valid = [d for d in (d_lead, d_side) if d is not None]
    return min(valid) if valid else None



class DrivingStyleRewardWrapper(gym.Wrapper):
    def __init__(
        self,
        env: gym.Env,
        a: float = 0.5,
        b: float = 0.5,
        c: float = 1.0,
        lane_penalty: float = 0.1,
        speed_max: float = 30.0,
        distance_threshold: float = 2.0,
        epsilon: float = 1e-6,
        k: float = 2.0,
    ):
        super().__init__(env)

        self.a = a
        self.b = b
        self.c = c
        self.lane_penalty = lane_penalty
        self.speed_max = speed_max
        self.distance_threshold = distance_threshold
        self.epsilon = epsilon
        self.k = k

        self.prev_lane_index = None
        
        base_env = env.unwrapped
        if hasattr(base_env, "configure"):
            base_env.configure({
                "observation": {
                    "type": "Kinematics",
                    "features": ["presence", "x", "y", "vx", "vy"],
                    "absolute": False,
                    "normalize": True,
                    "vehicles_count": 5
                }
            })


        self.reward_min = -self.c - self.lane_penalty
        self.reward_max = self.a + self.b

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        ego = getattr(self.env.unwrapped, "vehicle", None)
        self.prev_lane_index = ego.lane_index[2] if ego else None
        return obs

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)

        ego = getattr(self.env.unwrapped, "vehicle", None)

        speed = getattr(ego, "speed", 0.0) if ego else 0.0
        reward_speed = np.log(1 + speed) / np.log(1 + self.speed_max)

        min_dist = calculate_safe_distance(info, env=self.env)
        reward_distance = np.exp(-self.k / (min_dist + self.epsilon))

        hard_penalty = self.c if min_dist < self.distance_threshold else 0.0

        lane_pen = 0.0
        if ego:
            current_lane = ego.lane_index[2]
            if self.prev_lane_index is not None and current_lane != self.prev_lane_index:
                lane_pen = self.lane_penalty
            self.prev_lane_index = current_lane

        raw_reward = (
            self.a * reward_speed
            + self.b * reward_distance
            - hard_penalty
            - lane_pen
        )

        normalized_reward = (raw_reward - self.reward_min) / (self.reward_max - self.reward_min)

        return obs, normalized_reward, done, truncated, info
