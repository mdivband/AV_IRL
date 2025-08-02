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
    distance_lead = _distance_to_lead_from_env(env)
    distance_side = _nearest_vehicle_distance(env)
    distances = [d for d in (distance_lead, distance_side) if d is not None]
    return min(distances) if distances else None
