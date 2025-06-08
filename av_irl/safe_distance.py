import gymnasium as gym


def calculate_safe_distance(info: dict) -> float:
    """Return a continuous safe distance penalty from env ``info``.

    If ``distance_to_lead`` is provided in ``info`` it is inverted to obtain a
    penalty that increases as the vehicle gets closer to others. When a
    collision has occurred the penalty is ``1.0``. If no distance information is
    available a penalty of ``0.0`` is returned.
    """
    if info.get("crashed"):
        return 1.0
    distance = info.get("distance_to_lead")
    if distance is None:
        return 0.0
    return 1.0 / (distance + 1e-6)


class SafeDistanceRewardWrapper(gym.Wrapper):
    """Wrapper that subtracts a continuous safe distance penalty from the reward."""

    def __init__(self, env: gym.Env, weight: float = 1.0):
        super().__init__(env)
        self.weight = weight

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        penalty = calculate_safe_distance(info)
        reward -= self.weight * penalty
        return obs, reward, done, truncated, info


class RewardScaleWrapper(gym.Wrapper):
    """Scale the base environment reward by ``scale`` before other wrappers."""

    def __init__(self, env: gym.Env, scale: float = 1.0):
        super().__init__(env)
        self.scale = scale

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        reward *= self.scale
        return obs, reward, done, truncated, info