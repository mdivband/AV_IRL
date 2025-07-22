class DrivingStyleRewardWrapper(gym.Wrapper):
    """Reward = a * scaled_speed - b * (1 / distance to closest vehicle)."""
    def __init__(self, env: gym.Env, a: float = 1.0, b: float = 1.0, min_speed: float = 15.0, max_speed: float = 30.0):
        super().__init__(env)
        self.a = a
        self.b = b
        self.min_speed = min_speed
        self.max_speed = max_speed

    def scale_speed(self, speed):
        speed = max(self.min_speed, min(speed, self.max_speed))
        return (speed - self.min_speed) / (self.max_speed - self.min_speed)

    def step(self, action):
        obs, _, done, truncated, info = self.env.step(action)
        vehicle = getattr(self.env.unwrapped, "vehicle", None)
        speed = getattr(vehicle, "speed", 0.0) if vehicle else 0.0
        scaled_speed = self.scale_speed(speed)
        penalty = calculate_safe_distance(info, env=self.env)
        reward = self.a * scaled_speed - self.b * penalty
        return obs, reward, done, truncated, info
