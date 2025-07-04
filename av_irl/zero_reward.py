import gymnasium as gym

class ZeroRewardWrapper(gym.Wrapper):
    """Replace environment reward with zero at every step."""

    def step(self, action):
        obs, _rew, done, truncated, info = self.env.step(action)
        return obs, 0.0, done, truncated, info
