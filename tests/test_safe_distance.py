import pytest
import gymnasium as gym
from av_irl.safe_distance import (
    calculate_safe_distance,
    SafeDistanceRewardWrapper,
    RewardScaleWrapper,
)


def test_calculate_safe_distance_distance():
    info = {"distance_to_lead": 2.0}
    assert pytest.approx(calculate_safe_distance(info), 1e-6) == 0.5


def test_calculate_safe_distance_crash():
    info = {"crashed": True}
    assert calculate_safe_distance(info) == 1.0


def test_wrapper_penalty_applied():
    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
            self.step_called = False

        def reset(self, *, seed=None, options=None):
            return [0.0], {}

        def step(self, action):
            self.step_called = True
            return [0.0], 1.0, True, False, {"distance_to_lead": 1.0}

    env = DummyEnv()
    wrapped = SafeDistanceRewardWrapper(env)
    obs, reward, done, truncated, info = wrapped.step(0)
    assert reward < 1.0
    assert done
    assert env.step_called


def test_reward_scale_wrapper():
    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))

        def reset(self, *, seed=None, options=None):
            return [0.0], {}

        def step(self, action):
            return [0.0], 2.0, True, False, {}

    env = DummyEnv()
    wrapped = RewardScaleWrapper(env, scale=0.5)
    obs, reward, done, truncated, info = wrapped.step(0)
    assert reward == 1.0