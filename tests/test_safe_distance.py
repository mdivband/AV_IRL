import pytest
import gymnasium as gym
import numpy as np
from av_irl.safe_distance import (
    calculate_safe_distance,
    SafeDistanceRewardWrapper,
    RewardScaleWrapper,
    TimePenaltyWrapper,
)


def test_calculate_safe_distance_distance():
    class DummyVehicle:
        def __init__(self):
            self.position = [0.0, 0.0]

        def lane_distance_to(self, other, lane=None):
            return other.position[0] - self.position[0]

    class DummyRoad:
        def __init__(self):
            self.vehicles = []

        def neighbour_vehicles(self, vehicle, lane_index=None):
            lead = DummyVehicle()
            lead.position = [2.0, 0.0]
            return lead, None

        def close_vehicles_to(self, *args, **kwargs):
            return []

    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
            self.vehicle = DummyVehicle()
            self.road = DummyRoad()

        def reset(self, *, seed=None, options=None):
            return [0.0], {}

        def step(self, action):
            return [0.0], 1.0, True, False, {}

    env = DummyEnv()
    penalty = calculate_safe_distance({}, env)
    assert pytest.approx(penalty, 1e-6) == 0.5


def test_calculate_safe_distance_crash():
    info = {"crashed": True}
    assert calculate_safe_distance(info) == 1.0


def test_wrapper_penalty_applied():
    class DummyVehicle:
        def __init__(self):
            self.position = [0.0, 0.0]

        def lane_distance_to(self, other, lane=None):
            return other.position[0] - self.position[0]

    class DummyRoad:
        def __init__(self):
            self.vehicles = []

        def neighbour_vehicles(self, vehicle, lane_index=None):
            lead = DummyVehicle()
            lead.position = [1.0, 0.0]
            return lead, None

        def close_vehicles_to(self, *args, **kwargs):
            return []

    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
            self.step_called = False
            self.vehicle = DummyVehicle()
            self.road = DummyRoad()

        def reset(self, *, seed=None, options=None):
            return [0.0], {}

        def step(self, action):
            self.step_called = True
            return [0.0], 1.0, True, False, {}

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


def test_time_penalty_wrapper():
    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))

        def reset(self, *, seed=None, options=None):
            return [0.0], {}

        def step(self, action):
            return [0.0], 1.0, True, False, {}

    env = DummyEnv()
    wrapped = TimePenaltyWrapper(env, time_penalty=0.2)
    obs, reward, done, truncated, info = wrapped.step(0)
    assert reward == 0.8


def test_calculate_safe_distance_side_vehicle():
    class DummyVehicle:
        def __init__(self):
            self.position = [0.0, 0.0]
            self.lane_index = 0

        def lane_distance_to(self, other, lane=None):
            return other.position[0] - self.position[0]

    class DummyRoad:
        def __init__(self):
            self.network = self
            self.vehicles = []

        def get_lane(self, index):
            return self

        def local_coordinates(self, position):
            return position[0], 0.0

        def on_lane(self, position, s, lat, margin=1):
            return True

        def neighbour_vehicles(self, vehicle, lane_index=None):
            class Lead(DummyVehicle):
                def __init__(self):
                    super().__init__()
                    self.position = [2.0, 0.0]

            return Lead(), None

        def close_vehicles_to(self, vehicle, distance=float("inf"), count=1,
                              see_behind=True, sort=True):
            others = [v for v in self.vehicles if v is not vehicle]
            if not others:
                return []
            # Sort by Euclidean distance
            others.sort(key=lambda v: np.linalg.norm(np.array(v.position) - np.array(vehicle.position)))
            return others[:count]

    class DummyEnv(gym.Env):
        def __init__(self):
            self.action_space = gym.spaces.Discrete(1)
            self.observation_space = gym.spaces.Box(low=0, high=1, shape=(1,))
            self.vehicle = DummyVehicle()
            self.road = DummyRoad()
            # Add the ego and a side vehicle to the road
            side = DummyVehicle()
            side.position = [0.0, 0.5]
            self.road.vehicles = [self.vehicle, side]

        def reset(self, *, seed=None, options=None):
            return [0.0], {}

        def step(self, action):
            return [0.0], 1.0, False, False, {}

    env = DummyEnv()
    penalty = calculate_safe_distance({}, env)
    assert pytest.approx(penalty, 1e-5) == 2.0
    wrapped = SafeDistanceRewardWrapper(env)
    obs, reward, done, truncated, info = wrapped.step(0)
    assert pytest.approx(reward, 1e-5) == -1.0