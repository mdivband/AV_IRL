import numpy as np
from stable_baselines3 import PPO
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.util import make_vec_env
from av_irl import SafeDistanceRewardWrapper, RewardScaleWrapper, TimePenaltyWrapper


def train_airl_with_coeff(a: float, b: float, steps: int = 1000):
    env = make_vec_env(
        "highway-fast-v0",
        n_envs=1,
        env_kwargs={"initial_spacing": 2.0},
    )
    env = RewardScaleWrapper(env, scale=a)
    env = SafeDistanceRewardWrapper(env, weight=b)
    env = TimePenaltyWrapper(env)

    learner = PPO("MlpPolicy", env, n_steps=64, batch_size=64, verbose=0)
    reward_net = BasicShapedRewardNet(env.observation_space, env.action_space)

    airl = AIRL(demonstrations=[], venv=env, gen_algo=learner, reward_net=reward_net)
    airl.train(steps)
    return reward_net


def main():
    net_a = train_airl_with_coeff(1.0, 0.5)
    net_b = train_airl_with_coeff(0.5, 1.0)

    dummy_obs = np.zeros(net_a.observation_space.shape)[None]
    print("Reward estimate (a=1.0,b=0.5):", net_a.predict_processed(dummy_obs)[0])
    print("Reward estimate (a=0.5,b=1.0):", net_b.predict_processed(dummy_obs)[0])


if __name__ == "__main__":
    main()