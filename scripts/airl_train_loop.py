import numpy as np
import pickle
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from av_irl import SafeDistanceRewardWrapper, TimePenaltyWrapper


def train_airl(env_name, rollout_filename, learner:PPO, rng, ts):
    def wrap_env(e, _):
        e = SafeDistanceRewardWrapper(e)
        e = TimePenaltyWrapper(e)
        return e

    venv = make_vec_env(
        env_name,
        n_envs=8,
        parallel=True,
        rng=rng,
        env_kwargs={"config": {"ego_spacing": 3.0}},
        post_wrappers=[wrap_env],
    )

    learner.set_env(venv)

    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )
    venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)
    print(f"open rollouts file: {rollout_filename}")

    with open(rollout_filename, "rb") as f:
        loaded_rollouts = pickle.load(f)

    print(f"loaded_rollouts length: {len(loaded_rollouts)}")
    airl_trainer = AIRL(
        demonstrations=loaded_rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=2,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        logdir=log_path
    )

    learner_reward_before_training, _ = evaluate_policy(learner, venv, 30)
    print(f"Reward before training: {learner_reward_before_training}")
    airl_trainer.train(int(ts))
    learner_reward_after_training, _ = evaluate_policy(learner, venv, 30)
    print(f"Reward after training: {learner_reward_after_training}")

if __name__ == '__main__':
    rng = np.random.default_rng()
    log_path = '/lyceum/tg4u22/project/output/'
    suffixes = ['1k', '2k', '4k', '8k', '16k', '32k', '64k']
    n_cpu = 6
    batch_size = 512
    ts = int(1e5)

    for suffix in suffixes:
        print(f"Training for suffix: {suffix}")

        env_name_h = "highway-fast-v0"

        venv = make_vec_env(
            env_name_h,
            n_envs=8,
            parallel=True,
            rng=rng,
            env_kwargs={"config": {"ego_spacing": 3.0}},
            post_wrappers=[wrap_env],
        )


        learner = PPO(
            'MlpPolicy',
            env=venv,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            n_steps=batch_size*12 // n_cpu,
            batch_size=batch_size,
            n_epochs=10,
            learning_rate=5e-4,
            gamma=0.8,
            verbose=2,
            tensorboard_log=log_path,
            log_path=log_path,
            device='cuda'
        )

        train_airl(
            env_name="highway-fast-v0",
            rollout_filename=f"/lyceum/tg4u22/project/rollouts_1_hf_{suffix}.pkl",
            learner=learner,
            rng=rng,
            ts=ts
        )

        train_airl(
            env_name="merge-v0",
            rollout_filename=f"/lyceum/tg4u22/project/rollouts_2_m2_{suffix}.pkl",
            learner=learner,
            rng=rng,
            ts=ts
        )

        model_name_suffix=f"e{suffix}_ts{ts}_ts{ts}_mlt_old"
        learner.save(f'/lyceum/tg4u22/project/model2/airl_learner_august_{model_name_suffix}')
