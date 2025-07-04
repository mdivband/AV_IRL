import numpy as np
import pickle
from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_nets import BasicShapedRewardNet
from imitation.util.networks import RunningNorm
from imitation.util.util import make_vec_env
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from av_irl import ZeroRewardWrapper
import os
import logging
from highway_env.envs.merge_env import MergeEnv
import argparse


def _silent_is_terminated(self) -> bool:
    return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
MergeEnv._is_terminated = _silent_is_terminated


def train_airl(env_name, rollout_filename, learner: PPO, rng, ts, log_path: str):

    venv = make_vec_env(
        env_name,
        n_envs=8,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": {"ego_spacing": 3.0}},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )

    learner.set_env(venv)

    reward_net = BasicShapedRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
    )
    venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)
    logging.info(f"open rollouts file: {rollout_filename}")
    
    with open(rollout_filename, "rb") as f:
        loaded_rollouts = pickle.load(f)

    n_transitions = sum(len(traj) for traj in loaded_rollouts)
    demo_batch = min(1024, n_transitions)
    logging.info(
        "loaded %d trajectories totalling %d transitions, demo batch size %d",
        len(loaded_rollouts),
        n_transitions,
        demo_batch,
    )

    airl_trainer = AIRL(
        demonstrations=loaded_rollouts,
        demo_batch_size=demo_batch,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=2,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        log_dir=log_path,
        allow_variable_horizon=True,
    )

    learner_reward_before_training, _ = evaluate_policy(learner, venv, 30)
    logging.info(f"Reward before training: {learner_reward_before_training}")
    airl_trainer.train(int(ts))
    learner_reward_after_training, _ = evaluate_policy(learner, venv, 30)
    logging.info(f"Reward after training: {learner_reward_after_training}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    
    parser = argparse.ArgumentParser(description="train an airl learner")
    parser.add_argument("--ts", default= 1e5, help="timesteps")
    args = parser.parse_args()   
    
    rng = np.random.default_rng()
    # log_path = '/lyceum/tg4u22/project/output/'
    log_path = os.path.join(os.getcwd(), 'output')
    suffixes = ['1k', '2k', '4k', '8k', '16k', '32k', '64k', '128k']
    n_cpu = 6
    batch_size = 512
    ts = int(args.ts)

    for suffix in suffixes:
        logging.info(f"Training for suffix: {suffix}")

        env_name_h = "highway-fast-v0"

        venv = make_vec_env(
            env_name_h,
            n_envs=8,
            parallel=True,
            rng=rng,
            env_make_kwargs={"config": {"ego_spacing": 3.0}},
            post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
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
            device='cuda'
        )

        train_airl(
            env_name="highway-fast-v0",
            # rollout_filename=f"/lyceum/tg4u22/project/rollouts_1_hf_{suffix}.pkl",
            rollout_filename='./rollout/rollout_ts100_h2.pkl',
            learner=learner,
            rng=rng,
            ts=ts,
            log_path=log_path,
        )

        train_airl(
            env_name="merge-v0",
            # rollout_filename=f"/lyceum/tg4u22/project/rollouts_2_m2_{suffix}.pkl",
            rollout_filename='./rollout/rollout_ts100_h2.pkl',
            learner=learner,
            rng=rng,
            ts=ts,
            log_path=log_path,
        )

        model_name_suffix=f"e{suffix}_ts{ts}_ts{ts}_mlt_old"
        # learner.save(f'/lyceum/tg4u22/project/model/airl_learner_august_{model_name_suffix}')
         
        learner.save('model/airl_learner')