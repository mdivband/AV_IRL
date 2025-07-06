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
from typing import Optional
from highway_env.envs.merge_env import MergeEnv
import argparse


def _silent_is_terminated(self) -> bool:
    return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
MergeEnv._is_terminated = _silent_is_terminated


class EarlyStopping:
    """Simple early stopping for adversarial training."""

    def __init__(self, patience: int, eval_env):
        self.patience = patience
        self.eval_env = eval_env
        self.best_reward = -float("inf")
        self.wait = 0
        self.stop_step: Optional[int] = None

    def __call__(self, round_num: int, trainer: AIRL) -> None:
        step = (round_num + 1) * trainer.gen_train_timesteps
        mean_reward, _ = evaluate_policy(trainer.gen_algo, self.eval_env, 5)
        if mean_reward > self.best_reward:
            self.best_reward = mean_reward
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stop_step = step
                print(f"Early stopping at {step} steps")
                raise StopIteration



def train_airl(
    env_name: str,
    rollout_filename: str,
    learner: PPO,
    rng,
    ts: int,
    log_path: str,
    patience: int,
) -> Optional[int]:

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

    eval_env = make_vec_env(
        env_name,
        n_envs=1,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": {"ego_spacing": 3.0}},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )
    eval_env = RewardVecEnvWrapper(eval_env, reward_net.predict_processed)

    stopper = EarlyStopping(patience, eval_env)
    try:
        airl_trainer.train(int(ts), callback=lambda r: stopper(r, airl_trainer))
    except StopIteration:
        pass

    learner_reward_after_training, _ = evaluate_policy(learner, venv, 30)
    logging.info(f"Reward after training: {learner_reward_after_training}")

    return stopper.stop_step

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="train an airl learner")
    parser.add_argument("--env", default="highway-fast-v0", help="environment name")
    parser.add_argument("--rollout", required=True, help="rollout pickle file")
    parser.add_argument("--out", default="model/airl_learner.zip", help="path to save the learner")
    parser.add_argument("--ts", type=int, default=100000, help="training timesteps")
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="evaluations to wait for improvement before stopping",
    )
    args = parser.parse_args()

    rng = np.random.default_rng()
    log_path = os.path.join(os.getcwd(), 'output')

    venv = make_vec_env(
        args.env,
        n_envs=8,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": {"ego_spacing": 3.0}},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )

    batch_size = 1024
    n_steps = 8192
    policy_kwargs = dict(
        net_arch=dict(
            pi=[1024, 1024, 512],
            vf=[1024, 1024, 512]
        )
    )

    learner = PPO(
        'MlpPolicy',
        env=venv,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.95,
        verbose=2,
        tensorboard_log=log_path,
        ent_coef=0.01,
        device='cuda')

    stop_step = train_airl(
        env_name=args.env,
        rollout_filename=args.rollout,
        learner=learner,
        rng=rng,
        ts=args.ts,
        log_path=log_path,
        patience=args.patience,
    )

    if stop_step is not None:
        logging.info("Training stopped early at %d steps", stop_step)

    learner.save(args.out)

