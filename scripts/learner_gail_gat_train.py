import os, logging, argparse, pickle, pathlib
from typing import Optional
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.util import make_vec_env
from av_irl import ZeroRewardWrapper
from av_irl.gat import GATRewardNet
from highway_env.envs.merge_env import MergeEnv

# Silence highway termination warning
def _silent_is_terminated(self) -> bool:
    return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
MergeEnv._is_terminated = _silent_is_terminated

KINEMATICS_REL_CFG = {
    "ego_spacing": 3.0,
    "observation": {
        "type": "Kinematics",
        "features": ["presence", "x", "y", "vx", "vy"],
        "absolute": False,
        "normalize": True,
        "vehicles_count": 5
    }
}

class EarlyStopping:
    def __init__(self, patience: int, eval_env):
        self.patience = patience
        self.eval_env = eval_env
        self.best_reward = -float("inf")
        self.wait = 0
        self.stop_step: Optional[int] = None

    def __call__(self, round_num: int, trainer: GAIL) -> None:
        step = (round_num + 1) * trainer.gen_train_timesteps
        mean_r, _ = evaluate_policy(trainer.gen_algo, self.eval_env, 5)
        if mean_r > self.best_reward:
            self.best_reward = mean_r
            self.wait = 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stop_step = step
                print(f"[EarlyStop] Stopping at {step} steps")
                raise StopIteration

def train_gail_gat(
    env_name: str,
    rollout_filename: str,
    learner: PPO,
    rng,
    ts: int,
    log_path: str,
    patience: int,
):
    venv = make_vec_env(
        env_name,
        n_envs=8,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": KINEMATICS_REL_CFG},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )
    learner.set_env(venv)

    reward_net = GATRewardNet(
        venv.observation_space,
        venv.action_space,
        feature_dim=5,   # presence, x, y, vx, vy
        hidden_gat=32,
        hidden_mlp=64,
        use_running_norm=True,
    )
    venv = RewardVecEnvWrapper(venv, reward_net.predict_processed)

    logging.info("Loading rollouts from %s", rollout_filename)
    with open(rollout_filename, "rb") as f:
        demos = pickle.load(f)

    n_transitions = sum(len(t) for t in demos)
    demo_batch = min(1024, n_transitions)

    gail_trainer = GAIL(
        demonstrations=demos,
        demo_batch_size=demo_batch,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=2,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        log_dir=log_path,
        allow_variable_horizon=True,
    )

    eval_env = make_vec_env(
        env_name,
        n_envs=4,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": KINEMATICS_REL_CFG},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )
    eval_env = RewardVecEnvWrapper(eval_env, reward_net.predict_processed)
    stopper = EarlyStopping(patience, eval_env)

    try:
        gail_trainer.train(int(ts), callback=lambda r: stopper(r, gail_trainer))
    except StopIteration:
        pass

    return reward_net

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="highway-fast-v0")
    parser.add_argument("--out", default="model/gail_gat.zip")
    parser.add_argument("--rout", default="model/gail_gat_reward.pt")
    parser.add_argument("--ts", type=int, default=100000)
    parser.add_argument("--a", type=float, default=1)
    parser.add_argument("--b", type=float, default=1)
    parser.add_argument("--size", type=int, default=8000)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng()
    log_dir = os.path.join(os.getcwd(), "output")

    venv_init = make_vec_env(
        args.env,
        n_envs=8,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": KINEMATICS_REL_CFG},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )

    learner = PPO(
        "MlpPolicy",
        env=venv_init,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),
        n_steps=8192,
        batch_size=1024,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log=log_dir,
        ent_coef=0.01,
        device="cuda",
    )

    logging.info("Training GAT-GAIL on highway-fast-v0")
    reward_net_hf = train_gail_gat(
        env_name="highway-fast-v0",
        rollout_filename=f"rollout/hf_a{args.a}_b{args.b}_{args.size}.pkl",
        learner=learner,
        rng=rng,
        ts=args.ts,
        log_path=log_dir,
        patience=args.patience,
    )

    logging.info("Training GAT-GAIL on merge-v0")
    reward_net_mg = train_gail_gat(
        env_name="merge-v0",
        rollout_filename=f"rollout/mg_a{args.a}_b{args.b}_{args.size}.pkl",
        learner=learner,
        rng=rng,
        ts=args.ts,
        log_path=log_dir,
        patience=args.patience,
    )

    learner.save(args.out)
    print("Saved GAT-GAIL learner ->", args.out)
    
    # Save the final reward network
    reward_path = args.rout
    torch.save(reward_net_mg, reward_path)
    print("Saved GAT reward network ->", reward_path)
