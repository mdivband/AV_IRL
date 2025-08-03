#!/usr/bin/env python3
"""
AIRL with Slot-Attention discriminator.
Relative-state Kinematics observation (Highway-env).
SlotAttention(K=3, D=32) encoder.
PPO generator.
"""

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="pygame.pkgdata"
)  # quiet highway-env

import os
import logging
import argparse
import pickle
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms.adversarial.airl import AIRL
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.util.util import make_vec_env
from imitation.rewards.reward_nets import RewardNet
import gymnasium as gym

from highway_env.envs.merge_env import MergeEnv
from av_irl import ZeroRewardWrapper
torch.backends.cudnn.benchmark = True

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
        "vehicles_count": 5,
    },
}

#N_VEHICLES = KINEMATICS_REL_CFG["observation"]["vehicles_count"]
F_DIM = len(KINEMATICS_REL_CFG["observation"]["features"])

class SlotAttention(nn.Module):
    def __init__(self, num_slots: int = 4, dim: int = 32, iters: int = 3):
        super().__init__()
        self.num_slots, self.iters, self.dim = num_slots, iters, dim
        self.scale = dim**-0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_logsigma = nn.Parameter(torch.zeros(1, 1, dim))

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.gru = nn.GRUCell(dim, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)
        )

        self.norm_x = nn.LayerNorm(dim)
        self.norm_slots = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, D = x.shape
        x = self.norm_x(x)

        mu = self.slots_mu.expand(B, self.num_slots, -1)
        sigma = self.slots_logsigma.exp().expand_as(mu)
        slots = mu + torch.randn_like(mu) * sigma

        for _ in range(self.iters):
            slots_prev = slots

            q = self.to_q(self.norm_slots(slots))
            k = self.to_k(x)
            v = self.to_v(x)

            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) * self.scale
            attn = attn_logits.softmax(dim=-1) + 1e-8 
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum("bkn,bnd->bkd", attn, v)

            slots = self.gru(
                updates.reshape(-1, D), slots_prev.reshape(-1, D)
            ).view(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots.reshape(B, self.num_slots * D)

class SlotRewardNet(RewardNet):
    def __init__(self, obs_space, act_space,
                 num_slots=4, slot_dim=32, hidden=64):
        super().__init__(obs_space, act_space)

        self.F = 5
        obs_dim = obs_space.shape[-1]
        assert obs_dim % self.F == 0
        self.N = obs_dim // self.F

        if isinstance(act_space, gym.spaces.Box):
            act_dim = int(np.prod(act_space.shape))
        elif isinstance(act_space, gym.spaces.Discrete):
            act_dim = 1
        else:
            raise TypeError(f"Unsupported action space: {act_space}")

        self.encoder = SlotAttention(num_slots=num_slots, dim=slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(num_slots * slot_dim + act_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, obs: torch.Tensor, acts: torch.Tensor, **kwargs):
        B = obs.shape[0]
        x = obs.view(B, self.N, self.F)
        slot_vec = self.encoder(x)
        cat = torch.cat([slot_vec, acts], dim=-1)
        return self.mlp(cat)

    def reward(self, obs, acts, **kwargs):
        return self.forward(obs, acts)


class EarlyStopping:
    def __init__(self, patience: int, eval_env):
        self.patience = patience
        self.eval_env = eval_env
        self.best, self.wait, self.stop_step = -float("inf"), 0, None

    def __call__(self, round_num: int, trainer: AIRL) -> None:
        step = (round_num + 1) * trainer.gen_train_timesteps
        mean_r, _ = evaluate_policy(trainer.gen_algo, self.eval_env, 5)
        if mean_r > self.best:
            self.best, self.wait = mean_r, 0
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stop_step = step
                print(f"[EarlyStop] no improvement -> stop at {step}")
                raise StopIteration


def train_airl_once(
    env_name: str,
    rollout_pkl: str,
    learner: PPO,
    rng,
    timesteps: int,
    log_dir: str,
    patience: int,
) -> SlotRewardNet:
    venv = make_vec_env(
        env_name,
        n_envs=12,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": KINEMATICS_REL_CFG},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )
    learner.set_env(venv)

    reward_net = SlotRewardNet(
        venv.observation_space,
        venv.action_space,
        num_slots=3,
        slot_dim=32,
        hidden=64,
    )
    venv = RewardVecEnvWrapper(venv, reward_net.reward)

    with open(rollout_pkl, "rb") as f:
        demos = pickle.load(f)
    demo_batch = min(1024, sum(len(t) for t in demos))

    trainer = AIRL(
        demonstrations=demos,
        demo_batch_size=demo_batch,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        log_dir=log_dir,
        n_disc_updates_per_round=2,
        gen_replay_buffer_capacity=2048,
        allow_variable_horizon=True,
    )

    eval_env = make_vec_env(
        env_name,
        n_envs=8,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": KINEMATICS_REL_CFG},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )
    eval_env = RewardVecEnvWrapper(eval_env, reward_net.reward)
    stopper = EarlyStopping(patience, eval_env)

    try:
        trainer.train(int(timesteps), callback=lambda r: stopper(r, trainer))
    except StopIteration:
        pass

    return reward_net


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="highway-fast-v0")
    parser.add_argument("--out", default="model/airl_learner.zip")
    parser.add_argument("--ts", type=int, default=100_000)
    parser.add_argument("--a", type=float, default=1)
    parser.add_argument("--b", type=float, default=1)
    parser.add_argument("--size", type=int, default=8000)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    rng = np.random.default_rng()
    log_dir = os.path.join(os.getcwd(), "output")

    dummy_env = make_vec_env(
        args.env,
        n_envs=12,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": KINEMATICS_REL_CFG},
        post_wrappers=[lambda e, _: ZeroRewardWrapper(e)],
    )
    learner = PPO(
        "MlpPolicy",
        env=dummy_env,
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

    train_airl_once(
        env_name="highway-fast-v0",
        rollout_pkl=f"rollout/hf_a{args.a}_b{args.b}_{args.size}.pkl",
        learner=learner,
        rng=rng,
        timesteps=args.ts,
        log_dir=log_dir,
        patience=args.patience,
    )

    reward_net_final = train_airl_once(
        env_name="merge-v0",
        rollout_pkl=f"rollout/mg_a{args.a}_b{args.b}_{args.size}.pkl",
        learner=learner,
        rng=rng,
        timesteps=args.ts,
        log_dir=log_dir,
        patience=args.patience,
    )

    learner.save(args.out)
    reward_path = pathlib.Path(args.out).with_suffix("").with_suffix("_reward.pt")
    torch.save(reward_net_final, reward_path)
    print("Saved policy ->", args.out)
    print("Saved reward ->", reward_path)


if __name__ == "__main__":
    main()
