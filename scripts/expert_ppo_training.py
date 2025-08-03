import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement
from highway_env.envs.merge_env import MergeEnv
from av_irl import DrivingStyleRewardWrapper

torch.backends.cudnn.benchmark = True

class StopTrainingOnNoModelImprovementAndLog(StopTrainingOnNoModelImprovement):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_step = None

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not continue_training:
            self.stop_step = self.parent.num_timesteps
            if self.verbose >= 1:
                print(f"[Callback] Early stopping at {self.stop_step} steps")
        return continue_training


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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ts", type=int, default=int(1e5),
                        help="Total timesteps per phase")
    parser.add_argument("--a", type=float, default=1.0,
                        help="Coefficient for speed reward")
    parser.add_argument("--b", type=float, default=1.0,
                        help="Coefficient for distance reward")
    parser.add_argument("--eval-freq", type=int, default=10000,
                        help="Steps between evaluations")
    parser.add_argument("--patience", type=int, default=5,
                        help="Evals to wait before early-stop")
    args = parser.parse_args()

    log_path   = "logs"
    model_name = "model/expert_ppo_mlt_h1_m_h2"

    ts        = args.ts
    a, b      = args.a, args.b
    patience  = args.patience

    def wrap_env(env):
        return DrivingStyleRewardWrapper(
            env,
            a=a,
            b=b,
        )

    n_cpu = 12
    env = make_vec_env(
        "highway-fast-v0",
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(config=KINEMATICS_REL_CFG),
        wrapper_class=wrap_env,
    )

    expert = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=dict(pi=[512, 256], vf=[512, 256])),
        n_steps=8192,
        batch_size=2048,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        ent_coef=0.01,
        tensorboard_log=log_path,
        verbose=2,
        device="cuda",
    )

    batch_steps = n_cpu * 8192
    eval_freq   = 2 * batch_steps
    eval_env = make_vec_env(
        "highway-fast-v0",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(config=KINEMATICS_REL_CFG),
        wrapper_class=wrap_env,
    )
    stop_cb = StopTrainingOnNoModelImprovementAndLog(patience, verbose=1)
    eval_cb = EvalCallback(
        eval_env,
        callback_on_new_best=stop_cb,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        verbose=1,
    )

    print("=== Training Phase 1: highway-fast-v0 ===")
    expert.learn(total_timesteps=ts, callback=eval_cb)
    print("Phase-1 timesteps:", expert.num_timesteps)

    if stop_cb.stop_step:
        expert.save(model_name)
        print("Saved after early stop.")
        exit()

    env_m = make_vec_env(
        "merge-v0",
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(config=KINEMATICS_REL_CFG),
        wrapper_class=wrap_env,
    )
    expert.set_env(env_m)

    eval_env_m = make_vec_env(
        "merge-v0",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(config=KINEMATICS_REL_CFG),
        wrapper_class=wrap_env,
    )
    stop_cb = StopTrainingOnNoModelImprovementAndLog(patience, verbose=1)
    eval_cb = EvalCallback(eval_env_m, callback_on_new_best=stop_cb,
                           eval_freq=eval_freq, n_eval_episodes=3, verbose=1)

    print("=== Training Phase 2: merge-v0 ===")
    expert.learn(total_timesteps=int(ts / 4), callback=eval_cb)
    print("Phase-2 timesteps:", expert.num_timesteps)

    if stop_cb.stop_step:
        expert.save(model_name)
        exit()

    env_h2 = make_vec_env(
        "highway-v0",
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(config=KINEMATICS_REL_CFG),
        wrapper_class=wrap_env,
    )
    expert.set_env(env_h2)

    eval_env_h2 = make_vec_env(
        "highway-v0",
        n_envs=8,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=dict(config=KINEMATICS_REL_CFG),
        wrapper_class=wrap_env,
    )
    stop_cb = StopTrainingOnNoModelImprovementAndLog(patience, verbose=1)
    eval_cb = EvalCallback(eval_env_h2, callback_on_new_best=stop_cb,
                           eval_freq=eval_freq, n_eval_episodes=3, verbose=1)

    print("=== Training Phase 3: highway-v0 ===")
    expert.learn(total_timesteps=ts, callback=eval_cb)
    print("Phase-3 timesteps:", expert.num_timesteps)

    expert.save(model_name)
    print("Model saved successfully.")
