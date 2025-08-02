import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from av_irl import DrivingStyleRewardWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import Wrapper
import argparse
from highway_env.envs.merge_env import MergeEnv
from stable_baselines3.common.callbacks import (
    EvalCallback,
    StopTrainingOnNoModelImprovement,
)
import torch
torch.backends.cudnn.benchmark = True

class StopTrainingOnNoModelImprovementAndLog(StopTrainingOnNoModelImprovement):
    """Stop training when there is no improvement and log the stop step."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stop_step = None

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if not continue_training:
            self.stop_step = self.parent.num_timesteps
            if self.verbose >= 1:
                print(f"Early stopping at {self.stop_step} steps")
        return continue_training



# a: 0.8, b: 0.2
# a: 0.2, b: 0.8
# a: 0.5, b: 0.5
def _silent_is_terminated(self) -> bool:
    return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
MergeEnv._is_terminated = _silent_is_terminated

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ts",
        type=int,
        default=int(1e5),
        help="Total timesteps to train the expert for each phase",
    )
    parser.add_argument("--a", type=float, default=1.0, help="Coefficient for speed term")
    parser.add_argument("--b", type=float, default=1.0, help="Coefficient for distance penalty")
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10000,
        help="Number of steps between evaluations",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Evaluations to wait for improvement before early stopping",
    )
    args = parser.parse_args()
    log_path = 'logs'
    model_name = 'model/expert_ppo_mlt_h1_m_h2'
    ts = args.ts
    a = args.a
    b = args.b
    eval_freq = args.eval_freq
    patience = args.patience
 
    # batch_size = 64
    # env = make_vec_env('highway-fast-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    
    def wrap_env(e):
        return DrivingStyleRewardWrapper(e, a, b)

    n_cpu = 8
    env = make_vec_env(
        'highway-fast-v0',
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0, 'simulation_frequency': 7, 'policy_frequency': 2, 'duration': 15}},
        wrapper_class=wrap_env,
    )

    print('Building Model')

    batch_size = 2048
    n_steps = 8192 
    policy_kwargs = dict(net_arch=dict(pi=[512, 256], vf=[512, 256]))


    expert = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=policy_kwargs,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log=log_path,
        ent_coef=0.01,
        device='cuda')

    print('Training in highway 1')
    BATCH_STEPS = n_cpu * n_steps
    eval_freq   = 2*BATCH_STEPS
    eval_env = make_vec_env(
        'highway-fast-v0',
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0, 'simulation_frequency': 7, 'policy_frequency': 2, 'duration': 15}},
        wrapper_class=wrap_env,
    )
    stop_callback = StopTrainingOnNoModelImprovementAndLog(patience, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        verbose=1,
    )
    expert.learn(total_timesteps=ts, callback=eval_callback)
    print("Trained timesteps (scenario 1, highway 1):", expert.num_timesteps)
    if stop_callback.stop_step is not None:
        print(f"Training stopped early at {stop_callback.stop_step} steps")
        expert.save(model_name)
        print('Saving Completed. (early stop)')
        exit()

    
    env_m = make_vec_env(
        'merge-v0',
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0, 'simulation_frequency': 7, 'policy_frequency': 2, 'duration': 15}},
        wrapper_class=wrap_env,
    )

    expert.set_env(env_m)
    print('Training in merge')
    eval_env = make_vec_env(
        'merge-v0',
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0, 'simulation_frequency': 7, 'policy_frequency': 2, 'duration': 15}},
        wrapper_class=wrap_env,
    )
    stop_callback = StopTrainingOnNoModelImprovementAndLog(patience, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        verbose=1,
    )
    expert.learn(total_timesteps=int(ts/4), callback=eval_callback)
    print("Trained timesteps (scenario 2, merge-v0):", expert.num_timesteps)
    if stop_callback.stop_step is not None:
        print(f"Training stopped early at {stop_callback.stop_step} steps")
        expert.save(model_name)
        print('Saving Completed. (early stop)')
        exit()
    
    env_h2 = make_vec_env(
        'highway-v0',
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0, 'simulation_frequency': 7, 'policy_frequency': 2, 'duration': 15}},
        wrapper_class=wrap_env,
    )

    expert.set_env(env_h2)
    print('Training in highway 2')
    eval_env = make_vec_env(
        'highway-v0',
        n_envs=4,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0, 'simulation_frequency': 7, 'policy_frequency': 2, 'duration': 15}},
        wrapper_class=wrap_env,
    )
    stop_callback = StopTrainingOnNoModelImprovementAndLog(patience, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_callback,
        eval_freq=eval_freq,
        n_eval_episodes=3,
        verbose=1,
    )
    expert.learn(total_timesteps=ts, callback=eval_callback)
    print("Trained timesteps (scenario 3, highway 2):", expert.num_timesteps)
    expert.save(model_name)
    print('Saving Completed. (scenario 3, highway 2)')
    if stop_callback.stop_step is not None:
        print(f"Training stopped early at {stop_callback.stop_step} steps")




