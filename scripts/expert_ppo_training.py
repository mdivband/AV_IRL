from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from av_irl import DrivingStyleRewardWrapper
from stable_baselines3.common.vec_env import SubprocVecEnv
from gymnasium import Wrapper
import argparse
from highway_env.envs.merge_env import MergeEnv

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
    args = parser.parse_args()
    log_path = 'logs'
    model_name = 'model/expert_ppo_mlt_h1_m_h2'
    ts = args.ts
    a = args.a
    b = args.b
    train = True
    if train:
        n_cpu = 6
        batch_size = 512
        # batch_size = 64
        # env = make_vec_env('highway-fast-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
        
        def wrap_env(e):
            return DrivingStyleRewardWrapper(e, a, b)


        env = make_vec_env(
            'highway-fast-v0',
            n_envs=n_cpu,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={'config': {'ego_spacing': 3.0}},
            wrapper_class=wrap_env,
        )

    print('Building Model')

    expert = PPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size*12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.9,
        verbose=2,
        tensorboard_log=log_path,
        device='cuda')

    print('Training in highway 1')
    expert.learn(total_timesteps=ts)
    # print('Training Completed (scenario 1, highway 1). Saving...')
    print("Trained timesteps (scenario 1, highway 1):", expert.num_timesteps)
    # expert.save(model_name)
    # print('Saving Completed. (scenario 1, highway 1)')

    # using a different env to continue train the expert, use merge env
    # env_m = make_vec_env('merge-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    
    env_m = make_vec_env(
        'merge-v0',
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0}},
        wrapper_class=wrap_env,
    )

    # expert2 = PPO.load(model_name)
    expert.set_env(env_m)
    print('Training in merge')
    expert.learn(total_timesteps=ts)
    # print('Training Completed (scenario 2, merge-v0). Saving...')
    print("Trained timesteps (scenario 2, merge-v0):", expert.num_timesteps)
    # expert2.save(model_name)
    # print('Saving Completed. (scenario 2, merge-v0)')

    # using a different env to continue train the expert, use more vehicles, 50->70
    # env_h2 = make_vec_env('highway-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv, env_kwargs={'vehicles_count': 70})
    # env_h2 = make_vec_env('highway-v0', n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    
    env_h2 = make_vec_env(
        'highway-v0',
        n_envs=n_cpu,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={'config': {'ego_spacing': 3.0}},
        wrapper_class=wrap_env,
    )

    # expert3 = PPO.load(model_name)
    expert.set_env(env_h2)
    print('Training in highway 2')
    expert.learn(total_timesteps=ts)
    # print('Training Completed (scenario 3, highway 2). Saving...')
    print("Trained timesteps (scenario 3, highway 2):", expert.num_timesteps)
    expert.save(model_name)
    print('Saving Completed. (scenario 3, highway 2)')



