import gym
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

def evaluate_average_reward(model, env, n_eval_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    return mean_reward, std_reward

def collect_trajectory(model, env, max_steps=200):
    obs = env.reset()
    done = False
    traj = []

    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        try:
            position = env.unwrapped.vehicle.position
            traj.append(position)
        except:
            break
        if done or truncated:
            break
    return np.array(traj)

def run_comparison(model_paths, labels, env_name="highway-fast-v0", output_dir="/mnt/data", max_steps=200):
    env = gym.make(env_name)
    env.config["duration"] = max_steps
    env.reset()

    rewards = []
    trajectories = []

    for path in model_paths:
        model = PPO.load(path)
        mean_reward, std_reward = evaluate_average_reward(model, env)
        rewards.append((mean_reward, std_reward))
        traj = collect_trajectory(model, env, max_steps)
        trajectories.append(traj)

    # Plot reward comparison
    plt.figure(figsize=(6, 4))
    means = [r[0] for r in rewards]
    stds = [r[1] for r in rewards]
    x = np.arange(len(labels))
    plt.bar(x, means, yerr=stds, capsize=5, color=['green', 'blue', 'orange'])
    plt.xticks(x, labels)
    plt.ylabel("Average Episode Reward")
    plt.title("Reward Comparison")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/reward_comparison_bar.png")
    plt.close()

    # Plot trajectory comparison
    plt.figure(figsize=(7, 5))
    for traj, label, color in zip(trajectories, labels, ['green', 'blue', 'orange']):
        if len(traj) > 0:
            plt.plot(traj[:, 0], traj[:, 1], label=label, linewidth=2, alpha=0.8, color=color)
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/trajectory_comparison.png")
    plt.close()

    return f"{output_dir}/reward_comparison_bar.png", f"{output_dir}/trajectory_comparison.png"

model_paths = [
    "./model/expert_ppo_mlt_h1_m_h2.zip",
    "./model/gail_learner.zip",
    "./model/airl_learner.zip"
]

labels = ["Expert", "GAIL", "AIRL"]

run_comparison(
    model_paths=model_paths,
    labels=labels,
    env_name="highway-fast-v0",
    output_dir="./analysis",
    max_steps=200
)
