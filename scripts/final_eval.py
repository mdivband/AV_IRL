import argparse
import random
from typing import List, Tuple

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

# Default model locations. Override on the command line if needed.
DEFAULT_EXPERT_PATH = "model/manual_test/expert_ppo_mlt_h1_m_h2"
DEFAULT_LEARNER_PATH = "model/manual_test/expert_ppo_mlt_h1_m_h2"


def make_env(env_code: str) -> gym.Env:
    """Create an unwrapped evaluation environment."""

    if env_code == "r":
        return gym.make("roundabout-v0", render_mode="rgb_array")
    if env_code == "i":
        return gym.make("intersection-v1", render_mode="rgb_array")
    if env_code == "h2":
        return gym.make("highway-v0", render_mode="rgb_array")
    if env_code == "h":
        env = gym.make("highway-fast-v0", render_mode="rgb_array")
        env.config["vehicles_count"] = 30
        env.config["vehicles_density"] = 1.5
        return env
    if env_code == "h3":
        env = gym.make("highway-v0", render_mode="rgb_array")
        env.config["vehicles_count"] = 60
        env.config["vehicles_density"] = 1.2
        env.config["lanes_count"] = 5
        return env
    if env_code == "m":
        return gym.make("merge-v0", render_mode="rgb_array")
    raise ValueError(f"Unknown environment code: {env_code}")


def generate_seeds(n: int) -> List[int]:
    rng = random.Random(0)
    return [rng.randrange(0, 2 ** 32) for _ in range(n)]


def run_episode(
    model: PPO, env: gym.Env, seed: int, render: bool = False
) -> Tuple[float, List[Tuple[int, int]], List[Tuple[int, float]]]:
    """Run one episode and return the raw environment rewards."""

    obs, info = env.reset(seed=seed)
    done = truncated = False
    total_reward = 0.0
    actions = []
    rewards = []
    step = 0
    while not (done or truncated):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        if render:
            env.render()
        actions.append((step, int(action)))
        rewards.append((step, reward))
        total_reward += reward
        step += 1
    return total_reward, actions, rewards


def evaluate(model: PPO, env: gym.Env, seeds: List[int], render: bool = False):
    """Return episode scores and optionally count crashes."""

    scores = []
    actions = []
    rewards = []
    crashes = 0
    for seed in seeds:
        score, act, rew = run_episode(model, env, seed, render=render)
        scores.append(score)
        actions.append(act)
        rewards.append(rew)
        expected = 17 if env.spec.id == "merge-v0" else 30
        if len(act) != expected:
            crashes += 1
    return scores, actions, rewards, crashes


def compute_correlations(expert_data, learner_data, seeds: List[int]):
    expert_actions, expert_rewards = expert_data
    learner_actions, learner_rewards = learner_data
    action_corr = []
    reward_corr = []
    valid_seeds = []
    for idx, (ea, la) in enumerate(zip(expert_actions, learner_actions)):
        if len(ea) != len(la):
            print(f"seed {seeds[idx]}, len(e): {len(ea)}, len(l): {len(la)}")
            continue
        y1 = [v for _, v in ea]
        y2 = [v for _, v in la]
        if np.var(y1) == 0 or np.var(y2) == 0:
            continue
        action_corr.append(float(np.corrcoef(y1, y2)[0, 1]))
        valid_seeds.append(seeds[idx])
        y1 = [v for _, v in expert_rewards[idx]]
        y2 = [v for _, v in learner_rewards[idx]]
        reward_corr.append(float(np.corrcoef(y1, y2)[0, 1]))
    return action_corr, reward_corr, valid_seeds


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained policies")
    parser.add_argument(
        "agent",
        choices=["e", "l", "c"],
        help="Agent to evaluate: expert(e), learner(l) or compare(c)",
    )
    parser.add_argument(
        "env",
        choices=["r", "i", "h", "h2", "h3", "m"],
        help="Environment code: r=roundabout, i=intersection, h=highway-fast, h2=highway, h3=highway variant, m=merge",
    )
    parser.add_argument(
        "--num-seeds",
        type=int,
        default=10,
        help="Number of random seeds to evaluate",
    )
    parser.add_argument(
        "--expert-path",
        default=DEFAULT_EXPERT_PATH,
        help="Path to the expert model",
    )
    parser.add_argument(
        "--learner-path",
        default=DEFAULT_LEARNER_PATH,
        help="Path to the learner model",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment during evaluation",
    )
    args = parser.parse_args()

    seeds = generate_seeds(args.num_seeds)
    env = make_env(args.env)
    expert = PPO.load(args.expert_path)
    learner = PPO.load(args.learner_path)

    if args.agent == "e":
        scores, _, _, _ = evaluate(expert, env, seeds, render=args.render)
        print(f"Expert mean score: {np.mean(scores):.2f}")
        print(f"Expert median score: {np.median(scores):.2f}")
    elif args.agent == "l":
        scores, _, _, _ = evaluate(learner, env, seeds, render=args.render)
        print(f"Learner mean score: {np.mean(scores):.2f}")
        print(f"Learner median score: {np.median(scores):.2f}")
    else:
        exp_scores, exp_actions, exp_rewards, _ = evaluate(
            expert, env, seeds, render=args.render
        )
        ler_scores, ler_actions, ler_rewards, _ = evaluate(
            learner, env, seeds, render=args.render
        )
        action_corr, reward_corr, valid_seeds = compute_correlations(
            (exp_actions, exp_rewards), (ler_actions, ler_rewards), seeds
        )
        print(f"Expert mean score: {np.mean(exp_scores):.2f}")
        print(f"Learner mean score: {np.mean(ler_scores):.2f}")
        if action_corr:
            print(f"Action correlation median: {np.median(action_corr):.2f}")
            print(f"Reward correlation median: {np.median(reward_corr):.2f}")
    print(f"seeds: {seeds}")


if __name__ == "__main__":
    main()

