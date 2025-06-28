import argparse
import logging
import pickle
import numpy as np
from stable_baselines3 import PPO
from imitation.data.rollout import rollout, make_min_episodes
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from av_irl import SafeDistanceRewardWrapper, TimePenaltyWrapper


def wrap_env(env, _):
    env = SafeDistanceRewardWrapper(env)
    env = TimePenaltyWrapper(env)
    return env


def generate_rollouts(model_path: str, env_name: str, output: str, episodes: int, seed: int) -> None:
    logging.info(
        "Loading expert model from %s and generating %d episodes in %s",
        model_path,
        episodes,
        env_name,
    )
    rng = np.random.default_rng(seed)
    venv = make_vec_env(
        env_name,
        n_envs=1,
        rng=rng,
        env_make_kwargs={"config": {"ego_spacing": 3.0}},
        post_wrappers=[wrap_env, lambda e, _: RolloutInfoWrapper(e)],
    )
    model = PPO.load(model_path, env=venv)
    trajectories = rollout(model, venv, make_min_episodes(episodes), rng=rng)
    logging.info("Collected %d trajectories", len(trajectories))
    with open(output, "wb") as f:
        pickle.dump(trajectories, f)
    logging.info("Saved rollouts to %s", output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expert rollouts")
    parser.add_argument("--model-path", required=True, help="Path to PPO expert model")
    parser.add_argument("--env-name", default="highway-fast-v0", help="Gymnasium environment name")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to record")
    parser.add_argument("--output", required=True, help="File to save rollouts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    generate_rollouts(args.model_path, args.env_name, args.output, args.episodes, args.seed)


if __name__ == "__main__":
    main()
