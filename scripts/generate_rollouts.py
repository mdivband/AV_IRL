import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import argparse
import logging
import pickle
import numpy as np
from stable_baselines3 import PPO
from imitation.data.rollout import rollout, make_min_episodes
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env
from av_irl import SafeDistanceRewardWrapper, TimePenaltyWrapper
from av_irl import DrivingStyleRewardWrapper
from highway_env.envs.merge_env import MergeEnv

def _silent_is_terminated(self) -> bool:
    return self.vehicle.crashed or bool(self.vehicle.position[0] > 370)
MergeEnv._is_terminated = _silent_is_terminated


# def wrap_env(env, _):
#     env = SafeDistanceRewardWrapper(env)
#     return env


def generate_rollouts(model_path: str, env_name: str, output: str, episodes: int, seed: int, a: float, b: float) -> None:
    logging.info(
        "Loading expert model from %s and generating %d episodes in %s",
        model_path,
        episodes,
        env_name,
    )
    
    def wrap_env(e, _):
        return DrivingStyleRewardWrapper(e, a, b)

    rng = np.random.default_rng(seed)
    venv = make_vec_env(
        env_name,
        n_envs=8,
        parallel=True,
        rng=rng,
        env_make_kwargs={'config': {'ego_spacing': 3.0, 'simulation_frequency': 7, 'policy_frequency': 2, 'duration': 15}},
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
    parser.add_argument("--a", type=float, default=0.5, help="Coefficient for speed term")
    parser.add_argument("--b", type=float, default=0.5, help="Coefficient for distance penalty")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    generate_rollouts(args.model_path, args.env_name, args.output, args.episodes, args.seed, args.a, args.b)


if __name__ == "__main__":
    main()
