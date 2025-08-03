import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")

import argparse, logging, pickle, numpy as np
from stable_baselines3 import PPO
from imitation.data.rollout import rollout, make_min_episodes
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.util.util import make_vec_env

from av_irl import DrivingStyleRewardWrapper
from highway_env.envs.merge_env import MergeEnv

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


def generate_rollouts(
    model_path: str,
    env_name: str,
    output: str,
    episodes: int,
    seed: int,
    a: float,
    b: float,
) -> None:
    logging.info(
        "Loading expert model from %s and generating %d episodes in %s",
        model_path,
        episodes,
        env_name,
    )

    def wrap_env(e, _):
        return DrivingStyleRewardWrapper(e, a=a, b=b)

    rng = np.random.default_rng(seed)

    venv = make_vec_env(
        env_name,
        n_envs=12,
        parallel=True,
        rng=rng,
        env_make_kwargs={"config": KINEMATICS_REL_CFG},
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
    parser.add_argument("--env-name", default="highway-fast-v0", help="Gymnasium env ID")
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes")
    parser.add_argument("--output", required=True, help="Pickle file to save rollouts")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--a", type=float, default=0.5, help="Speed-reward weight")
    parser.add_argument("--b", type=float, default=0.5, help="Distance-reward weight")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    generate_rollouts(
        model_path=args.model_path,
        env_name=args.env_name,
        output=args.output,
        episodes=args.episodes,
        seed=args.seed,
        a=args.a,
        b=args.b,
    )


if __name__ == "__main__":
    main()
