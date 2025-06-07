# AV_IRL

Code for training and evaluating autonomous vehicle agents using inverse reinforcement learning.

## Project Layout

```
scripts/                Example training and evaluation scripts
    airl_expert_ppo_training.py
    airl_train_loop.py
    gail_train_loop.py
    final_eval.py
requirements.txt        Python dependencies
LICENSE                 Project license (MIT)
```

The scripts rely on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3),
[imitation](https://github.com/HumanCompatibleAI/imitation) and `gymnasium` environments.
Each script can be executed directly once the dependencies are installed.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

Run one of the training loops using the provided CLI options:

```bash
python scripts/airl_expert_ppo_training.py --timesteps 100000
```

The evaluation script allows comparison between an expert and a trained learner:

```bash
python scripts/final_eval.py e h --num-seeds 10
```

Refer to the source of each script for configurable parameters.