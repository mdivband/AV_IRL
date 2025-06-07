 # AV_IRL

This project contains utilities for training and evaluating autonomous vehicle agents using inverse reinforcement learning (IRL). It demonstrates adversarial IRL techniques based on **Generative Adversarial Imitation Learning (GAIL)** and **Adversarial Inverse Reinforcement Learning (AIRL)**. Training scripts rely on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [imitation](https://github.com/HumanCompatibleAI/imitation), and `gymnasium` environments representing highway-style driving tasks.

## Project Layout

```
scripts/                Training and evaluation scripts
    airl_expert_ppo_training.py
    airl_train_loop.py
    gail_train_loop.py
    final_eval.py
requirements.txt        Python dependencies
LICENSE                 Project license (MIT)
```

## Setup

We recommend using Miniconda to manage the Python environment:

```bash
conda create -n av_irl python=3.10
conda activate av_irl
pip install -r requirements.txt
```

Alternatively, a standard virtual environment works as well:

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
