# Predicting Autonomous Vehicles Behaviour in Unknown Environments Using IRL and Attention

This project contains utilities for training and evaluating autonomous vehicle agents using inverse reinforcement learning (IRL) and attention. It demonstrates adversarial IRL techniques based on **Generative Adversarial Imitation Learning (GAIL)** and **Adversarial Inverse Reinforcement Learning (AIRL)**. Training scripts rely on [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [imitation](https://github.com/HumanCompatibleAI/imitation), and `gymnasium` environments representing highway-style driving tasks.

We further explore **structure-aware reward networks** by integrating **Slot Attention** and **Graph Attention Networks (GAT)** as neural encoders in both AIRL and GAIL. These help improve generalisation and interpretability in high-dimensional, object-centric driving scenarios.

All environments are wrapped with a **SafeDistanceRewardWrapper** which subtracts a continuous penalty for driving too close to nearby vehicles.

## Project Layout

```
scripts/                Training and evaluation scripts
    expert_ppo_training.py
    learner_airl_train.py              # AIRL + MLP
    learner_airl_slot_train.py         # AIRL + Slot Attention
    learner_airl_gat_train.py          # AIRL + GAT
    learner_gail_train.py              # GAIL + MLP
    learner_gail_slot_train.py         # GAIL + Slot Attention
    learner_gail_gat_train.py          # GAIL + GAT
av_irl/                 Custom wrappers and modules (reward wrappers etc)
    gat.py                  GATRewardNet module
    slot_attention.py       SlotAttention and SlotRewardNet module
requirements.txt        Python dependencies
LICENSE                 Project license (MIT)
```

## Setup

We recommend using Miniconda to manage the Python environment:

```bash
conda create -n av_irl python=3.10
conda activate av_irl
pip install -r requirements.txt
pip install -e .
```

Installing in editable mode ensures the `av_irl` package is importable when running scripts.

## Usage

Run one of the training loops using the provided CLI options:

```bash
python scripts/expert_ppo_training.py --timesteps 100000
```

Run AIRL or GAIL with custom reward encoders:

```bash
python scripts/learner_airl_slot_train.py    # AIRL + Slot Attention
python scripts/learner_gail_gat_train.py     # GAIL + GAT
```

To reproduce the full workflow (3 driving styles expert-agents training + AIRL/GAIL learner-agents training)  on a workstation:

```bash
nohup bash scripts/run_pipeline.sh &
```

This trains experts for three `(a, b)` settings, generates rollouts in two environments, and trains AIRL/GAIL learners using different rollout sizes.

### Coefficients `a` and `b`

`a` controls the weight of the speed reward and `b` determines the weight of the safe-distance applied by `SafeDistanceRewardWrapper`.

### Structure-Aware Reward Networks

To address the challenge of modelling high-dimensional, structured observations (such as ego + multiple surrounding vehicles), we provide two alternative encoders:

- **SlotRewardNet** uses **Slot Attention** to group and attend to object-like features (vehicles) without explicit structure.
- **GATRewardNet** uses **Graph Attention Network (GAT)** to model relational attention between vehicles using an implicit interaction graph.

Each can be used as a drop-in replacement in AIRL or GAIL by changing the training script. This allows comparing whether different reward network architectures generalise better across environments and styles.
