conda create -n av_irl python=3.10
conda activate av_irl
pip install -r requirements.txt
pip install -e .
```

Alternatively, a standard virtual environment works as well:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Installing in editable mode ensures the `av_irl` package is importable when running scripts.

## Usage

Run one of the training loops using the provided CLI options:

```bash
python scripts/airl_expert_ppo_training.py --timesteps 100000
```

The evaluation script prints the mean episode score and mean safe distance penalty for an expert or trained learner. Model paths can be overridden with command line options:

```bash
python scripts/final_eval.py e h --num-seeds 10 --expert-path model/expert.zip --learner-path model/learner.zip
```

To reproduce the full workflow on a workstation you can run the convenience script:

```bash
nohup bash scripts/run_pipeline.sh &
```

This trains experts for three `(a, b)` settings, generates rollouts in two environments and trains AIRL/GAIL learners using different rollout sizes.

To compare AIRL reward estimates with different coefficients run:

```bash
python scripts/reward_comparison.py
```
The script trains two AIRL models using coefficients `a` and `b`.
`RewardScaleWrapper` scales the environment reward by `a` while
`SafeDistanceRewardWrapper` subtracts `b` times the safe-distance penalty.
`TimePenaltyWrapper` always applies a small per-step penalty.

### Coefficients `a` and `b`

`a` controls how strongly the base environment reward is scaled. A value
greater than one amplifies the original reward signal, while a value between
zero and one dampens it.  `b` determines the weight of the safe-distance
penalty applied by `SafeDistanceRewardWrapper`.

### Why compare them for AIRL?

AIRL attempts to recover a reward function that explains the expert
demonstrations.  By training two AIRL models under different `(a, b)` settings
and comparing their predicted rewards, you can check whether AIRL is sensitive
to reward scaling versus safety penalties.  If the estimated reward functions
change noticeably when `a` and `b` are swapped, AIRL has not fully recovered a
reward that is invariant to reward shaping.

