#!/usr/bin/env bash
#
# Run full training pipeline with multiple experts and learners.
# Launch with:  nohup bash scripts/run_pipeline.sh > pipeline.log 2>&1 &
#

set -euo pipefail

mkdir -p model rollout logs

# Expert coefficient pairs (a,b)
AS=(0.2 0.5 0.8)
BS=(0.8 0.5 0.2)

ROLLOUT_SIZES=(100 500 2000 8000 32000 64000)
LEARNER_TS=(100000 200000)
ENVS=("highway-fast-v0" "merge-v0")
ENVS_L=("highway-fast-v0")

log() { echo "$(date '+%F %T') $*"; }

#######################################
# 1. Train experts (unless model exists)
#######################################
for i in "${!AS[@]}"; do
  a=${AS[$i]}
  b=${BS[$i]}
  expert_zip="model/expert_a${a}_b${b}.zip"

  if [[ -f "$expert_zip" ]]; then
    log "Expert model $expert_zip already exists – skipping training."
  else
    log "Training expert a=$a b=$b"
    python scripts/expert_ppo_training.py --ts 2000000 --a "$a" --b "$b" \
      > "logs/expert_a${a}_b${b}.log" 2>&1
    mv model/expert_ppo_mlt_h1_m_h2.zip "$expert_zip"
  fi

  #######################################
  # 2. Generate rollouts (unless file exists)
  #######################################
  for env in "${ENVS[@]}"; do
    short=$([[ "$env" == "highway-fast-v0" ]] && echo "hf" || echo "mg")
    for size in "${ROLLOUT_SIZES[@]}"; do
      roll_pkl="rollout/${short}_a${a}_b${b}_${size}.pkl"

      if [[ -f "$roll_pkl" ]]; then
        log "Rollout $roll_pkl already exists – skipping generation."
        continue
      fi

      log "Generating rollouts → $roll_pkl"
      python scripts/generate_rollouts.py \
        --model-path "$expert_zip" \
        --env-name "$env" \
        --episodes "$size" \
        --output "$roll_pkl" \
        --a "$a" --b "$b" \
        > "logs/rollout_${short}_a${a}_b${b}_${size}.log" 2>&1
    done
  done
done

#######################################
# 3. Train learners (unless model exists)
#######################################
for i in "${!AS[@]}"; do
  a=${AS[$i]}
  b=${BS[$i]}

  for env in "${ENVS_L[@]}"; do
    short=$([[ "$env" == "highway-fast-v0" ]] && echo "hf" || echo "mg")

    for size in "${ROLLOUT_SIZES[@]}"; do
      roll_pkl="rollout/${short}_a${a}_b${b}_${size}.pkl"

      # Skip if rollout missing
      if [[ ! -f "$roll_pkl" ]]; then
        log "Required rollout $roll_pkl not found – skipping learners for this setup."
        continue
      fi

      for alg in airl gail airl_slot; do
        for ts in "${LEARNER_TS[@]}"; do
          learner_zip="model/${alg}_a${a}_b${b}_${size}_ts${ts}.zip"

          if [[ -f "$learner_zip" ]]; then
            log "Learner model $learner_zip already exists – skipping training."
            continue
          fi

          log "Training $alg  env=$env  rollouts=$size  ts=$ts"
          case "$alg" in
            airl)
              python scripts/learner_airl_train_loop.py \
                --env "$env" --a "$a" --b "$b" --size "$size" \
                --out "$learner_zip" --ts "$ts" \
                > "logs/${alg}_a${a}_b${b}_${size}_ts${ts}.log" 2>&1
              ;;
            gail)
              python scripts/learner_gail_train_loop.py \
                --env "$env" --a "$a" --b "$b" --size "$size" \
                --out "$learner_zip" --ts "$ts" \
                > "logs/${alg}_a${a}_b${b}_${size}_ts${ts}.log" 2>&1
              ;;
            airl_slot)
              python scripts/learner_airl_slot_train.py \
                --env "$env" --a "$a" --b "$b" --size "$size" \
                --out "$learner_zip" --ts "$ts" \
                > "logs/${alg}_a${a}_b${b}_${size}_ts${ts}.log" 2>&1
              ;;
          esac
        done
      done
    done
  done
done


log "Pipeline complete"

