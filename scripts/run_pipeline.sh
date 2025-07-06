#!/bin/bash

# Run full training pipeline with multiple experts and learners.
# Recommended to launch via: nohup bash scripts/run_pipeline.sh > pipeline.log 2>&1 &

set -e

mkdir -p model rollout logs

# Expert coefficient pairs (a,b)
AS=(0.2 0.5 0.8)
BS=(0.8 0.5 0.2)

ROLLOUT_SIZES=(32000 64000 256000)
LEARNER_TS=(100000 200000)
ENVS=("highway-fast-v0" "merge-v0")

log() { echo "$(date '+%F %T') $*"; }

# Train experts
for i in ${!AS[@]}; do
  a=${AS[$i]}
  b=${BS[$i]}
  log "Training expert a=$a b=$b"
  python scripts/expert_ppo_training.py --ts 2000000 --a "$a" --b "$b" \
    > "logs/expert_a${a}_b${b}.log" 2>&1
  mv model/expert_ppo_mlt_h1_m_h2.zip "model/expert_a${a}_b${b}.zip"

  for env in "${ENVS[@]}"; do
    short=$( [[ "$env" == "highway-fast-v0" ]] && echo "hf" || echo "mg" )
    for size in "${ROLLOUT_SIZES[@]}"; do
      out="rollout/${short}_a${a}_b${b}_${size}.pkl"
      log "Generating rollouts $out"
      python scripts/generate_rollouts.py \
        --model-path "model/expert_a${a}_b${b}.zip" \
        --env-name "$env" \
        --episodes "$size" \
        --output "$out" \
        --a "$a" --b "$b" \
        > "logs/rollout_${short}_a${a}_b${b}_${size}.log" 2>&1
    done
  done

done

# Train learners
for i in ${!AS[@]}; do
  a=${AS[$i]}
  b=${BS[$i]}
  for env in "${ENVS[@]}"; do
    short=$( [[ "$env" == "highway-fast-v0" ]] && echo "hf" || echo "mg" )
    for size in "${ROLLOUT_SIZES[@]}"; do
      roll="rollout/${short}_a${a}_b${b}_${size}.pkl"
      for alg in airl gail; do
        for ts in "${LEARNER_TS[@]}"; do
          out="model/${alg}_a${a}_b${b}_${short}_${size}_ts${ts}.zip"
          log "Training $alg env=$env rollouts=$size ts=$ts"
          if [[ "$alg" == "airl" ]]; then
            python scripts/learner_airl_train_loop.py \
              --env "$env" --rollout "$roll" --out "$out" --ts "$ts" \
              > "logs/${alg}_a${a}_b${b}_${short}_${size}_ts${ts}.log" 2>&1
          else
            python scripts/learner_gail_train_loop.py \
              --env "$env" --rollout "$roll" --out "$out" --ts "$ts" \
              > "logs/${alg}_a${a}_b${b}_${short}_${size}_ts${ts}.log" 2>&1
          fi
        done
      done
    done
  done

done

log "Pipeline complete"
