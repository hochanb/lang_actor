#!/usr/bin/env bash
# Language-conditioned ACT on the merged UR5 ManiUniCon dataset (12 tasks).
# Chunk 100 for training; at execution the policy re-plans every step and temporally ensembles the
# overlapping chunks (temporal_ensemble_coeff=0.01, n_action_steps=1, as in the ACT paper).
# Keeps only the newest $KEEP_LAST_CHECKPOINTS (default 10) checkpoints.
# Extra lerobot-train args are forwarded, e.g.: scripts/train_lang_act.sh --steps=1000 --output_dir=outputs/test
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train.py \
  --policy.type=lang_act \
  --dataset.repo_id=local/ur5_maniunicon \
  --dataset.root=data/ur5_maniunicon \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.chunk_size=100 \
  --policy.n_action_steps=1 \
  --policy.temporal_ensemble_coeff=0.01 \
  --batch_size=16 \
  --num_workers=12 \
  --steps=100000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --job_name=lang_act_ur5 \
  --output_dir=outputs/train/lang_act_ur5 \
  "$@"
