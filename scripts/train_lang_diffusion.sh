#!/usr/bin/env bash
# Language-conditioned Diffusion Policy on the merged UR5 ManiUniCon dataset (12 tasks).
# Predicts a 32-step action chunk (horizon), executes the first 16, samples with DDIM in 10 steps.
# 480x640 frames are resized to 240x320 then randomly cropped to 216x288 (crop_ratio=0.9).
# drop_n_last_frames = horizon - n_action_steps - n_obs_steps + 1 = 32 - 16 - 2 + 1 = 15.
# Keeps only the newest $KEEP_LAST_CHECKPOINTS (default 10) checkpoints.
# Extra lerobot-train args are forwarded, e.g.: scripts/train_lang_diffusion.sh --steps=1000 --output_dir=outputs/test
set -euo pipefail
cd "$(dirname "$0")/.."
python scripts/train.py \
  --policy.type=lang_diffusion \
  --dataset.repo_id=local/ur5_maniunicon \
  --dataset.root=data/ur5_maniunicon \
  --policy.push_to_hub=false \
  --policy.device=cuda \
  --policy.resize_shape='[240, 320]' \
  --policy.crop_ratio=0.9 \
  --policy.horizon=32 \
  --policy.n_action_steps=16 \
  --policy.drop_n_last_frames=15 \
  --policy.noise_scheduler_type=DDIM \
  --policy.num_inference_steps=10 \
  --batch_size=64 \
  --num_workers=12 \
  --steps=100000 \
  --save_freq=5000 \
  --log_freq=100 \
  --wandb.enable=false \
  --job_name=lang_diffusion_ur5 \
  --output_dir=outputs/train/lang_diffusion_ur5 \
  "$@"
