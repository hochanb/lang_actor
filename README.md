# lang_actor

## Environment

- `lerobot/`: clone of huggingface/lerobot (main @ `ca69a206`, v0.6.2), installed editable (not tracked here)
- conda env: `lang_actor` (Python 3.12, torch 2.11.0+cu128)
  - extras: `training, diffusion, pusht, aloha, test, transformers-dep`

```bash
conda create -n lang_actor python=3.12 && conda activate lang_actor
git clone https://github.com/huggingface/lerobot.git && git -C lerobot checkout ca69a206
pip install -e "lerobot[training,diffusion,pusht,aloha,test,transformers-dep]"
pip install -e . --no-deps    # lerobot_policy_lang plugin
```

## Training (ACT / Diffusion Policy)

```bash
# ACT on ALOHA sim
lerobot-train --policy.type=act --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --env.type=aloha --env.task=AlohaTransferCube-v0 \
  --policy.push_to_hub=false --wandb.enable=false --output_dir=outputs/train/act_aloha

# Diffusion Policy on PushT
lerobot-train --policy.type=diffusion --dataset.repo_id=lerobot/pusht \
  --env.type=pusht \
  --policy.push_to_hub=false --wandb.enable=false --output_dir=outputs/train/diffusion_pusht
```

Common options: `--steps`, `--batch_size`, `--save_freq`, `--eval_freq`, `--num_workers`, `--policy.device=cuda`.

## Evaluation

```bash
MUJOCO_GL=egl lerobot-eval --policy.path=outputs/train/diffusion_pusht/checkpoints/last/pretrained_model \
  --env.type=pusht --eval.n_episodes=50 --eval.batch_size=10
```

## Python API

```python
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.policies.diffusion import DiffusionConfig, DiffusionPolicy
from lerobot.policies import make_pre_post_processors
```

For a custom training loop example, see `lerobot/examples/training/train_policy.py`.

## Language-conditioned ACT / DP (`lerobot_policy_lang/`)

LeRobot policy plugin (auto-registered via the `lerobot_policy_` package prefix; installed editable).
Language encoder: `google-bert/bert-base-uncased` (110M), always frozen and not part of the policy's parameters:
it is never optimized nor saved in checkpoints, and is loaded from the HF cache by name. The dataset's `task`
string is tokenized by the preprocessor and pooled (masked mean or CLS) into one 768-d embedding.

| policy type      | how language conditions the policy |
|------------------|------------------------------------|
| `lang_act`       | projected to `dim_model`, added as one extra transformer-encoder token `[latent, lang, state, img...]` |
| `lang_diffusion` | projected to `language_embed_dim` (128), concatenated to the U-Net FiLM global condition |

Language options (both policies): `--policy.language_model_name`, `--policy.language_pooling={mean,cls}`,
`--policy.tokenizer_max_length`.

```bash
lerobot-train --policy.type=lang_act --dataset.repo_id=<repo> --policy.push_to_hub=false ...
lerobot-train --policy.type=lang_diffusion --dataset.repo_id=<repo> --policy.push_to_hub=false ...
```

Tests: `python -m pytest tests -q`

## UR5 ManiUniCon dataset (`Ngseo/ur5-maniunicon`, gated)

12 tasks / 420 episodes / 149,907 frames @ 30 fps, 7-dim joint state & action, 2 cams (wrist `camera_0`,
scene `camera_1`, 480x640 AV1). The Hub repo stores each task as its own sub-dataset; merge them once:

```bash
hf auth login                                  # needs access to the gated repo
python scripts/prepare_ur5_maniunicon.py       # -> data/ur5_maniunicon (repo_id local/ur5_maniunicon)
```

Train (extra args are forwarded to `lerobot-train`; `scripts/train.py` wraps it and keeps only the newest
`KEEP_LAST_CHECKPOINTS`=10 step checkpoints):

```bash
tmux new -s lang_act "CUDA_VISIBLE_DEVICES=0 scripts/train_lang_act.sh 2>&1 | tee outputs/logs/lang_act_ur5.log"
tmux new -s lang_dp  "CUDA_VISIBLE_DEVICES=1 scripts/train_lang_diffusion.sh 2>&1 | tee outputs/logs/lang_diffusion_ur5.log"
```

| | lang_act | lang_diffusion |
|---|---|---|
| chunk | 100 (train), temporal ensembling at execution (coeff 0.01, re-plan every step) | horizon 32, execute 16 |
| sampling | - | DDIM, 10 steps |
| images | 480x640 | resize 240x320, random crop 216x288 |
| batch / steps / save | 16 / 100k / every 5k | 64 / 100k / every 5k |
| train speed (RTX 5000 Ada) | ~0.2 s/step (~5.5 h) | ~0.34 s/step (~9.5 h) |
| inference | ~10 ms every step | ~60 ms per new chunk (every 16 steps), queue pop otherwise |
| checkpoint (incl. optimizer) | ~0.6 GB | ~3.2 GB |

## Evaluation on the dataset (offline)

```bash
python scripts/eval_language_swap.py <ckpt>/pretrained_model   # correct vs. swapped instruction (ratio ~1.0x = ignored)
python scripts/eval_horizon.py <ckpt>/pretrained_model         # chunk error vs. steps ahead, against holding the state
python scripts/eval_replay.py <ckpt>/pretrained_model          # frame-by-frame replay with select_action (~30-40 min)
```

Final checkpoints (100k steps), joint L1 in degrees on training episodes:

| | encoder | 16 steps ahead (hold 3.75) | 100 steps ahead (hold 12.81) | replay (hold 0.97) | gripper acc | swap: cup/bowl | swap: cup color / block color / dice |
|---|---|---|---|---|---|---|---|
| lang_act | BERT | 0.73 | 1.15 | 0.59 | 99.6% | 10.63x | 1.00x / 1.00x / 1.00x |
| lang_act | SBERT | 0.76 | 1.01 | 0.59 | 99.3% | 14.94x | 1.00x / 1.00x / 1.00x |
| lang_diffusion | BERT | 0.43 | - | 0.41 | 97.9% | 3.08x | 1.01x / 1.01x / 1.03x |
| lang_diffusion | SBERT | 0.47 | - | 0.42 | 98.2% | 2.00x | 1.17x / 1.05x / 0.98x |

Both encoders follow the demonstrations but ignore word-order swaps (red<->blue, dice order); this is the
intended baseline behavior.

## Checkpoints (private, HF Hub)

`hochanb/lang_act_bert_ur5`, `hochanb/lang_act_sbert_ur5`, `hochanb/lang_diffusion_bert_ur5`,
`hochanb/lang_diffusion_sbert_ur5` (policy weights + processors, no optimizer state, no BERT weights).

```python
import lerobot_policy_lang
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
policy = get_policy_class("lang_act").from_pretrained("hochanb/lang_act_bert_ur5")
pre, post = make_pre_post_processors(policy.config, pretrained_path="hochanb/lang_act_bert_ur5")
```
