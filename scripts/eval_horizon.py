"""Action-chunk prediction error vs. horizon on UR5 ManiUniCon, against a "hold" baseline.

Samples frames from every task, predicts one action chunk with the correct instruction and reports joint L1
to the recorded actions at several horizons k (steps ahead, 30 fps), next to the error of simply holding the
current joint state. Per-step replay errors are tiny for any policy (the robot moves <1 deg per frame), so the
long-horizon numbers are what show whether the policy predicts the demonstrated motion.

Usage:
    python scripts/eval_horizon.py <checkpoint>/pretrained_model [--per-task 16] [--seed 0]
"""

import argparse
import math
import random

import torch

import lerobot_policy_lang  # noqa: F401  (registers the policy types)
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

IMAGE_STATE_KEYS = ["observation.state", "observation.images.camera_0", "observation.images.camera_1"]
HORIZONS = [1, 8, 16, 32, 64, 100]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-root", default="data/ur5_maniunicon")
    parser.add_argument("--per-task", type=int, default=16, help="frames sampled per task")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint)
    cfg = policy.config
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
    fps = 30
    is_diffusion = cfg.type == "lang_diffusion"
    offset = cfg.n_obs_steps - 1 if is_diffusion else 0  # DP's chunk starts at the current observation

    delta_timestamps = {"action": [i / fps for i in cfg.action_delta_indices]}
    if is_diffusion:
        obs_ts = [i / fps for i in cfg.observation_delta_indices]
        delta_timestamps.update({key: obs_ts for key in IMAGE_STATE_KEYS})
    dataset = LeRobotDataset("local/ur5_maniunicon", root=args.dataset_root, delta_timestamps=delta_timestamps)
    episodes = dataset.meta.episodes

    episodes_by_task: dict[str, list[int]] = {}
    for ep in range(dataset.meta.total_episodes):
        episodes_by_task.setdefault(episodes[ep]["tasks"][0], []).append(ep)

    rng = random.Random(args.seed)
    pred_err, hold_err, valid_all = [], [], []
    for task, eps in sorted(episodes_by_task.items()):
        items = []
        for _ in range(args.per_task):
            ep = rng.choice(eps)
            start, end = int(episodes[ep]["dataset_from_index"]), int(episodes[ep]["dataset_to_index"])
            items.append(dataset[rng.randrange(start, end)])
        obs = {key: torch.stack([it[key] for it in items]) for key in IMAGE_STATE_KEYS}
        gt = torch.stack([it["action"] for it in items])
        valid = ~torch.stack([it["action_is_pad"] for it in items])
        policy.reset()
        kwargs = {}
        if is_diffusion:
            generator = torch.Generator(device=cfg.device).manual_seed(args.seed)
            kwargs["noise"] = torch.randn(len(items), cfg.horizon, gt.shape[-1], device=cfg.device, generator=generator)
        with torch.inference_mode():
            pred = postprocessor(policy.predict_action_chunk(preprocessor({**obs, "task": [task] * len(items)}), **kwargs))
        n = pred.shape[1]
        target, mask = gt[:, offset : offset + n], valid[:, offset : offset + n]
        state = obs["observation.state"][:, -1] if is_diffusion else obs["observation.state"]
        pred_err.append((pred.cpu() - target).abs()[..., :6].mean(-1))
        hold_err.append((state[:, None] - target).abs()[..., :6].mean(-1))
        valid_all.append(mask)
    pred_err, hold_err, valid_all = torch.cat(pred_err), torch.cat(hold_err), torch.cat(valid_all)

    deg = 180 / math.pi
    print(f"{args.checkpoint}  ({cfg.type}, {cfg.language_model_name})")
    print(f"  {'k (steps ahead)':>16s} {'policy (deg)':>12s} {'hold (deg)':>10s}")
    for k in HORIZONS:
        if k > pred_err.shape[1]:
            break
        m = valid_all[:, k - 1]
        print(f"  {k:16d} {pred_err[m, k - 1].mean() * deg:12.2f} {hold_err[m, k - 1].mean() * deg:10.2f}")


if __name__ == "__main__":
    main()
