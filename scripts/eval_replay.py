"""How well does a lang_* checkpoint imitate the dataset actions? Open-loop replay on UR5 ManiUniCon.

For a few episodes per task, feed the recorded observations (+ the episode's instruction) to `select_action`
frame by frame, exactly as at deployment (ACT: temporal ensembling, DP: execute n_action_steps of each chunk),
and compare the executed action with the recorded action. A "hold" baseline (action = current joint state)
shows how much of the motion the policy actually explains.

Note: this is open-loop on dataset observations, not a closed-loop rollout on the robot.

Usage:
    python scripts/eval_replay.py <checkpoint>/pretrained_model [--episodes-per-task 2] [--seed 0]
"""

import argparse
import math
import random

import torch

import lerobot_policy_lang  # noqa: F401  (registers the policy types)
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

OBS_KEYS = ["observation.state", "observation.images.camera_0", "observation.images.camera_1"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-root", default="data/ur5_maniunicon")
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint)
    cfg = policy.config
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
    dataset = LeRobotDataset("local/ur5_maniunicon", root=args.dataset_root)
    episodes = dataset.meta.episodes

    episodes_by_task: dict[str, list[int]] = {}
    for ep in range(dataset.meta.total_episodes):
        episodes_by_task.setdefault(episodes[ep]["tasks"][0], []).append(ep)

    rng = random.Random(args.seed)
    deg = 180 / math.pi
    print(f"{args.checkpoint}  ({cfg.type}, {cfg.language_model_name})")
    print(f"  {'task':55s} {'joint L1 (deg)':>14s} {'p95':>6s} {'hold':>6s} {'grip acc':>8s}")
    totals = {"err": [], "hold": [], "grip": []}
    for task, eps in sorted(episodes_by_task.items()):
        errs, holds, grips = [], [], []
        for ep in rng.sample(eps, min(args.episodes_per_task, len(eps))):
            policy.reset()
            start, end = int(episodes[ep]["dataset_from_index"]), int(episodes[ep]["dataset_to_index"])
            for idx in range(start, end):
                item = dataset[idx]
                obs = {key: item[key] for key in OBS_KEYS} | {"task": task}
                with torch.inference_mode():
                    pred = postprocessor(policy.select_action(preprocessor(obs)))[0].cpu()
                gt, state = item["action"], item["observation.state"]
                errs.append((pred[:6] - gt[:6]).abs().mean() * deg)
                holds.append((state[:6] - gt[:6]).abs().mean() * deg)
                grips.append(float((pred[6] > 0.5) == (gt[6] > 0.5)))
        errs, holds, grips = torch.stack(errs), torch.stack(holds), torch.tensor(grips)
        totals["err"].append(errs), totals["hold"].append(holds), totals["grip"].append(grips)
        print(
            f"  {task[:55]:55s} {errs.mean():14.2f} {errs.quantile(0.95):6.2f} {holds.mean():6.2f} "
            f"{grips.mean() * 100:7.1f}%"
        )
    errs, holds, grips = (torch.cat(v) for v in totals.values())
    print(
        f"  {'ALL':55s} {errs.mean():14.2f} {errs.quantile(0.95):6.2f} {holds.mean():6.2f} "
        f"{grips.mean() * 100:7.1f}%"
    )


if __name__ == "__main__":
    main()
