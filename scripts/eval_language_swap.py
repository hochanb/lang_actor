"""Does a lang_* checkpoint actually use the instruction? Offline swap test on the UR5 ManiUniCon dataset.

For frames sampled from each group of language-ambiguous tasks, predict an action chunk with the correct
instruction and with a swapped one (its counterpart in the pair, or another dice order), and compare the
L1 error to the ground-truth chunk. ratio ~1.0x means the instruction is ignored for that group.

Usage:
    python scripts/eval_language_swap.py <checkpoint>/pretrained_model [--per-task 12] [--seed 0]
"""

import argparse
import random

import torch

import lerobot_policy_lang  # noqa: F401  (registers the policy types)
from lerobot.configs import PreTrainedConfig
from lerobot.datasets import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

IMAGE_STATE_KEYS = ["observation.state", "observation.images.camera_0", "observation.images.camera_1"]
SWAP_PAIRS = [
    ("move the red cup to the blue one", "move the blue cup to the red one"),
    ("move the cup and the bowl to the plate", "move the cup and the plate to the bowl"),
    ("put the red block on the blue plate", "put the blue block on the red plate"),
]
GROUP_NAMES = ["cup color", "cup/bowl/plate", "block color"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--dataset-root", default="data/ur5_maniunicon")
    parser.add_argument("--per-task", type=int, default=12, help="frames sampled per task")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = PreTrainedConfig.from_pretrained(args.checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(args.checkpoint)
    cfg = policy.config
    preprocessor, postprocessor = make_pre_post_processors(cfg, pretrained_path=args.checkpoint)
    fps = 30
    is_diffusion = cfg.type == "lang_diffusion"

    delta_timestamps = {"action": [i / fps for i in cfg.action_delta_indices]}
    if is_diffusion:
        obs_ts = [i / fps for i in cfg.observation_delta_indices]
        delta_timestamps.update({key: obs_ts for key in IMAGE_STATE_KEYS})
    dataset = LeRobotDataset("local/ur5_maniunicon", root=args.dataset_root, delta_timestamps=delta_timestamps)

    tasks = list(dataset.meta.tasks.index)
    dice = [t for t in tasks if t.startswith("arrange the dice")]
    swap = {a: b for a, b in SWAP_PAIRS} | {b: a for a, b in SWAP_PAIRS}
    groups = dict(zip(GROUP_NAMES, [list(p) for p in SWAP_PAIRS], strict=True)) | {"dice order (6-way)": dice}

    episodes = dataset.meta.episodes
    episodes_by_task: dict[str, list[int]] = {}
    for ep in range(dataset.meta.total_episodes):
        episodes_by_task.setdefault(episodes[ep]["tasks"][0], []).append(ep)

    rng = random.Random(args.seed)
    print(f"{args.checkpoint}  ({cfg.type}, {cfg.language_model_name})")
    for group, group_tasks in groups.items():
        items = []
        for task in group_tasks:
            for _ in range(args.per_task):
                ep = rng.choice(episodes_by_task[task])
                start, end = int(episodes[ep]["dataset_from_index"]), int(episodes[ep]["dataset_to_index"])
                items.append(dataset[rng.randrange(start, end)])
        obs = {key: torch.stack([it[key] for it in items]) for key in IMAGE_STATE_KEYS}
        gt = torch.stack([it["action"] for it in items])
        valid = ~torch.stack([it["action_is_pad"] for it in items])
        correct = [it["task"] for it in items]
        swapped = [swap.get(t) or rng.choice([d for d in dice if d != t]) for t in correct]

        errors = {}
        for label, instr in [("correct", correct), ("swapped", swapped)]:
            policy.reset()
            kwargs = {}
            if is_diffusion:
                generator = torch.Generator(device=cfg.device).manual_seed(args.seed)
                kwargs["noise"] = torch.randn(
                    len(items), cfg.horizon, gt.shape[-1], device=cfg.device, generator=generator
                )
            pred = postprocessor(policy.predict_action_chunk(preprocessor({**obs, "task": instr}), **kwargs))
            # DP's chunk starts at the current observation (index n_obs_steps - 1 of the action window).
            offset = cfg.n_obs_steps - 1 if is_diffusion else 0
            target = gt[:, offset : offset + pred.shape[1]]
            mask = valid[:, offset : offset + pred.shape[1]]
            err = (pred.cpu() - target).abs()[..., :6].mean(-1)  # joints only
            errors[label] = ((err * mask).sum() / mask.sum()).item()
        ratio = errors["swapped"] / errors["correct"]
        print(
            f"  {group:20s} chunk joint L1 (rad): correct {errors['correct']:.4f}  "
            f"swapped {errors['swapped']:.4f}  ratio {ratio:.2f}x"
        )


if __name__ == "__main__":
    main()
