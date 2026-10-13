"""Download Ngseo/ur5-maniunicon and merge its 12 per-task sub-datasets into one multi-task LeRobotDataset.

Each task folder in the Hub repo is a standalone LeRobot v3.0 dataset with a single task string. Merging them
gives one dataset whose `task` column holds the 12 instructions, which is what the lang_* policies condition on.

Usage:
    python scripts/prepare_ur5_maniunicon.py [--output-root data/ur5_maniunicon] [--tasks dice_bbr dice_brb ...]
Then train with:
    lerobot-train --dataset.repo_id=local/ur5_maniunicon --dataset.root=data/ur5_maniunicon ...
"""

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

from lerobot.datasets.aggregate import aggregate_datasets

HUB_REPO_ID = "Ngseo/ur5-maniunicon"
ALL_TASKS = [
    "red2blue_cup",
    "blue2red_cup",
    "cup_bowl_2_plate",
    "cup_plate_2_bowl",
    "red_block_to_blue_plate",
    "blue_block_to_red_plate",
    "dice_brr",
    "dice_rbr",
    "dice_rbb",
    "dice_brb",
    "dice_bbr",
    "dice_rrb",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("data/ur5_maniunicon"))
    parser.add_argument("--output-repo-id", default="local/ur5_maniunicon")
    parser.add_argument("--tasks", nargs="+", default=ALL_TASKS, choices=ALL_TASKS)
    args = parser.parse_args()

    if args.output_root.exists():
        raise FileExistsError(f"{args.output_root} already exists; remove it to rebuild the merged dataset.")

    snapshot = Path(
        snapshot_download(
            HUB_REPO_ID, repo_type="dataset", allow_patterns=[f"{task}/*" for task in args.tasks]
        )
    )
    aggregate_datasets(
        repo_ids=[f"{HUB_REPO_ID}/{task}" for task in args.tasks],
        roots=[snapshot / task for task in args.tasks],
        aggr_repo_id=args.output_repo_id,
        aggr_root=args.output_root,
    )
    logging.info("Merged %d tasks into %s", len(args.tasks), args.output_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
