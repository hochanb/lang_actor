"""`lerobot-train` that keeps only the newest N step checkpoints.

Accepts exactly the same arguments as `lerobot-train`. After each checkpoint is saved (right when the `last`
symlink is updated), older `checkpoints/<step>` directories beyond the newest `KEEP_LAST_CHECKPOINTS`
(env var, default 10) are deleted.
"""

import logging
import os
import shutil
from pathlib import Path

from lerobot.scripts import lerobot_train

KEEP_LAST_CHECKPOINTS = int(os.environ.get("KEEP_LAST_CHECKPOINTS", "10"))

_update_last_checkpoint = lerobot_train.update_last_checkpoint


def _update_last_checkpoint_and_prune(checkpoint_dir: Path) -> None:
    _update_last_checkpoint(checkpoint_dir)
    step_dirs = sorted(
        (d for d in checkpoint_dir.parent.iterdir() if d.is_dir() and not d.is_symlink() and d.name.isdigit()),
        key=lambda d: int(d.name),
    )
    for old in step_dirs[:-KEEP_LAST_CHECKPOINTS]:
        logging.info("Removing old checkpoint %s (keeping newest %d)", old, KEEP_LAST_CHECKPOINTS)
        shutil.rmtree(old, ignore_errors=True)


lerobot_train.update_last_checkpoint = _update_last_checkpoint_and_prune

if __name__ == "__main__":
    lerobot_train.main()
