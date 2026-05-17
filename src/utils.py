import logging
import random
from pathlib import Path

import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_checkpoint(
    state: dict,
    is_best: bool,
    checkpoint_dir: Path,
):
    """
    Save the current training checkpoint.

    Args:
        state: A dictionary containing the model's state and optimizer's state.
        is_best: A boolean flag to determine if the current checkpoint is the best based on validation loss.
        checkpoint_dir: The directory path where checkpoints are saved.
    """
    last_path = checkpoint_dir / "last.pth"
    best_path = checkpoint_dir / "best.pth"
    try:
        torch.save(state, last_path)
        logging.info(f"Checkpoint saved successfully at {last_path}")

        if is_best:
            torch.save(state["model"], best_path)
            logging.info(f"New best checkpoint saved successfully at {best_path}")
    except OSError as e:
        logging.error(f"Saving checkpoint failed: {e}", exc_info=True)
        raise
