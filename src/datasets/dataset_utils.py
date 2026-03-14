#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch
from torch.utils import data as torch_data


def get_random_state_description(idx: int) -> str:
    """
    Get the random state description of the current sample.
    idx: the index of the current sample, argument of dataset.__getitem__
    """
    worker_id = torch_data.get_worker_info().id if torch_data.get_worker_info() is not None else 0
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    description = f"r/w: {rank}/{worker_id}, idx: {idx}, random state: {torch.randn(1).item()}"
    return description
