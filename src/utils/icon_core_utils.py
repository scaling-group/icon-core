#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

"""
this file contains custom utils not included in the original template but useful for our research.
For project-specific utils, please create new files like xxx_project_utils.py
"""

from omegaconf import DictConfig


def get_dataset_name(data_cfg: DictConfig, dataloader_idx: int) -> str:
    """
    usually cfg.data.valid/test is a dict of datasets
    this function returns the name of the dataset for the given dataloader index
    """
    key = list(data_cfg.keys())[dataloader_idx]
    dataset_name = data_cfg[key].name
    return dataset_name
