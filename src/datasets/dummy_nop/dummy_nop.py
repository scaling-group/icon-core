#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets import dataset_utils as dsu


class DummyNopDataset(Dataset):
    def __init__(self, f_len: int, g_len: int, fx_dim: int, fy_dim: int, gx_dim: int, gy_dim: int):
        self.f_len = f_len
        self.g_len = g_len
        self.fx_dim = fx_dim
        self.fy_dim = fy_dim
        self.gx_dim = gx_dim
        self.gy_dim = gy_dim

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        # get random state description in the beginning of __getitem__, to monitor the random state of each sample
        description = ""
        description += f"dataset: {self.__class__.__name__}, "
        description += dsu.get_random_state_description(idx)

        fx_samples = torch.randn(1, self.f_len, self.fx_dim)
        fy_samples = torch.randn(1, self.f_len, self.fy_dim)
        fm_samples = torch.ones(1, self.f_len, dtype=torch.bool)
        gx_samples = torch.randn(1, self.g_len, self.gx_dim)
        gm_samples = torch.ones(1, self.g_len, dtype=torch.bool)

        data = {
            "fx": fx_samples,
            "fy": fy_samples,
            "fm": fm_samples,
            "gx": gx_samples,
            "gm": gm_samples,
        }

        label = torch.randn(1, self.g_len, self.gy_dim)

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": data,
            "label": label,
        }
