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


class DummyViconDataset(Dataset):
    def __init__(self, ex_num: int, f_shape: tuple[int, int, int], g_shape: tuple[int, int, int]):
        self.ex_num = ex_num
        self.f_shape = f_shape  # (f_dim, f_h, f_w)
        self.g_shape = g_shape  # (g_dim, g_h, g_w)

    def __len__(self):
        return 500

    def __getitem__(self, idx):
        description = ""
        description += f"dataset: {self.__class__.__name__}, "
        description += dsu.get_random_state_description(idx)

        ex_f = torch.randn(1, self.ex_num, *self.f_shape)
        ex_g = torch.randn(1, self.ex_num, *self.g_shape)
        qn_f = torch.randn(1, 1, *self.f_shape)
        qn_g = torch.randn(1, 1, *self.g_shape)

        data = {
            "ex_f": ex_f,
            "ex_g": ex_g,
            "qn_f": qn_f,
        }

        label = qn_g

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": data,
            "label": label,
        }
