#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################


"""
This is the dataset for the conservation law data used in the following paper:

@article{yang2024pde,
  title={{PDE} generalization of in-context operator networks: A study on {1D} scalar nonlinear conservation laws},
  author={Yang, Liu and Osher, Stanley J},
  journal={Journal of Computational Physics},
  volume={519},
  pages={113379},
  year={2024},
  publisher={Elsevier}
}

Data generation reference:
- https://github.com/scaling-group/icon-tutorial/tree/main/src/datagen (entry)
- https://github.com/scaling-group/icon-tutorial/blob/main/scripts/datagen.sh (example command)

"""

import glob
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class WenoDataset(Dataset):
    def __init__(self, file_paths, demo_num, base_seed=None):
        # Expand glob patterns if file_paths is a string
        if isinstance(file_paths, str):
            self.file_paths = sorted(glob.glob(file_paths))
        else:
            self.file_paths = file_paths

        log.info(f"{len(self.file_paths)} files found in {file_paths}: {self.file_paths}")

        self.demo_num = demo_num
        self.base_seed = base_seed
        self.indices = []
        self.file_handles = {}

        # Collecting a list of (file_path, group_name) for record indexing
        for file_path in self.file_paths:
            if not os.path.exists(file_path):
                log.warning(f"File {file_path} does not exist, skipping...")
                continue
            try:
                with h5py.File(file_path, "r") as f:
                    self.indices.extend([(file_path, key) for key in f])
            except Exception as e:
                log.warning(f"Could not read file {file_path}: {e}")
                continue

        log.info(f"total number of records: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # if seed is not None, use the seed and idx to control the randomness
        # this is useful for validation and testing,
        # so that the results can be reproduced regardless of workers, devices, etc.
        if self.base_seed is not None:
            rng = torch.Generator()
            rng.manual_seed(self.base_seed + idx)  # randomness is only from the seed and idx
        else:
            rng = None

        file_path, group_name = self.indices[idx]

        if file_path not in self.file_handles:
            self.file_handles[file_path] = h5py.File(file_path, "r")

        group = self.file_handles[file_path][group_name]
        equation = group["equation"][()].decode("utf-8")
        # param = torch.tensor(group["param"][:], dtype=torch.float32)

        # Check if data is already a tensor or numpy array
        cond_k_data = group["cond_k"][:]
        cond_v_data = group["cond_v"][:]
        qoi_k_data = group["qoi_k"][:]
        qoi_v_data = group["qoi_v"][:]

        # Convert to tensor properly
        if isinstance(cond_k_data, torch.Tensor):
            cond_k = cond_k_data
            cond_v = cond_v_data
            qoi_k = qoi_k_data
            qoi_v = qoi_v_data
        elif isinstance(cond_k_data, np.ndarray):
            cond_k = torch.tensor(cond_k_data, dtype=torch.float32)
            cond_v = torch.tensor(cond_v_data, dtype=torch.float32)
            qoi_k = torch.tensor(qoi_k_data, dtype=torch.float32)
            qoi_v = torch.tensor(qoi_v_data, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported data type: {type(cond_k_data)}")

        # random select from 100 pairs
        num_pairs = cond_k.shape[0]
        random_indices = torch.randperm(num_pairs, generator=rng)
        demo_indices = random_indices[: self.demo_num]
        quest_indices = random_indices[self.demo_num : self.demo_num + 1]

        # add the batch dimension
        demo_cond_k = cond_k[None, demo_indices, :, :]
        demo_cond_v = cond_v[None, demo_indices, :, :]
        demo_qoi_k = qoi_k[None, demo_indices, :, :]
        demo_qoi_v = qoi_v[None, demo_indices, :, :]
        quest_cond_k = cond_k[None, quest_indices, :, :]
        quest_cond_v = cond_v[None, quest_indices, :, :]
        quest_qoi_k = qoi_k[None, quest_indices, :, :]
        quest_qoi_v = qoi_v[None, quest_indices, :, :]

        demo_cond_mask = torch.ones_like(demo_cond_k, dtype=torch.bool)[..., 0]
        demo_qoi_mask = torch.ones_like(demo_qoi_k, dtype=torch.bool)[..., 0]
        quest_cond_mask = torch.ones_like(quest_cond_k, dtype=torch.bool)[..., 0]
        quest_qoi_mask = torch.ones_like(quest_qoi_k, dtype=torch.bool)[..., 0]

        # Convert to pytree format
        data = {
            "description": np.array([equation] * demo_cond_k.shape[0], dtype=np.dtypes.StringDType()),
            "data": {
                "demo_cond_k": demo_cond_k,
                "demo_cond_v": demo_cond_v,
                "demo_cond_mask": demo_cond_mask,
                "demo_qoi_k": demo_qoi_k,
                "demo_qoi_v": demo_qoi_v,
                "demo_qoi_mask": demo_qoi_mask,
                "quest_cond_k": quest_cond_k,
                "quest_cond_v": quest_cond_v,
                "quest_cond_mask": quest_cond_mask,
                "quest_qoi_k": quest_qoi_k,
                "quest_qoi_mask": quest_qoi_mask,
            },
            "label": quest_qoi_v,
        }

        return data

    def __del__(self):
        # Clean up file handles when the dataset is destroyed
        for file_handle in self.file_handles.values():
            if file_handle is not None:
                file_handle.close()


if __name__ == "__main__":
    from rich import print as rprint

    import src.datasets.pytree_utils as ptu

    dataset = WenoDataset(file_paths="./data/data_weno_cubic/train*.h5", demo_num=5)
    # Test loading a sample
    for i, sample in enumerate(dataset):
        if i >= 5:  # Only process first 5 samples
            break
        tree = ptu.get_print_info(sample, print_lv=1, info=f"Viola Test Sample # {i}")
        rprint(tree)
        rprint("")  # add a newline after the sample
