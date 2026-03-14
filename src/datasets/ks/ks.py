#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class KSDataset(Dataset):
    """Kuramoto-Sivashinsky (KS) equation simulation dataset.

    Dataset Information
    -------------------
    Source: Kuramoto-Shivashinsky (KS) equation simulations
    Original Authors: Brandstetter, Johannes and Welling, Max and Worrall, Daniel E
    Repository: https://github.com/brandstetter-johannes/LPSDA
    License: MIT License

    To generate the data:
        1. git clone https://github.com/brandstetter-johannes/LPSDA.git
        2. cd LPSDA
        3. python generate/generate_data.py --experiment=KS --train_samples=512 \
               --valid_samples=512 --test_samples=512 --L=64 --nt=500

    Data Structure
    --------------
    Three HDF5 files: KS_train_512.h5, KS_valid_512.h5, KS_test_512.h5.
    Each file contains a top-level group named after the split ('train', 'valid', 'test'),
    with the following datasets:

    pde_{nt}-{nx}:
        KS equation solution tensor.
        Shape: (512, 140, 256) = samples x time_steps x spatial_points.
        512 independent samples, each retaining the last 140 time steps (nt_effective=140),
        with 256 spatial discretization points (nx=256).

    x:
        Spatial coordinates for each sample.
        Shape: (512, 256) = samples x spatial_points.
        Values from 0 to L*(1-1/256) = 63.75, with step size L/nx = 0.25.

    dx:
        Spatial grid spacing.
        Shape: (512,). Each element equals 0.25.

    t:
        Temporal coordinates.
        Shape: (512, 140) = samples x time_steps.
        Physical time moments for the last 140 time points of each trajectory.
        Time sampling: 500 equally-spaced points in [0, T] then truncated to last 140 steps,
        where T randomly varies between 90 and 110.

    dt:
        Temporal step size.
        Shape: (512,). Value = T/(nt-1) = T/499 per sample; varies slightly due to random T.

    License
    -------
    MIT License

    Copyright (c) 2023 brandstetter-johannes

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    def __init__(
        self,
        path: str,
        split: str,
        nt: int,
        nx: int,
        n_input_times: int,
        n_output_times: int,
        min_time_step: int,
        max_time_step: int,
    ):
        """
        path: path to dataset
        split: [train, valid, test]
        nt: temporal resolution
        nx: spatial resolution
        """
        super().__init__()
        self.split = split
        self.dataset = f"pde_{nt}-{nx}"
        self.file = h5py.File(path, "r")
        self.n_traj = self.file[self.split][self.dataset].shape[0]
        self.n_input_times = n_input_times
        self.n_output_times = n_output_times
        self.min_time_step = min_time_step
        self.max_time_step = max_time_step
        self._build_metadata()

    def _build_metadata(self):
        steps = self.file[self.split][self.dataset].shape[1]
        steps = min(steps, self.max_time_step + 1) - max(0, self.min_time_step)
        windows_per_traj = max(0, steps - self.n_input_times - self.n_output_times + 1)

        self.n_steps_per_traj = steps
        self.n_windows_per_traj = windows_per_traj
        self.time_index_offset = max(self.min_time_step, 0)
        self.length = self.n_traj * self.n_windows_per_traj

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        sample_idx = idx // self.n_windows_per_traj
        time_idx = idx % self.n_windows_per_traj + self.time_index_offset
        time_in_offset = time_idx + self.n_input_times
        input_fields = torch.Tensor(self.file[self.split][self.dataset][sample_idx, time_idx:time_in_offset]).unsqueeze(
            0
        )
        output_fields = torch.Tensor(
            self.file[self.split][self.dataset][sample_idx, time_in_offset : (time_in_offset + self.n_output_times)]
        ).unsqueeze(0)
        x = torch.Tensor(self.file[self.split]["x"][sample_idx])
        input_t = torch.Tensor(self.file[self.split]["t"][sample_idx, time_idx:time_in_offset])
        output_t = torch.Tensor(
            self.file[self.split]["t"][sample_idx, time_in_offset : (time_in_offset + self.n_output_times)]
        )
        description = f"KS equation, input_t={input_t}, output_t={output_t[0:2]}..., x_shape={x.shape}"

        return {
            "description": np.array([description], dtype=np.dtypes.StringDType()),
            "data": input_fields,
            "label": output_fields,
        }
