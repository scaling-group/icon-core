#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import FNO1d as NeuralOpFNO1d


class FNO1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        num_layers: int,
        num_modes: int,
    ):
        super().__init__()
        self.model = NeuralOpFNO1d(
            n_modes_height=num_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=hidden_channels * 2,
            projection_channels=hidden_channels * 2,
            n_layers=num_layers,
            non_linearity=F.gelu,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is [B, 1, L]
        # FNO1D expects [B, in_channels, L]
        # No need to permute since dimensions already match in KS case

        # Apply FNO model
        out = self.model(x)

        return out
