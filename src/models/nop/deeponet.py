#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch
import torch.nn as nn
from deepxde.nn.pytorch import DeepONetCartesianProd as DeepXDeDeepONet


class DeepONet(nn.Module):
    """
    Model from: Learning nonlinear operators via DeepONet based on the universal approximation theorem of operators.
    """

    def __init__(
        self,
        branch_layer_sizes: list[int],
        trunk_layer_sizes: list[int],
        activation: str = "relu",
    ):
        super().__init__()
        self.model = DeepXDeDeepONet(
            layer_sizes_branch=branch_layer_sizes,
            layer_sizes_trunk=trunk_layer_sizes,
            activation=activation,
            kernel_initializer="Glorot normal",
        )

    def forward(self, x_branch: torch.Tensor, x_trunk: torch.Tensor) -> torch.Tensor:
        # x_branch: [B, branch_input_dim] — function evaluations at sensor points
        # x_trunk:  [B, trunk_input_dim]  — query locations
        return self.model((x_branch, x_trunk))
