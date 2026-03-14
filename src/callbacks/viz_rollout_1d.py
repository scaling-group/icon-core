#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image as Image

from src.datasets.pytree_utils import to_numpy

from . import viz_utils as vu
from .viz import Viz


class VizRollout1D(Viz):
    def __init__(
        self,
        dirpath: str,
        valid_batches_local: str,
        valid_batches_log: str,
        test_batches_local: str,
        test_batches_log: str,
    ):
        super().__init__(
            dirpath,
            valid_batches_local,
            valid_batches_log,
            test_batches_local,
            test_batches_log,
        )
        self.category = "viz_rollout_1d"

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
        """
        rollout_errors: (n_samples, n_steps, L)
        Visualize the spatial error distribution with color gradient
        """
        rollout_errors = to_numpy(outputs["errors"]["rollout_errors"])

        n_steps = rollout_errors.shape[1]
        n_points = rollout_errors.shape[2]  # L dimension
        x_spatial = np.arange(n_points)
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create a colormap for time steps
        colors = plt.cm.viridis(np.linspace(0, 1, n_steps))

        # Plot initial and final states with thicker lines
        mean_error_initial = np.mean(rollout_errors[:, 0, :], axis=0)
        mean_error_final = np.mean(rollout_errors[:, -1, :], axis=0)
        ax.plot(x_spatial, mean_error_initial, color=colors[0], linewidth=2, label="Initial")
        ax.plot(x_spatial, mean_error_final, color=colors[-1], linewidth=2, label="Final")

        # Plot intermediate states with thinner lines and gradient colors
        for t in range(1, n_steps - 1):
            mean_error = np.mean(rollout_errors[:, t, :], axis=0)
            ax.plot(x_spatial, mean_error, color=colors[t], alpha=0.3, linewidth=0.5)

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_title("Spatial Error Distribution", fontsize=14)
        ax.set_xlabel("Spatial Points", fontsize=12)
        ax.set_ylabel("Error", fontsize=12)
        ax.legend(loc="upper right", fontsize=10)

        # Add epoch and step information
        fig.suptitle(f"Epoch {trainer.current_epoch}, Step {trainer.global_step}", fontsize=16)

        # Optimize layout
        plt.tight_layout()

        # Convert to PIL image
        img = vu.merge_images([[fig]])
        plt.close(fig)

        return img
