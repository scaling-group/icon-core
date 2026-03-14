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


class VizRolloutError(Viz):
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
        self.category = "viz_rollout_error"

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
        """
        rollout_metrics: (n_samples, n_steps) - L2 norm of errors
        """
        rollout_metrics = to_numpy(outputs["metrics"]["rollout_error"])

        n_samples = rollout_metrics.shape[0]
        n_steps = rollout_metrics.shape[1]

        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Create time steps array
        x_time = np.arange(n_steps)

        # Create a colormap for individual trajectories using rainbow colors
        traj_colors = plt.cm.gist_rainbow(np.linspace(0, 0.8, n_samples))

        # Plot individual trajectories with different colors
        for i in range(n_samples):
            ax.plot(x_time, rollout_metrics[i, :n_steps], color=traj_colors[i], alpha=0.3, linewidth=1)

        # Calculate and plot mean trajectory with contrasting color
        mean_metrics = np.mean(rollout_metrics[:n_samples, :n_steps], axis=0)
        std_metrics = np.std(rollout_metrics[:n_samples, :n_steps], axis=0)

        # Use complementary color for mean and shaded area
        ax.plot(x_time, mean_metrics, color="black", linewidth=2.5, label="Mean Error")
        ax.fill_between(x_time, mean_metrics - std_metrics, mean_metrics + std_metrics, color="black", alpha=0.3)

        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_title("Rollout Error", fontsize=14)
        ax.set_xlabel("Time Step", fontsize=12)
        ax.set_ylabel("Rollout Error", fontsize=12)
        if np.min(rollout_metrics) >= 0:
            ax.set_ylim(bottom=0)

        # Add epoch and step information
        fig.suptitle(f"Epoch {trainer.current_epoch}, Step {trainer.global_step}", fontsize=16)

        # Optimize layout
        plt.tight_layout()

        # Convert to PIL image
        img = vu.merge_images([[fig]])
        plt.close(fig)

        return img
