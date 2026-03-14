#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################


import einops
import torch
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.icon_core_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class NopRolloutLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        # Training metrics
        self.train_metrics = MetricCollection(
            {
                "loss": MeanMetric(),
            }
        )

        # Validation metrics
        self.metric_names = [
            "rollout_error",
            "error_step_0",
            "error_step_1",
        ]

        self.valid_metrics = torch.nn.ModuleDict(
            {
                self.cfg.data.valid[key].name: MetricCollection({k: MeanMetric() for k in self.metric_names})
                for key in self.cfg.data.valid
            }
        )

    def get_trainable_networks(self):
        return self.net

    def _loss_function(self, batch: PyTree) -> torch.Tensor:
        latent = self.rollout_preprocess(batch["data"])
        latent = self.rollout_step(latent)
        pred = self.rollout_postprocess(latent)
        diff = pred - batch["label"]
        return einops.reduce(diff**2, "b ... -> b", "mean")

    def rollout_preprocess(self, data: PyTree) -> PyTree:
        """
        Conduct rollout preprocess, you can override this function
        input: batch["data"]
        currently just a tensor, but can be a PyTree in general
        """
        return data

    def rollout_step(self, latent: PyTree) -> PyTree:
        """
        conduct rollout step, you can override this function
        the input is the output of rollout_preprocess or the previous rollout_step
        apart from model forward, you can also add processings like concatenating conditions/control inputs, etc.
        """
        latent = self._model_forward(latent)
        return latent

    def rollout_postprocess(self, latent: PyTree) -> PyTree:
        """
        conduct rollout postprocess, you can override this function
        the input is the output of rollout_step
        """
        return latent

    def rollout(self, batch: PyTree) -> PyTree:
        """
        only for validation
        label for valid is different from train: train:[batch, 1, ...], valid:[batch, time_steps, ...]
        for n steps, rollout consists of:
        rollout_preprocess, rollout_step * n, rollout_postprocess
        """

        rollout_steps = batch["label"].shape[1]
        latents = []
        preds = []

        with torch.no_grad():
            latents.append(self.rollout_preprocess(batch["data"]))
            for _ in range(rollout_steps):
                latents.append(self.rollout_step(latents[-1]))
            for latent in latents:
                preds.append(self.rollout_postprocess(latent))

        latents = torch.cat(latents, dim=1)
        preds = torch.cat(preds, dim=1)
        return {"preds": preds, "latents": latents}

    def get_rollout_errors(self, preds: PyTree, batch: PyTree) -> PyTree:
        """
        Calculate errors between predictions and ground truth.
        Args:
            preds: PyTree containing predictions from rollout
        Returns:
            PyTree containing errors with full dimensions
        """
        all_labels = torch.cat([batch["data"], batch["label"]], dim=1)
        errors = preds["preds"] - all_labels
        error_step_0 = errors[:, 0, ...]
        error_step_1 = errors[:, 1, ...]
        return {
            "rollout_errors": errors,
            "error_step_0": error_step_0,
            "error_step_1": error_step_1,
        }

    def get_rollout_metrics(self, errors: PyTree) -> PyTree:
        """
        Calculate metrics from errors, maintaining batch dimension.
        Args:
            errors: PyTree containing errors from get_rollout_errors
        Returns:
            PyTree containing metrics with batch dimension
        """
        metrics = einops.reduce(errors["rollout_errors"] ** 2, "b t ... -> b t", "mean") ** 0.5  # L2 norm, keep t dim
        metrics_step_0 = einops.reduce(errors["error_step_0"] ** 2, "b ... -> b", "mean") ** 0.5
        metrics_step_1 = einops.reduce(errors["error_step_1"] ** 2, "b ... -> b", "mean") ** 0.5
        return {
            "rollout_error": metrics,
            "error_step_0": metrics_step_0,
            "error_step_1": metrics_step_1,
        }

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics.values():
            metrics.reset()

    def training_step(self, batch: PyTree, batch_idx: int) -> torch.Tensor:
        loss = self._loss_function(batch)
        # Update metrics
        self.train_metrics["loss"](loss)
        # Log metrics
        self.log("train/loss", self.train_metrics["loss"], on_step=True, on_epoch=False)
        return loss.mean()  # pool over batch in the end

    ############ validation #############
    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0):
        """
        Returns:
            dict[str, torch.Tensor]:
                preds (dict[str, torch.Tensor]): predicted values
                errors (dict[str, torch.Tensor]): errors, not pooled
                metrics (dict[str, torch.Tensor]): metrics, in the similar shape of (bs, )
        """
        preds = self.rollout(batch)
        errors = self.get_rollout_errors(preds, batch)
        metrics = self.get_rollout_metrics(errors)

        dataset_name = cu.get_dataset_name(self.cfg.data.valid, dataloader_idx)

        for metric_name in self.valid_metrics[dataset_name]:
            self.valid_metrics[dataset_name][metric_name].update(metrics[metric_name])

        for metric_name in self.valid_metrics[dataset_name]:
            self.log(
                f"{dataset_name}/{metric_name}",
                self.valid_metrics[dataset_name][metric_name],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )

        return {"preds": preds, "errors": errors, "metrics": metrics}
