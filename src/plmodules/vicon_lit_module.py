#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from optree import PyTree
from torchmetrics import MeanMetric, MetricCollection

import src.utils.icon_core_utils as cu
from src.plmodules.base_lit_module import BaseLitModule


class ViconLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

        self.train_metrics = MeanMetric()

        # Use MetricCollection to group metrics
        self.metric_names = [
            "loss",  # total loss
            "error",  # error
            # add more metrics here
        ]

        self.valid_metrics = torch.nn.ModuleDict(
            {
                self.cfg.data.valid[key].name: MetricCollection({k: MeanMetric() for k in self.metric_names})
                for key in self.cfg.data.valid
            }
        )

    def get_trainable_networks(self):
        return self.net

    def _prompt_normalization(self, x: torch.Tensor):
        mean = x.mean(dim=(1, 3, 4), keepdim=True)  # Mean across seq, H, W -> (batch_size, 1, dim, 1, 1)
        std = x.std(dim=(1, 3, 4), keepdim=True) + 1e-5  # Std across seq, H, W -> (batch_size, 1, dim, 1, 1)

        x_normalized = (x - mean) / std

        return x_normalized, mean, std

    def network_inference(self, data: PyTree):
        dummy_label = torch.zeros_like(data["ex_g"][:, -1:, :, :, :])
        g = data["ex_g"]
        f = torch.cat((data["ex_f"], data["qn_f"]), dim=1)
        f_norm, f_mean, f_std = self._prompt_normalization(f)
        g_norm, g_mean, g_std = self._prompt_normalization(g)
        g_norm = torch.cat((g_norm, dummy_label), dim=1)

        outputs = self._model_forward(f_norm, g_norm)
        # denormalize the predicted g using the mean and std of the g
        denormalized_outputs = {}
        for key, tensor in outputs.items():
            denormalized_outputs[key] = tensor * g_std + g_mean

        return denormalized_outputs

    def _get_ground_truth_all(self, batch: PyTree):
        g = batch["data"]["ex_g"]
        ground_truth = torch.cat((g, batch["label"]), dim=1)
        return ground_truth

    def _get_pred_all(self, outputs: dict):
        ex_pred = outputs["ex_pred"]
        qn_pred = outputs["qn_pred"]
        all_pred = torch.cat([ex_pred, qn_pred], dim=1)
        return all_pred

    def _get_pred_qn(self, outputs: dict):
        qn_pred = outputs["qn_pred"]
        return qn_pred

    def _loss_function(self, pred: torch.Tensor, target: torch.Tensor):
        return F.mse_loss(pred, target)

    def _loss_all(self, batch: PyTree) -> torch.Tensor:
        # used for training
        outputs = self.network_inference(batch["data"])
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(batch)
        loss = self._loss_function(all_pred, all_ground_truth)

        return loss

    def _error_all(self, batch: PyTree) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.network_inference(batch["data"])
        all_pred = self._get_pred_all(outputs)
        all_ground_truth = self._get_ground_truth_all(batch)
        error = all_pred - all_ground_truth
        return all_pred, error

    def _loss_qn(self, batch: PyTree) -> torch.Tensor:
        data = batch["data"]
        label = batch["label"]
        outputs = self.network_inference(data)
        qn_pred = self._get_pred_qn(outputs)
        loss = self._loss_function(qn_pred, label)
        return loss

    def _error_qn(self, batch: PyTree) -> torch.Tensor:
        data = batch["data"]
        label = batch["label"]
        outputs = self.network_inference(data)
        qn_pred = self._get_pred_qn(outputs)
        error = qn_pred - label
        return error

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics.values():
            metrics.reset()

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        loss = self._loss_all(batch)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss

    ############ validation #############
    def validation_step(self, batch, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        loss = self._loss_all(batch)
        preds, errors = self._error_all(batch)

        metrics = {"loss": loss.mean(), "error": errors.mean()}

        valid_name = cu.get_dataset_name(self.cfg.data.valid, dataloader_idx)
        for metric_name in self.metric_names:
            self.valid_metrics[valid_name][metric_name].update(metrics[metric_name])

        for metric_name in self.metric_names:
            self.log(
                f"{valid_name}/{metric_name}",
                self.valid_metrics[valid_name][metric_name],
                on_step=False,
                on_epoch=True,
                add_dataloader_idx=False,
            )
        return {"preds": preds, "errors": errors, "metrics": metrics}
