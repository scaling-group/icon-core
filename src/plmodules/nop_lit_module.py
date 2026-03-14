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


class NopLitModule(BaseLitModule):
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

    def network_inference(self, data: PyTree):
        memory = torch.cat([data["fx"], data["fy"]], dim=-1)
        query = data["gx"]
        # mask is not supported yet
        outputs = self._model_forward(memory=memory, query=query)
        return outputs

    def _loss_function(self, batch: PyTree) -> torch.Tensor:
        pred = self.network_inference(batch["data"])
        return F.mse_loss(pred, batch["label"])

    def get_pred(self, data: PyTree) -> torch.Tensor:
        return self.network_inference(data)

    def get_error(self, batch: PyTree) -> torch.Tensor:
        pred = self.get_pred(batch["data"])
        return torch.abs(pred - batch["label"])

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics.values():
            metrics.reset()

    def training_step(self, batch: PyTree, batch_idx: int) -> torch.Tensor:
        loss = self._loss_function(batch)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss

    ############ validation #############
    def validation_step(self, batch: PyTree, batch_idx: int, dataloader_idx: int = 0) -> dict:
        loss = self._loss_function(batch)
        errors = self.get_error(batch)
        preds = self.get_pred(batch["data"])

        # TODO: suggest using sample-wise metrics, i.e. each of shape (batch, ...)
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
