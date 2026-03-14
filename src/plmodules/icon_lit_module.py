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


class IconLitModule(BaseLitModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self._net_compiled = True  # hard code to True, doesn't support compile for now

        self.train_metrics = MeanMetric()

        # Use MetricCollection to group metrics
        self.metric_names = [
            "loss",  # total loss
            "quest_qoi_v",  # error
        ]

        self.valid_metrics = torch.nn.ModuleDict(
            {
                self.cfg.data.valid[key].name: MetricCollection({k: MeanMetric() for k in self.metric_names})
                for key in self.cfg.data.valid
            }
        )

    def get_trainable_networks(self):
        return self.net

    def network_inference(self, data: PyTree, **kwargs):
        """Network inference for ICON model"""

        # Forward pass
        outputs = self._model_forward(data=data, **kwargs)
        return outputs

    def _build_train_label(self, batch: PyTree):
        demo_qoi_v = batch["data"]["demo_qoi_v"][:, self.cfg.loss.shot_num_min :, :, :]
        train_label = torch.cat([demo_qoi_v, batch["label"]], dim=1)
        return train_label

    def _loss_function(self, batch: PyTree) -> torch.Tensor:
        """Loss function for ICON model"""
        pred = self.network_inference(batch["data"], mode="train")
        train_label = self._build_train_label(batch)
        return einops.reduce((pred - train_label) ** 2, "b num qoi_len dim -> b", "mean")

    def get_preds(self, data: PyTree) -> torch.Tensor:
        """Get predictions"""
        quest_qoi_v, attn_weights = self.network_inference(data, mode="test", need_weights=True)
        return {
            "quest_qoi_v": quest_qoi_v,
            "attn_weights": attn_weights,
        }

    def get_errors(self, preds, batch: PyTree) -> torch.Tensor:
        """Get error metrics"""
        return {
            "quest_qoi_v": einops.reduce(
                torch.abs(preds["quest_qoi_v"] - batch["label"]), "b 1 qoi_len dim -> b", "mean"
            ),
        }

    ############ training #############

    def on_train_start(self) -> None:
        for metrics in self.valid_metrics.values():
            metrics.reset()

    def training_step(self, batch: PyTree, batch_idx: int) -> torch.Tensor:
        loss = self._loss_function(batch)

        self.train_metrics(loss)
        self.log("train/loss", self.train_metrics, on_step=True, on_epoch=True)

        return loss.mean()

    ############ validation #############
    def validation_step(self, batch: PyTree, batch_idx: int, dataloader_idx: int = 0) -> dict:
        loss = self._loss_function(batch)
        preds = self.get_preds(batch["data"])
        errors = self.get_errors(preds, batch)

        metrics = {"loss": loss, **errors}

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
