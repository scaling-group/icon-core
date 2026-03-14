#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################
from abc import ABC, abstractmethod
from collections.abc import Sequence

import hydra
import lightning as L
import torch
from omegaconf import DictConfig
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class BaseLitModule(L.LightningModule, ABC):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()

        self.save_hyperparameters(logger=False)
        self.cfg = cfg

        self.net = hydra.utils.instantiate(cfg.model)
        # compile the network only once, skip if it's already been compiled.
        self._net_compiled = False

        sdpa_map = {
            "cudnn": SDPBackend.CUDNN_ATTENTION,
            "math": SDPBackend.MATH,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "flash": SDPBackend.FLASH_ATTENTION,
        }

        self.sdpa_backends = [sdpa_map[backend] for backend in self.cfg.accelerate.sdpa]

    def _model_forward(self, *args, **kwargs):
        with sdpa_kernel(self.sdpa_backends):
            return self.net(*args, **kwargs)

    def setup(self, stage: str) -> None:
        if self.cfg.accelerate.compile and stage == "fit" and torch.__version__ >= "2.0.0" and not self._net_compiled:
            self.net = torch.compile(self.net)
            self._net_compiled = True

    def get_lr_scheduler(self, optimizer):
        scheduler = hydra.utils.instantiate(self.cfg.opt.scheduler)(optimizer=optimizer)
        return {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        }

    #! you need to specify which nets should be optimized
    @abstractmethod
    def get_trainable_networks(self) -> nn.Module | Sequence[nn.Module]: ...

    def get_optimizer(self):
        nets = self.get_trainable_networks()
        if isinstance(nets, nn.Module):
            nets = [nets]

        if self.cfg.opt.optimizer._target_ == "torch.optim.AdamW":
            optimizer = hydra.utils.instantiate(self.cfg.opt.optimizer)(
                params=[p for net in nets for p in net.parameters() if p.requires_grad],
            )
        elif self.cfg.opt.optimizer._target_ == "src.opt.optimizers.muon.Muon":
            from src.opt.optimizers.muon import Muon

            named_params = [(name, p) for net in nets for (name, p) in net.named_parameters()]
            muon_params, adamw_params = Muon.split_muon_adamw_params(named_params)

            optimizer = hydra.utils.instantiate(self.cfg.opt.optimizer)(
                muon_params=muon_params,
                adamw_params=adamw_params,
            )
        else:
            raise ValueError(f"Optimizer {self.cfg.opt.optimizer} not supported")
        return optimizer

    def configure_optimizers(self):
        optimizer = self.get_optimizer()
        lr_scheduler = self.get_lr_scheduler(optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
        }
