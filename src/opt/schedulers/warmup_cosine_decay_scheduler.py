#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import numpy as np
from torch import optim


class PercentageWarmupCosineDecayScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iters, warmup_percent, decay_percent, end_lr_factor):
        self.warmup_iters = int(warmup_percent * max_iters // 100)
        self.decay_iters = int(decay_percent * max_iters // 100)
        self.end_lr_factor = end_lr_factor
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        if epoch <= self.warmup_iters:
            lr_factor = epoch * 1.0 / max(self.warmup_iters, 1)
        else:
            progress = (epoch - self.warmup_iters) / (self.decay_iters - self.warmup_iters)
            # Clamp progress to [0, 1] to handle cases where epoch > max_iters
            progress = np.clip(progress, 0.0, 1.0)
            # Cosine decay from 1.0 to end_lr_factor
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            lr_factor = self.end_lr_factor + (1.0 - self.end_lr_factor) * cosine_decay
        return lr_factor
