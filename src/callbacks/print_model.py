#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import lightning as L
from tabulate import tabulate

from src.utils.rich_utils import print_config_tree


class PrintModel(L.Callback):
    def on_train_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # print the model details customly
        pl_module.print(type(pl_module.net))
        model = pl_module.net.module if hasattr(pl_module.net, "module") else pl_module.net
        headers = ["Parameter Name", "Shape", "Requires Grad"]
        table_data = [(name, str(param.shape), param.requires_grad) for name, param in model.named_parameters()]
        pl_module.print(tabulate(table_data, headers=headers, tablefmt="grid"))

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        pl_module.print(f"Total Parameters: {total_params:,}")
        pl_module.print(f"Trainable Parameters: {trainable_params:,}")

        pl_module.print(f"SDPA backends: {pl_module.sdpa_backends}")
        print_config_tree(pl_module.cfg, resolve=True)
