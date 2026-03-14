#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

from pathlib import Path

import lightning as L
from rich import print as rprint

import src.datasets.pytree_utils as ptu
import src.utils.icon_core_utils as cu


class SaveBatchInfo(L.Callback):
    def __init__(
        self,
        dirpath: str,
        print_lv_local: int,
        print_lv_log: int,
        train_batches_local: str,
        train_batches_log: str,
        valid_batches_local: str,
        valid_batches_log: str,
        test_batches_local: str,
        test_batches_log: str,
    ):
        self.dirpath = dirpath
        self.print_lv_local = print_lv_local
        self.print_lv_log = print_lv_log
        self.train_batches_local = eval(train_batches_local)
        self.train_batches_log = eval(train_batches_log)
        self.valid_batches_local = eval(valid_batches_local)
        self.valid_batches_log = eval(valid_batches_log)
        self.test_batches_local = eval(test_batches_local)
        self.test_batches_log = eval(test_batches_log)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        # Get actual dataloader index from batch if available (for WeightedCycleLoader)
        dataloader_idx = (
            batch.get("_dataloader_idx")
            if isinstance(batch, dict) and "_dataloader_idx" in batch
            else batch_idx % len(pl_module.cfg.data.train)
        )

        if batch_idx in self.train_batches_log or batch_idx in self.train_batches_local:
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.train, dataloader_idx)

        if batch_idx in self.train_batches_log and trainer.global_rank == 0:
            tree = ptu.get_print_info(
                batch,
                print_lv=self.print_lv_log,
                info=f"Train Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx}",
            )
            rprint(tree)
            rprint("")  # add a newline after each batch

        if batch_idx in self.train_batches_local:
            tree = ptu.get_print_info(
                batch,
                print_lv=self.print_lv_local,
                info=f"Train Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx}",
            )
            dirpath = Path(self.dirpath) / "train"
            dirpath.mkdir(parents=True, exist_ok=True)
            filename = Path(self.dirpath) / "train" / f"rank_{trainer.global_rank}.txt"
            with open(filename, "a") as f:  # save to file, append to file end
                rprint(tree, file=f)
                rprint("", file=f)  # add a newline after each batch

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx in self.valid_batches_log or batch_idx in self.valid_batches_local:
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)

        if batch_idx in self.valid_batches_log and trainer.global_rank == 0:
            tree = ptu.get_print_info(
                batch,
                print_lv=self.print_lv_log,
                info=f"Valid Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx}",
            )
            rprint(tree)
            rprint("")  # add a newline after each batch

        if batch_idx in self.valid_batches_local:
            tree = ptu.get_print_info(
                batch,
                print_lv=self.print_lv_local,
                info=f"Valid Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx}",
            )
            dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
            dirpath.mkdir(parents=True, exist_ok=True)
            filename = (
                Path(self.dirpath)
                / "valid"
                / f"step_{trainer.global_step}"
                / dataset_name
                / f"rank_{trainer.global_rank}.txt"
            )

            with open(filename, "a") as f:  # save to file, append to file end
                rprint(tree, file=f)
                rprint("", file=f)  # add a newline after each batch

    def on_test_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        if batch_idx in self.test_batches_log or batch_idx in self.test_batches_local:
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)

        if batch_idx in self.test_batches_log and trainer.global_rank == 0:
            tree = ptu.get_print_info(
                batch,
                print_lv=self.print_lv_log,
                info=f"Test Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx}",
            )
            rprint(tree)
            rprint("")  # add a newline after each batch

        if batch_idx in self.test_batches_local:
            tree = ptu.get_print_info(
                batch,
                print_lv=self.print_lv_local,
                info=f"Test Dataset # {dataloader_idx} - {dataset_name} - Batch {batch_idx}",
            )
            dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
            dirpath.mkdir(parents=True, exist_ok=True)
            filename = (
                Path(self.dirpath)
                / "test"
                / f"step_{trainer.global_step}"
                / dataset_name
                / f"rank_{trainer.global_rank}.txt"
            )

            with open(filename, "a") as f:  # save to file, append to file end
                rprint(tree, file=f)
                rprint("", file=f)  # add a newline after each batch
