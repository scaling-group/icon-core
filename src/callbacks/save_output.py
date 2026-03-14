#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import pickle
from pathlib import Path

import lightning as L
from optree import PyTree

import src.datasets.pytree_utils as ptu
import src.utils.icon_core_utils as cu


class SaveOutput(L.Callback):
    def __init__(
        self,
        dirpath: str,
        valid_batches_local: str,  # save batches in local machine
        test_batches_local: str,  # save batches in local machine
    ):
        super().__init__()
        self.dirpath = dirpath
        self.valid_batches_local = eval(valid_batches_local)
        self.test_batches_local = eval(test_batches_local)

    def on_validation_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        if batch_idx in self.valid_batches_local:
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
            valid_dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
            self._save_output(valid_dirpath, batch, outputs, batch_idx, trainer.global_rank)

    def on_test_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        if batch_idx in self.test_batches_local:
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
            test_dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
            self._save_output(test_dirpath, batch, outputs, batch_idx, trainer.global_rank)

    def _save_output(self, dirpath: Path, batch: PyTree, outputs: dict, batch_idx: int, rank: int) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)
        full_path = dirpath / f"{batch_idx}_rank{rank}.pkl"

        batch_np = ptu.to_numpy(batch)
        outputs_np = ptu.to_numpy(outputs)
        out_dict = {
            "batch": batch_np,
            "outputs": outputs_np,
        }

        with open(full_path, "wb") as f:
            pickle.dump(out_dict, f)
