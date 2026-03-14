#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

from pathlib import Path

import lightning as L
from optree import PyTree

import src.datasets.pytree_utils as ptu
import src.utils.icon_core_utils as cu


class SaveMetric(L.Callback):
    def __init__(
        self,
        dirpath: str,
    ) -> None:
        super().__init__()
        self.dirpath = dirpath

    def on_validation_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        """Cache valid batch outputs. Only save metrics since they are smaller than preds and errors."""
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
        valid_dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
        self._save_metrics(valid_dirpath, batch, outputs, trainer.global_rank)

    def on_test_batch_end(self, trainer, pl_module, outputs: dict, batch: PyTree, batch_idx, dataloader_idx=0):
        """Cache test batch outputs. Only save metrics since they are smaller than preds and errors."""
        dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
        test_dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
        self._save_metrics(test_dirpath, batch, outputs, trainer.global_rank)

    def _save_metrics(self, dirpath: Path, batch: PyTree, outputs: dict, rank: int) -> None:
        dirpath.mkdir(parents=True, exist_ok=True)

        # save descriptions
        full_path = dirpath / f"description_rank{rank}.txt"
        description = ptu.get_discription_list(batch)  # [str] * bs
        with open(full_path, "a") as f:
            for desc in description:
                f.write(desc + "\n")

        # save metrics
        for key, tensor in ptu.to_numpy(outputs["metrics"]).items():
            file_key = key.replace("/", "_")
            full_path = dirpath / f"{file_key}_rank{rank}.csv"
            with open(full_path, "a") as f:
                if tensor.ndim == 0:  # scalar, sometimes metrics are not sample-wise
                    f.write(str(tensor) + "\n")
                else:  # (bs, ...)
                    for t in tensor:  # one line per sample
                        f.write(",".join(map(str, t.flatten())) + "\n")
