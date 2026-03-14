#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

from pathlib import Path

import lightning as L
import matplotlib.pyplot as plt
from lightning.pytorch import loggers
from PIL import Image

import src.utils.icon_core_utils as cu

from . import viz_utils as vu


class Viz(L.Callback):
    def __init__(
        self,
        dirpath: str,
        valid_batches_local: str,  # save batches in local machine
        valid_batches_log: str,  # log batches to remote wandb
        test_batches_local: str,  # save batches in local machine
        test_batches_log: str,  # log batches to remote wandb
    ):
        super().__init__()
        self.dirpath = dirpath
        self.valid_batches_local = eval(valid_batches_local)
        self.valid_batches_log = eval(valid_batches_log)
        self.test_batches_local = eval(test_batches_local)
        self.test_batches_log = eval(test_batches_log)
        self.category = "viz_base"  # override this in the child class

    def get_image(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0) -> Image.Image:
        """
        This is just a dummy function to test the callback and show basic usage.
        you can inherit this callback class and override this function.
        """
        fig = plt.figure(figsize=(4, 3))
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2])
        img = vu.merge_images([[fig]])  # merge a list of list of matplotlib plots or PIL images
        plt.close("all")
        return img  # PIL image

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx in self.valid_batches_log or batch_idx in self.valid_batches_local:
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.valid, dataloader_idx)
            img = self.get_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if img is None:
                return

        if batch_idx in self.valid_batches_local:
            dirpath = Path(self.dirpath) / "valid" / f"step_{trainer.global_step}" / dataset_name
            dirpath.mkdir(parents=True, exist_ok=True)
            img.save(dirpath / f"{batch_idx}_rank{trainer.global_rank}.png")  # save image in all processes

        if batch_idx in self.valid_batches_log:
            self.log_image(
                trainer,
                img,
                key=f"{dataset_name}",
                artifact_file=f"valid/"
                f"{self.__class__.__name__}/"
                f"{dataset_name}/"
                f"step_{trainer.global_step}"
                f"_{batch_idx}_rank{trainer.global_rank}.png",
            )

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx in self.test_batches_log or batch_idx in self.test_batches_local:
            dataset_name = cu.get_dataset_name(pl_module.cfg.data.test, dataloader_idx)
            img = self.get_image(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx)
            if img is None:
                return

        if batch_idx in self.test_batches_local:
            dirpath = Path(self.dirpath) / "test" / f"step_{trainer.global_step}" / dataset_name
            dirpath.mkdir(parents=True, exist_ok=True)
            img.save(dirpath / f"{batch_idx}_rank{trainer.global_rank}.png")  # save image in all processes

        if batch_idx in self.test_batches_log:
            self.log_image(
                trainer,
                img,
                key=f"{dataset_name}",
                artifact_file=f"test/"
                f"{self.__class__.__name__}/"
                f"{dataset_name}/"
                f"step_{trainer.global_step}"
                f"_{batch_idx}_rank{trainer.global_rank}.png",
            )

    def log_image(
        self,
        trainer: L.Trainer,
        img: Image.Image,
        key: str,
        artifact_file: str | None = None,
    ):
        for logger in trainer.loggers:
            if isinstance(logger, loggers.WandbLogger):
                logger.log_image(key=f"{self.category}/{key}", images=[img], step=trainer.global_step)
            elif isinstance(logger, loggers.TensorBoardLogger):
                # do whatever the tensorboard logger supports
                pass
            elif isinstance(logger, loggers.MLFlowLogger):
                # do whatever the mlflow logger supports
                # Note that the mlflow_logger is essentially an instance of mlflow.Client
                # https://github.com/Lightning-AI/pytorch-lightning/issues/3964#issuecomment-705348121
                # See https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html#mlflow.client.MlflowClient.log_image
                # for API
                # example usage:
                logger.experiment.log_image(
                    run_id=logger.run_id,
                    image=img,
                    # key=key,
                    artifact_file=artifact_file,
                    # step=trainer.global_step,
                )
                pass
