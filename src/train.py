#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import os
from pathlib import Path
from typing import Any

import hydra
import lightning as L
import rootutils
import torch
import torch._dynamo
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# ------------------------------------------------------------------------------------ #
# the setup_root above is equivalent to:
# - adding project root dir to PYTHONPATH
#       (so you don't need to force user to install project as a package)
#       (necessary before importing any local modules e.g. `from src import utils`)
# - setting up PROJECT_ROOT environment variable
#       (which is used as a base for paths in "configs/paths/default.yaml")
#       (this way all filepaths are the same no matter where you run the code)
# - loading environment variables from ".env" in root dir
#
# you can remove it if you:
# 1. either install project as a package or move entry files to project root dir
# 2. set `root_dir` to "." in "configs/paths/default.yaml"
#
# more info: https://github.com/ashleve/rootutils
# ------------------------------------------------------------------------------------ #

from src.utils import (  # noqa: E402
    RankedLogger,
    extras,
    # get_metric_value,
    instantiate_callbacks,
    instantiate_loggers,
    log_hyperparameters,
    task_wrapper,
)

log = RankedLogger(__name__, rank_zero_only=True)


@task_wrapper
def train(cfg: DictConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    This method is wrapped in optional @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """
    if cfg.accelerate.dynamo_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = cfg.accelerate.dynamo_cache_size_limit

    # https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    torch.set_float32_matmul_precision(cfg.accelerate.fp32_matmul_precision)

    # https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-and-later-devices
    # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
    torch.backends.cuda.matmul.allow_tf32 = cfg.accelerate.fp32_matmul_precision != "highest"
    # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
    torch.backends.cudnn.allow_tf32 = True

    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.seed is not None:
        L.seed_everything(cfg.seed, workers=True)

    # pass the whole config to the datamodule and plmodule
    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)(cfg=cfg)

    log.info(f"Instantiating model <{cfg.plmodule._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.plmodule)(cfg=cfg)

    log.info("Instantiating callbacks...")
    callbacks: list[Callback] = instantiate_callbacks(cfg.callbacks)

    log.info("Instantiating loggers...")
    logger: list[Logger] = instantiate_loggers(cfg.logger)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    if cfg.train:
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    else:
        trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger, max_steps=0)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    """
    Based on "restore ckpts" and "train", there are following cases:
    - restore ckpts is None, train = False: just eval random initialized model
    - restore ckpts is None, train = True: normal train, valid, and test
    - restore ckpts is not None, train = False: valid and test for ckpts
    - restore ckpts is not None, train = True: for each ckpts: load, train, valid, and test
    """

    if cfg.paths.get("restore_dir") is None and cfg.paths.get("restore_ckpts") is None:
        # no restore ckpts, train model from scratch
        ckpt_paths = [None]
    elif cfg.paths.get("restore_ckpts") is not None:  # try loading restore_ckpts first
        ckpt_paths = [Path(ckpt) for ckpt in cfg.paths.restore_ckpts]
    else:  # if restore_ckpts is None, load all ckpts in restore_dir
        ckpt_paths = Path(cfg.paths.restore_dir).glob("*.ckpt")
        # TODO: drop last.ckpt

    ckpt_paths = list(ckpt_paths)
    for ckpt_path in ckpt_paths:
        log.info(f"Found checkpoint: {ckpt_path}")

    if cfg.train:
        log.info("Entering training mode...")
        for ckpt_path in ckpt_paths:
            log.info(f"Fitting model with ckpt_path: {ckpt_path}")
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
    else:
        log.info("Entering validation mode...")
        # prepare dataloaders to avoid dataloader re-init
        datamodule.prepare_data()  # prepare data (e.g. download, extract, etc.)
        datamodule.setup(stage="validate")  # setup data on each process
        train_dataloader = datamodule.train_dataloader()
        valid_dataloaders = datamodule.val_dataloader()
        log.info("DataModule prepared and set up for validation.")
        for ckpt_path in ckpt_paths:
            log.info(f"Evaluating model with ckpt_path: {ckpt_path}")
            trainer.fit(
                model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=valid_dataloaders,
                ckpt_path=ckpt_path,
            )
            log.info("Running trainer.validate() for validation.")
            trainer.validate(model=model, dataloaders=valid_dataloaders, ckpt_path=ckpt_path)
    return {}, {}


# if train_custom.yaml exists, use it as default config file
# otherwise, need to specify config file in command line
config_file_name = "train_custom.yaml" if os.path.exists("./configs/train_custom.yaml") else None

# Register eval resolver for mathematical expressions in config
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.3", config_path="../configs/", config_name=config_file_name)
def main(cfg: DictConfig):
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    # apply extra utilities
    # (e.g. ask for tags if none are provided in cfg, print cfg tree, etc.)
    extras(cfg)

    # train the model
    train(cfg)
    # metric_dict, _ = train(cfg)

    # # safely retrieve metric value for hydra-based hyperparameter optimization
    # metric_value = get_metric_value(metric_dict=metric_dict, metric_name=cfg.get("optimized_metric"))

    # # return optimized metric
    # return metric_value


if __name__ == "__main__":
    main()
