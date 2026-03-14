#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import hydra
import torch
from lightning import LightningDataModule
from omegaconf import DictConfig
from torch.utils.data import DataLoader, DistributedSampler

from src.datasets import pytree_utils as ptu

from . import dataloader_utils as dlu


class BaseDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.cfg = cfg

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading and caching logic within.
        In case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        # <- call the data generation script here
        print("Preparing data...")

        for i, (key, cfg) in enumerate(self.cfg.data.train.items()):
            print(f"train dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"\t{k}: {v}")
            self.get_train_dataset_from_cfg(cfg)

        for i, (key, cfg) in enumerate(self.cfg.data.valid.items()):
            print(f"valid dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"\t{k}: {v}")
            self.get_valid_test_dataset_from_cfg(cfg)

        for i, (key, cfg) in enumerate(self.cfg.data.test.items()):
            print(f"test dataloader #{i}: {key}")
            for k, v in cfg.items():
                print(f"\t{k}: {v}")
            self.get_valid_test_dataset_from_cfg(cfg)

    def setup(self, stage: str | None = None) -> None:
        """
        called on each process on GPU
        """
        self.train_datasets = [self.get_train_dataset_from_cfg(cfg) for cfg in self.cfg.data.train.values()]
        self.valid_datasets = [self.get_valid_test_dataset_from_cfg(cfg) for cfg in self.cfg.data.valid.values()]
        self.test_datasets = [self.get_valid_test_dataset_from_cfg(cfg) for cfg in self.cfg.data.test.values()]

    def get_train_dataset_from_cfg(self, cfg):
        """
        return dataset(s) from a given config
        you can override this function to customize the dataset
        even return multiple datasets!
        """
        dataset = hydra.utils.instantiate(cfg.dataset)
        return {"dataset": dataset, "cfg": cfg}

    def get_valid_test_dataset_from_cfg(self, cfg):
        """
        return a dataset from a given config
        you can define your own dataset class and override this function
        """
        dataset = hydra.utils.instantiate(cfg.dataset)
        return {"dataset": dataset, "cfg": cfg}

    def get_train_collate_fn(self, cfg):
        """
        you can override this function to customize the collate function for training
        """
        return ptu.concat

    def get_valid_test_collate_fn(self, cfg):
        """
        you can override this function to customize the collate function for validation/test
        """
        return ptu.concat

    def get_train_dataloader(self, dataset, cfg):
        """
        wrap a dataset with a DataLoader for training
        in most cases, you don't need to override this function
        Args:
            dataset: a dataset object
            cfg: a config for the dataset (also with dataloader config)
        """
        # generator will only be created once in the very begining when global_step = 0
        # In our implementation, we can have different random states during the whole training process
        # what happens when we call iter(dataloader)?
        # according to https://discuss.pytorch.org/t/dataloader-persistent-workers-usage/189329
        # if not using persistent workers (by default), new workers will be created when dataloader.__iter__() is called
        # and the generator is used to initialize the worker seeds (which is different from last __iter__() call)
        # setting torch seeds inside worker_init_fn() did not work as expected in our practice.
        # we never tested persistent workers, so we don't know if it works.

        generator = dlu.get_dataloader_rng(
            base_seed=cfg.base_seed,
            enable_device_seed=cfg.enable_device_seed,
            # print_info=f"step = {self.trainer.global_step}, train: {cfg.name}",
            print_info=f"train: {cfg.name}",
            print_lv=self.cfg.print_lv,
        )

        common_kwargs = {
            "batch_size": cfg.batch_size_per_device,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "collate_fn": self.get_train_collate_fn(cfg),
            "generator": generator,
        }

        if torch.distributed.is_initialized():
            # if distributed, use DistributedSampler
            # we will wrap the dataloader in CycleLoader,
            # therefore lightning cannot automatically handle DistributedSampler and epoch management
            # use cfg.base_seed as seed. DistributedSampler will add epochs to the seed when __iter__() is called
            sampler = DistributedSampler(dataset=dataset, shuffle=True, seed=cfg.base_seed, drop_last=True)
            dataloader = DataLoader(dataset=dataset, sampler=sampler, **common_kwargs)
            return {"dataloader": dataloader, "sampler": sampler}
        else:
            # if not distributed, a plain sampler will suffice
            dataloader = DataLoader(dataset=dataset, shuffle=True, drop_last=True, **common_kwargs)
            return {"dataloader": dataloader, "sampler": None}

    def get_valid_test_dataloader(self, dataset, cfg):
        """
        wrap a dataset with a DataLoader for validation/test
        in most cases, you don't need to override this function
        Args:
            dataset: a dataset object
            cfg: a config for the dataset (also with dataloader config)
        """

        generator = dlu.get_dataloader_rng(
            base_seed=cfg.base_seed,
            enable_device_seed=cfg.enable_device_seed,
            # print_info=f"step = {self.trainer.global_step}, valid|test: {cfg.name}",
            print_info=f"valid|test: {cfg.name}",
            print_lv=self.cfg.print_lv,
        )

        # generator will only be created once in the very begining when global_step = 0
        # but lightning can somehow use the generator to generate the same random states across validation epochs
        # Don't pass a worker_init_fn into valid_test_dataloader!
        # we found that even pass "lambda worker_id: None" into worker_init_fn
        # will cause different random states across validation epochs.
        # we never tested persistent workers, so we don't know if it works.

        common_kwargs = {
            "batch_size": cfg.batch_size_per_device,
            "num_workers": cfg.num_workers,
            "pin_memory": cfg.pin_memory,
            "collate_fn": self.get_valid_test_collate_fn(cfg),
            "generator": generator,
        }

        # We can rely on Lightning's built-in handling of DistributedSampler for validation/test
        return DataLoader(
            dataset=dataset,
            shuffle=False,  # careful: different from train_dataloader
            drop_last=False,  # careful: different from train_dataloader
            **common_kwargs,
        )

    def train_dataloader(self):
        """
        return a single cycle dataloader
        """
        dataloaders = [self.get_train_dataloader(**ds) for ds in self.train_datasets]

        # Check sampling mode configuration
        sampling_mode = self.cfg.data.sampling_mode

        if sampling_mode == "weighted":
            # Add weights to dataloader dictionaries
            for i, dataloader_dict in enumerate(dataloaders):
                dataset_cfg = self.train_datasets[i]["cfg"]
                # raise error if weight is not found
                if "weight" not in dataset_cfg:
                    raise ValueError(f"Weight not found in dataset config: {dataset_cfg}")
                weight = dataset_cfg.weight
                dataloader_dict["weight"] = weight
            return dlu.WeightedLoader(dataloaders)
        elif sampling_mode == "cycle":
            # Default to cycle sampling
            return dlu.CycleLoader(dataloaders)
        else:
            raise ValueError(f"Unknown sampling_mode: {sampling_mode}. Expected 'weighted' or 'cycle'.")

    def val_dataloader(self):
        """
        return a list of dataloaders for separate validation
        """
        dataloaders = [self.get_valid_test_dataloader(**ds) for ds in self.valid_datasets]
        return dataloaders  # don't wrap with CycleLoader

    def test_dataloader(self):
        """
        return a list of dataloaders for separate test
        """
        dataloaders = [self.get_valid_test_dataloader(**ds) for ds in self.test_datasets]
        return dataloaders  # don't wrap with CycleLoader

    def teardown(self, stage=None):
        pass
