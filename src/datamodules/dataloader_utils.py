#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import torch

# We follow the practice in
# https://pytorch.org/docs/stable/notes/randomness.html#dataloader
# to set the generator for the dataloader


def get_dataloader_rng(
    base_seed: int,
    enable_device_seed: bool,
    print_info: str,
    print_lv: int,
) -> torch.Generator:
    """
    Get the RNG for the dataloader.

    Args:
        base_seed (int): The mandatory base seed for the dataloader.
                         If you want a dynamic seed (which is not recommended), set it out of this function.
        enable_device_seed (bool): whether to use per-device seed.
                        If True, augment the seed with the device rank.
        print_info (str): the info to print.
        print_lv (int): the verbosity level.

    Returns:
        torch.Generator: The RNG for the dataloader.
    """
    generator = torch.Generator()
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    seed = base_seed + rank if enable_device_seed else base_seed
    generator.manual_seed(seed)
    if print_lv >= 2:
        print(f"dataloader rng, rank=[0x{rank:04x}]\tseed=[0x{seed:016x}]\t({print_info})", flush=True)
    return generator


class CycleLoader:
    """
    This class takes a list of dictionaries containing dataloader and sampler instances,
    and creates an iterator that cycles through them sequentially:
    step 1: dataloader 1
    step 2: dataloader 2
    step 3: dataloader 3
    step 4: dataloader 1
    step 5: dataloader 2
    step 6: dataloader 3
    ...
    When one dataloader is exhausted, it is reset and the cycle continues.

    This CycleLoader should never raise StopIteration. Therefore you can also wrap a single DataLoader
    with this class to create an infinite iterator.

    Attributes:
        loaders (list): A list of dictionaries containing dataloader, sampler, epoch, and iterator information.
        see __init__ for more details.

    Methods:
        __init__(loaders): Initializes with a list of dictionaries containing dataloaders and samplers
        __iter__(): Initializes iterators for each dataloader and returns self
        __next__(): Returns the next batch from the current dataloader, cycling through them
                    indefinitely. Resets exhausted dataloaders automatically.
    """

    def __init__(self, loaders: list[dict]):
        """
        Initialize the CycleLoader with a list of dictionaries containing dataloaders and samplers.

        Each dictionary in the list should have the following structure:
        {
            "dataloader": DataLoader | CycleLoader, can be mixed in the same list
            "sampler": DistributedSampler | None,
            "epoch": int,  # will be added to dict in __init__
            "iterator": Iterator | None  # will be added to dict in __init__
        }

        If a dataloader is using DistributedSampler,
        we must explicitly provide the DistributedSampler to the __init__ function and manually set the epoch.
        This is necessary since DistributedSampler requires manual epoch management. See example in
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        Lightning's automatic epoch handling is bypassed when dataloaders are wrapped in CycleLoader.

        Set sampler = None for other elements, including:
        - DataLoaders with other types of samplers, as they do not need manual epoch management.
        - CycleLoaders, as they should be able to handle the epoch management by themselves.
        """
        self.loaders = loaders
        # Initialize epoch count and iterator for each loader
        for loader in self.loaders:
            loader["epoch"] = 0
            loader["iterator"] = None

    def iter_loader(self, loader: dict):
        if loader["sampler"] is not None:
            loader["sampler"].set_epoch(loader["epoch"])
        loader["epoch"] += 1
        loader["iterator"] = iter(loader["dataloader"])

    def __iter__(self):
        # Keep an active iterator per sub-loader
        # at the beginning of each epoch before creating the DataLoader iterator
        for loader in self.loaders:
            self.iter_loader(loader)

        self.idx = 0  # which loader we're pulling from
        return self

    def __next__(self):
        try:
            # Attempt to get a batch from the current loader
            current_loader = self.loaders[self.idx]
            batch = next(current_loader["iterator"])  # raises StopIteration if the iterator is exhausted
            # Move to the next loader
            self.idx = (self.idx + 1) % len(self.loaders)
            return batch
        except StopIteration:
            # Current loader is exhausted; reset its iterator
            # at the beginning of each epoch before creating the DataLoader iterator
            current_loader = self.loaders[self.idx]
            self.iter_loader(current_loader)

            # Try again from the newly-reset iterator at the same index
            batch = next(current_loader["iterator"])
            # Here we didn't use recursive call to avoid infinite loop
            # If StopIteration is raised again, it means the dataloader is not enough for one batch
            # In this case, we will raise StopIteration
            self.idx = (self.idx + 1) % len(self.loaders)
            return batch


class WeightedLoader:
    """
    This class takes a list of dictionaries containing dataloader, sampler, and weight instances,
    and creates an iterator that cycles through them based on weighted probability sampling:

    Each step samples a dataloader based on its relative weight probability.
    When one dataloader is exhausted, it is reset and the sampling continues.

    This WeightedLoader should never raise StopIteration. Therefore you can also wrap a single DataLoader
    with this class to create an infinite iterator.

    Attributes:
        loaders (list): A list of dictionaries containing dataloader, sampler, weight, epoch, and iterator information.
        weights (torch.Tensor): Normalized probabilities for sampling each dataloader.
        see __init__ for more details.

    Methods:
        __init__(loaders): Initializes with a list of dictionaries containing dataloaders, samplers, and weights
        __iter__(): Initializes iterators for each dataloader and returns self
        __next__(): Returns the next batch from a probabilistically selected dataloader.
                    Resets exhausted dataloaders automatically.
    """

    def __init__(self, loaders: list[dict]):
        """
        Initialize the WeightedLoader with a list of dictionaries containing dataloaders, samplers, and weights.

        Each dictionary in the list should have the following structure:
        {
            "dataloader": DataLoader | CycleLoader, can be mixed in the same list
            "sampler": DistributedSampler | None,
            "weight": float,  # sampling weight for this dataloader
            "epoch": int,  # will be added to dict in __init__
            "iterator": Iterator | None  # will be added to dict in __init__
        }

        If a dataloader is using DistributedSampler,
        we must explicitly provide the DistributedSampler to the __init__ function and manually set the epoch.
        This is necessary since DistributedSampler requires manual epoch management. See example in
        https://pytorch.org/docs/stable/data.html#torch.utils.data.distributed.DistributedSampler
        Lightning's automatic epoch handling is bypassed when dataloaders are wrapped in WeightedLoader.

        Set sampler = None for other elements, including:
        - DataLoaders with other types of samplers, as they do not need manual epoch management.
        - CycleLoaders, as they should be able to handle the epoch management by themselves.
        """
        self.loaders = loaders

        # Extract and normalize weights
        weights = torch.tensor([loader.get("weight") for loader in self.loaders], dtype=torch.float32)
        self.weights = weights / weights.sum()  # Normalize to probabilities

        # Initialize epoch count and iterator for each loader
        for loader in self.loaders:
            loader["epoch"] = 0
            loader["iterator"] = None

    def iter_loader(self, loader: dict):
        if loader["sampler"] is not None:
            loader["sampler"].set_epoch(loader["epoch"])
        loader["epoch"] += 1
        loader["iterator"] = iter(loader["dataloader"])

    def __iter__(self):
        # Keep an active iterator per sub-loader
        # at the beginning of each epoch before creating the DataLoader iterator
        for loader in self.loaders:
            self.iter_loader(loader)

        return self

    def __next__(self):
        # Sample a loader index based on weights
        idx = torch.multinomial(self.weights, 1).item()

        try:
            # Attempt to get a batch from the selected loader
            current_loader = self.loaders[idx]
            batch = next(current_loader["iterator"])  # raises StopIteration if the iterator is exhausted

            # Add the actual dataloader index to the batch
            batch["_dataloader_idx"] = idx

            return batch
        except StopIteration:
            # Current loader is exhausted; reset its iterator
            # at the beginning of each epoch before creating the DataLoader iterator
            current_loader = self.loaders[idx]
            self.iter_loader(current_loader)

            # Try again from the newly-reset iterator
            batch = next(current_loader["iterator"])
            # Here we didn't use recursive call to avoid infinite loop
            # If StopIteration is raised again, it means the dataloader is not enough for one batch
            # In this case, we will raise StopIteration

            # Add the actual dataloader index to the batch
            batch["_dataloader_idx"] = idx

            return batch
