#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################

import hydra
from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.datamodules.base_datamodule import BaseDataModule


class ProcessDatasetWrapper(Dataset):
    """
    Wrapper dataset that applies post-processing to individual samples.
    This allows the processing to run in parallel with DataLoader workers.
    """

    def __init__(self, dataset, image_processor_cfg):
        self.dataset = dataset
        self.image_processor = hydra.utils.instantiate(image_processor_cfg)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Process images in the worker process
        for example in item["examples"]:
            example["processed_images"] = self.process_image(example["raw_images"])

        return item

    def process_image(self, image):
        """
        Process image with image_processor
        image: [..., c, h, w], usually uint between 0 and 255
        return [..., c, h', w'], usually float in [0, 1] or [-1, 1]
        """
        reshaped_image = image.reshape(-1, *image.shape[-3:])  # (..., c, h, w) -> (n, c, h, w)
        processed_image = self.image_processor(reshaped_image, return_tensors="pt")["pixel_values"]
        # (n, c, h', w') -> (..., c, h', w')
        processed_image = processed_image.reshape(image.shape[:-3] + processed_image.shape[-3:])
        return processed_image


class ControlDataModule(BaseDataModule):
    """
    ControlDataModule that moves image processing to individual workers for parallel execution.

    ImageProcessor always returns tensors on CPUs, even if the input is on GPUs.
    Therefore, we put the image processing in the DataModule instead of the models.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        # for example, put the following in the cfg.data yaml file
        # image_processor:
        #   _target_: transformers.AutoImageProcessor.from_pretrained
        #   pretrained_model_name_or_path: "google/vit-base-patch16-224"
        self.image_processor_cfg = self.cfg.data.image_processor

    def get_train_dataset_from_cfg(self, cfg):
        """
        Override to wrap dataset with processing wrapper
        """
        base_result = super().get_train_dataset_from_cfg(cfg)
        dataset = base_result["dataset"]

        # Wrap with processing dataset
        processed_dataset = ProcessDatasetWrapper(dataset, self.image_processor_cfg)

        return {"dataset": processed_dataset, "cfg": cfg}

    def get_valid_test_dataset_from_cfg(self, cfg):
        """
        Override to wrap dataset with processing wrapper
        """
        base_result = super().get_valid_test_dataset_from_cfg(cfg)
        dataset = base_result["dataset"]

        # Wrap with processing dataset
        processed_dataset = ProcessDatasetWrapper(dataset, self.image_processor_cfg)

        return {"dataset": processed_dataset, "cfg": cfg}
