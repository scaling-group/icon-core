---
name: configs
description: Hydra configuration file structure and conventions for models, datasets, callbacks, and training
---

# Configuration Files

All configuration files live in `configs/`, organized using the Hydra framework.

## Directory Structure

- `accelerate/` — Distributed training acceleration
- `callbacks/` — Lightning callbacks (e.g., model summary, checkpointing)
- `data/` — Dataset and dataloader configurations
- `datamodule/` — PyTorch Lightning datamodule configurations
- `experiment/` — Experiment-specific overrides
- `extras/` — Additional utility configurations
- `hydra/` — Hydra framework configurations
- `logger/` — Logging configurations (e.g., TensorBoard, WandB)
- `loss/` — Loss function configurations
- `model/` — Model configurations
- `opt/` — Optimizer configurations
- `paths/` — Path configurations
- `plmodule/` — PyTorch Lightning module configurations
- `trainer/` — Trainer configurations

Top-level training config files (e.g., `train_nop.yaml`, `train_nop_rollout.yaml`) specify which configs to use for a given project.

## Callback Configuration

There are two types of callback config files in `configs/callbacks/`:

- `single_callback_name.yaml` — Configuration for a single callback.
- `many_callbacks_project_name.yaml` — Lists all callback configs to be used in a project.

For a new project, create a `many_callbacks_project_name.yaml` listing the individual callback configs. If you add a new callback, also create a corresponding `single_callback_name.yaml`.
