---
name: workflow
description: Full workflow: environment setup, running experiments, reusing and implementing new components (models, datasets, plmodules, callbacks), and testing
---

# Workflow

This document describes the full project workflow: environment setup, running experiments, reusing and implementing new components, and testing.

## Environment Setup

### uv (recommended)

```sh
# consider adding "--index-url https://pypi.tuna.tsinghua.edu.cn/simple" if you have difficulty connecting to pypi.org
uv sync --extra cu118  # torch-cu118
uv sync --extra cu124  # torch-cu124
uv sync --extra cu126  # torch-cu126 (suggested)
uv sync --extra cu128  # torch-cu128
uv sync --extra cpu    # torch-cpu
```

You can also put the above commands in your scripts so that the environment is activated and synced before each training. See examples in `scripts_core`.

### Conda (alternative)

```sh
conda create -n core python=3.11 -y && conda activate core  # you can replace core with other names
```

We use pip for package installation:

```sh
pip install -r requirements/requirements-icon-core-cuda118.txt  # for cuda 11.8
pip install -r requirements/requirements-icon-core-cuda124.txt  # for cuda 12.4
```

### Pre-commit hooks

Pre-commit hooks check code on every commit: enforcing formatting and avoiding mistakes like uploading private keys. If checks fail, hooks will try to auto-fix — amend the changes and commit again. If auto-fix doesn't work, manually adjust according to the error message.

**Install once per local clone** (strongly suggested before your first commit):

```sh
# uv
uv run pre-commit install                                                        # HTTPS
uv run pre-commit install --config requirements/.pre-commit-config-ssh.yaml      # SSH

# conda (if not using uv)
conda activate your_env
pip install pre-commit  # skip if already installed
pre-commit install                                                               # HTTPS
pre-commit install --config requirements/.pre-commit-config-ssh.yaml             # SSH
```

Run on all files manually:

```sh
pre-commit run --all-files
```

Pre-commit hooks are also integrated in GitHub CI workflows (`.github/workflows`). To disable, delete that folder and run `uv run pre-commit uninstall` (or `pre-commit uninstall`).

## Running

### Example scripts

```sh
sh scripts_core/cpu.sh
```

### Run your project

```sh
uv run python src/train.py --config-name=train_your_project
```

### Machine-specific config

Machine-specific overrides can go in `configs/train_custom.yaml` (git-ignored):

```yaml
defaults:
  - train_your_project
  - _self_

paths:
  data_dir: ./project_data/
  log_dir: ./project_logs/
```

If `configs/train_custom.yaml` exists, `src/train.py` reads it by default:

```sh
uv run python src/train.py                      # uses train_custom.yaml
uv run python src/train.py trainer.max_steps=10  # with overrides
```

`configs/train_custom.yaml` is git-ignored, so it is only effective on your machine. All configs will be logged (including those in `configs/train_custom.yaml`), so reproducibility is preserved. For better collaboration, only include insignificant machine-specific configs (e.g. paths) in this file.

## Reusing Existing Components

This repository already contains a lot of components, including models, lightning modules, datasets, callbacks, etc. You should try to leverage them by reusing or inheriting from them in your project. Additionally, they serve as practical implementation references. Read them before implementing your own.

Do not modify files inherited from the core repository. These files start with a header in the beginning of the file:

```python
#######################################################
# This file belongs to the core repository.
# If your project repository is a fork of core,
# you are suggested to keep this file untouched in your project.
# This helps avoid merge conflicts when syncing from core.
#######################################################
```

Other files are not part of the core repository and are free to modify. Do not add the header in your newly created files, so other developers can easily identify them.

## Implementing New Components
When implementing new components, you generally need to implement both .py files and corresponding .yaml configuration files.

### Model

- Create new model architectures in `src/models/your_model_folder/your_model_file.py`. Use subdirectories to organize models.
- Create corresponding configuration files in `configs/model/`.
- Avoid setting default initialization arguments in model classes.

### Lightning Module

- Create new lightning modules for training and evaluation in `src/plmodules/your_plmodule_file.py`.
- Create corresponding configuration files in `configs/plmodule/`.
- Must implement `get_trainable_networks()` (abstract method from `BaseLitModule`) to specify which network(s) should be optimized.

### Dataset

- Create new torch datasets in `src/datasets/your_dataset_folder/your_dataset_file.py`.
- Create corresponding configuration files in `configs/data/your_dataset_folder/`. In this folder, create `train/`, `valid/`, and `test/` subfolders. For each training dataset, you should create a corresponding yaml file in `train/` subfolder. You can have multiple yaml files for multiple training datasets. Similarly for validation and testing datasets. In the end, you should also create a main dataset yaml file `configs/data/your_dataset_folder/your_dataset.yaml` to list all training, validation, and testing datasets to be used in the project.

### DataModule

- Create new lightning data modules and dataloaders in `src/datamodules/your_datamodule_file.py`.
- Create corresponding configuration files in `configs/datamodule/`.

### Callbacks

- Create new callbacks in `src/callbacks/your_callback_name.py`.
- Create corresponding configuration files in `configs/callbacks/`. See the `configs` skill for details on callback config conventions.
- For visualization callbacks, inherit from `Viz` (`src/callbacks/viz.py`) and override `get_image()` rather than implementing from scratch.

### Main Configuration

- Create a new configuration file `configs/train_project_name.yaml` as the main configuration file for the project. This file will indicate what configurations to be used in the project, including models, datasets, callbacks, etc.

### Running Scripts

- Create new scripts in `scripts_project_name/` directory. Please see the scripts in `scripts_core/` as references.

## Example Workflow

Let's take the `nop_rollout` (neural operator with rollout) project as an example. This project demonstrates how to implement a complete neural operator training pipeline with rollout validation.

### Model

We use 1D FNO model, which is implemented in `src/models/nop/fno.py`. The corresponding configuration is in `configs/model/fno1d.yaml`.

### Lightning Module

The training and validation logic is implemented in `src/plmodules/nop_rollout_lit_module.py`, which inherits from `BaseLitModule`. The configuration is in `configs/plmodule/nop_rollout.yaml`.

### Dataset

We use the Kuramoto-Shivashinsky (KS) equation simulation dataset. The dataset is implemented in `src/datasets/ks/ks.py`. The corresponding configurations are in `configs/data/ks/` folder. Note that we put the training, validation, and testing configurations in the `train/`, `valid/`, and `test/` subfolders, and create a main dataset yaml file `configs/data/ks/ks.yaml` to list all training, validation, and testing datasets to be used in the project.

In this project, we use two validation datasets: `ks_short` and `ks_long`, for validating the model's performance on short-term and long-term predictions, respectively. Therefore we have two corresponding validation configurations in `configs/data/ks/valid/` folder, and listed them in the main dataset yaml file `configs/data/ks/ks.yaml`.

### DataModule

We don't need new dataloader logic for this project, so we just reuse the existing `BaseDataModule` class.

### Callbacks

We implement two callbacks in `src/callbacks/viz_rollout_error.py ` and `src/callbacks/viz_rollout_1d.py` to visualize the rollout error and the rollout trajectory, respectively. Notably, we inherit from the `Viz` class in `src/callbacks/viz.py` for visualization.

We created two corresponding configuration files in `configs/callbacks/` folder: `viz_rollout_error.yaml` and `viz_rollout_1d.yaml`, and listed them in the main callback yaml file `configs/callbacks/many_callbacks_nop_rollout.yaml`, together with other existing callback configurations.

### Main Configuration

The main configuration file is `configs/train_nop_rollout.yaml`.

### Running Scripts

We didn't implement the running script, but here are two simple examples:

- CPU short training for debugging
```bash
#!/bin/sh
uv sync --extra cpu
export TORCH_COMPILE_DISABLE=1
uv run python src/train.py --config-name=train_nop_rollout trainer=cpu trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5
echo "Done"
```

- GPU full training
```bash
uv sync --extra cu126
uv run python src/train.py --config-name=train_nop_rollout
echo "Done"
```

## Modifying Project Components

Whenever you modify project components (those without the header in the beginning of the file), especially if you change their initialization parameters, be sure to update the relevant configuration files accordingly.

## Testing

### Unit Tests

You can add unit tests code in the same file as the code to be tested, under the `if __name__ == "__main__":` block. Alternatively, you can add individual test files `test_your_component.py` for complicated tests, but it is suggested to keep the test code in the same folder as the code to be tested.

You may have difficulty in importing local modules. An easy way is to run the test code in the root directory as follows:
```bash
uv sync --extra cpu # for cpu tests
# uv sync --extra cu126 # for gpu tests
uv run python -m src.subfolder.test_your_component
```
Here `uv run` ensures running in the virtual environment.

### End-to-End Tests

Refer to the scripts in `scripts_core/` for how to run code end-to-end.
