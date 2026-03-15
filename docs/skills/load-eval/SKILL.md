---
name: load-eval
description: How to load a checkpoint and run evaluation (inference only, no training)
---

# Load and Eval

To evaluate a trained model without training, pass `train=False` to `src/train.py` along with a path to the checkpoint(s).

## Option 1: Load a specific checkpoint

```sh
uv run python src/train.py --config-name=train_your_project \
    train=False \
    paths.restore_ckpts=["./logs/train/runs/2025-01-01_00-00-00/checkpoints/last.ckpt"]
```

## Option 2: Load all checkpoints in a directory

```sh
uv run python src/train.py --config-name=train_your_project \
    train=False \
    paths.restore_dir=./logs/train/runs/2025-01-01_00-00-00/checkpoints
```

All `.ckpt` files in the directory will be evaluated in sequence.

## How it works

- `train=False` skips the training loop and goes straight to validation.
- `paths.restore_ckpts` takes precedence over `paths.restore_dir` if both are set.
- If neither is set, the model runs with random weights (useful for sanity checks and training with random initialization)
- Results are saved to the same `logs/` structure as a training run (metrics, visualizations, etc.).

## Defaults

Both `restore_ckpts` and `restore_dir` default to `null` (defined in `configs/paths/default.yaml`). Override them on the command line as shown above, or set them in `configs/train_your_project.yaml` for repeated use:

```yaml
train: False
paths:
  restore_dir: ./logs/train/runs/2025-01-01_00-00-00/checkpoints
```
