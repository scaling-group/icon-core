#!/bin/sh
uv sync --extra cu126
uv tree

# Here we set a small max_steps, val_check_interval, and limit_val_batches for fast testing. For full training, you can remove these arguments.
uv run python src/train.py --config-name=train_vicon trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5

# load and eval (replace the folders with your own):
# uv run python src/train.py --config-name=train_vicon train=False paths.restore_dir=./logs/train/runs/2025-01-01_00-00-00/checkpoints
# uv run python src/train.py --config-name=train_vicon train=False paths.restore_ckpts=["./logs/train/runs/2025-01-01_00-00-00/checkpoints/last.ckpt"]
# see more details in src/train.py and configs/paths/default.yaml

echo "Done"
