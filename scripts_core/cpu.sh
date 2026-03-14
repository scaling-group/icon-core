#!/bin/sh
uv sync --extra cpu
uv tree

# There's a C++ compilation error when trying to use PyTorch's inductor on macOS.
# This is a known issue with PyTorch's inductor on Apple Silicon Macs.

export TORCH_COMPILE_DISABLE=1

uv run python src/train.py --config-name=train_vicon trainer=cpu trainer.max_steps=10 trainer.val_check_interval=5 trainer.limit_val_batches=5

echo "Done"
