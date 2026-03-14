---
name: logs-structure
description: Training logs directory structure for post-processing and analysis
---

# Logs Directory Structure Documentation

## Overview
The `logs/` directory contains training run outputs organized by timestamp-based folder names.

## Directory Structure

### Top Level: `logs/xxx/runs/`
- **Folder naming**: `YYYY-MM-DD_HH-MM-SS-MICROSECONDS` (e.g., `2025-07-12_18-11-28-389603`)
- **Purpose**: Each folder represents a single training run with its timestamp

### Run Directory Contents

#### 1. Configuration and Metadata
- **`config_tree.log`**: Complete configuration tree for the run
- **`wandb/latest-run/files/wandb-metadata.json`**: System metadata, git info, and run parameters

#### 2. Metrics Directory: `metric/valid/`
- **Structure**: `metric/valid/step_XXX/DATASET_NAME/`
- **Files per dataset step**:
  - `description_rankX.txt`: Sample descriptions for each GPU rank
  - `METRIC_NAME_rankX.csv`: Sample-wise metrics of METRIC_NAME for each GPU rank
- **Multi-GPU handling**: Files with `_rankX` suffix should be combined across all ranks
- **Content**: Each line in CSV files represents metrics for one sample

#### 3. Batch Information: `batch_info/`
- **Structure**: Similar to metrics - `batch_info/valid/step_XXX/DATASET_NAME/`
- **Files**: `rank_X.txt` containing detailed batch information and sample descriptions
- **Purpose**: Provides context for understanding what data was processed

#### 4. Checkpoints: `checkpoints/`
- **Files**: `step_X.ckpt`, `last.ckpt`
- **Purpose**: Model state at different training steps

#### 5. Visualizations: `viz/`
- **Structure**: `viz/valid/step_XXX/DATASET_NAME/`
- **Files**: `X_rankY.png` - visualization outputs per rank
- **Purpose**: Visual outputs for debugging and analysis

#### 6. Logs
- **`train.log`**: Training process logs
- **`tags.log`**: Tag information
- **`wandb/`**: Weights & Biases experiment tracking files

#### 7. CSV Outputs: `csv/`
- **`hparams.yaml`**: Hyperparameters
- **`metrics.csv`**: Aggregated metrics over time

## Key Points for Analysis

### Multi-GPU Considerations
- All files with `_rankX` suffixes need to be combined for complete metrics
- Each rank processes different data shards

### Sample-Level Metrics
- Each line in metric CSV files = one sample
- Sample descriptions in `description_rankX.txt` provide context
- Load and process raw metrics in "metric" folder instead of using the aggregated metrics in "csv" folder.

### Step-Based Organization
- `step_XXX` folders correspond to validation checkpoints
- Metrics collected at specific training intervals

### Dataset Organization
- Each dataset has its own subfolder
- Multiple datasets can be evaluated at each step

### Finding Run Details
1. Check `config_tree.log` for training configuration
2. Check `wandb/latest-run/files/wandb-metadata.json` for system info and git commit
3. Check `batch_info/` for data structure details
