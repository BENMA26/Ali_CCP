# Ali_CCP: Alibaba Click and Conversion Prediction

## Description

A deep learning pipeline for **Click-Through Rate (CTR)** and **Conversion Rate (CVR)** prediction on the [Alibaba CCP dataset](https://tianchi.aliyun.com/dataset/408), built with PyTorch.

This project implements a **two-tower retrieval architecture** with multi-task learning, covering the full recommendation pipeline from candidate retrieval to ranking:

- 🔍 **Recall Stage** — Dual-encoder two-tower model with negative sampling (random, popularity-weighted, in-batch)
- 🎯 **Ranking Stage** — Multi-task learning for joint CTR & CVR prediction
- 📦 **Sequence Modeling** — User behavior sequence preprocessing and encoding
- ⚡ **Training** — Supports pointwise / pairwise / listwise loss with early stopping and TensorBoard logging

### Project Structure

```
Ali_CCP/
├── main_lightning.py              # New training entry point (PyTorch Lightning version)
├── main.py                        # Original training code (unchanged)
├── src/
│   ├── model.py                   # Original model definition (unchanged)
│   ├── dataset.py                 # Original dataset (unchanged)
│   ├── datamodule.py              # New: Lightning data module (with negative sampling support)
│   └── lightning_module.py        # New: Lightning model module
```

### Usage

#### Basic Training

```bash
python main_lightning.py \
    --train_data_path /path/to/train.csv \
    --val_data_path /path/to/val.csv \
    --mode single \
    --epochs 20 \
    --batch_size 2048 \
    --lr 1e-3
```

#### Using Negative Sampling (Recommended)

Train with 50% of negative samples:

```bash
python main_lightning.py \
    --neg_sample_ratio 0.5 \
    --epochs 20 \
    --batch_size 2048
```

Train with 30% of negative samples:

```bash
python main_lightning.py \
    --neg_sample_ratio 0.3 \
    --epochs 20
```

#### Different Training Modes

**Pointwise BCE (recommended for imbalanced data):**
```bash
python main_lightning.py --mode single
```

**Pairwise BPR:**
```bash
python main_lightning.py --mode pair
```

**Listwise InfoNCE:**
```bash
python main_lightning.py --mode list --temperature 0.05
```

#### Advanced Configuration

**Train with GPU:**
```bash
python main_lightning.py --accelerator gpu --devices 1
```

**Mixed precision training:**
```bash
python main_lightning.py --precision 16
```

**Early Stopping:**
```bash
python main_lightning.py --early_stop_patience 5
```

**Resume from checkpoint:**
```bash
python main_lightning.py --resume_from_checkpoint checkpoints/last.ckpt
```

### Command Line Arguments

#### Data Parameters
- `--train_data_path`: Path to training data
- `--val_data_path`: Path to validation data
- `--neg_sample_ratio`: Negative sampling ratio (0.0-1.0), default 1.0

#### Model Parameters
- `--user_num`: Number of users, default 1000
- `--item_num`: Number of items, default 5000
- `--embed_dim`: Embedding dimension, default 16
- `--hidden_dims`: Hidden layer dimensions, default [128, 64]
- `--tower_out_dim`: Tower output dimension, default 32
- `--dropout`: Dropout ratio, default 0.1

#### Training Parameters
- `--mode`: Training mode (single/pair/list), default single
- `--epochs`: Number of epochs, default 20
- `--batch_size`: Batch size, default 2048
- `--lr`: Learning rate, default 1e-3
- `--weight_decay`: Weight decay, default 1e-5
- `--temperature`: InfoNCE temperature parameter, default 0.05

#### System Parameters
- `--num_workers`: Number of data loading workers, default 4
- `--accelerator`: Accelerator type (auto/cpu/gpu), default auto
- `--devices`: Number of devices to use, default 1
- `--precision`: Training precision (32/16/bf16), default 32
- `--seed`: Random seed, default 42

#### Checkpointing and Logging
- `--save_dir`: Model checkpoint directory, default checkpoints
- `--log_dir`: Log directory, default runs
- `--save_top_k`: Save top k models, default 3
- `--early_stop_patience`: Early stopping patience, default 5

## Installation

## Dataset