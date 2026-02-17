# Quick Start Guide

Here's how to get up and running with my positional encoding project.

## Setup

```bash
cd ml-positional-encoding

# Create environment
python -m venv venv
venv\Scripts\Activate.ps1  # Windows

# Install
pip install -r requirements.txt
```

## Quick Verify

```bash
python scripts/quick_demo.py          # 60-second full system check
jupyter notebook notebooks/demo.ipynb # Interactive walkthrough
```

## Test

```bash
python tests/test_positional_encoding.py
```

## Train

```bash
# Train with learned absolute position encoding
python src/train.py --pos_encoding learned_absolute --num_epochs 10

# Or try other methods
python src/train.py --pos_encoding learned_relative --num_epochs 10
python src/train.py --pos_encoding continuous --num_epochs 10
```

## Compare All Methods

```bash
python scripts/compare_all.py --num_epochs 20
```

This will train all methods and generate comparison plots.

## Common Commands

```bash
# Different encodings
python src/train.py --pos_encoding learned_absolute
python src/train.py --pos_encoding learned_relative
python src/train.py --pos_encoding continuous
python src/train.py --pos_encoding sinusoidal
python src/train.py --pos_encoding none  # Ablation

# Custom architecture
python src/train.py \
    --pos_encoding learned_absolute \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 6

# Fast test
python src/train.py --train_samples 1000 --val_samples 200 --num_epochs 5
```

## Troubleshooting

**"Module not found"**
```bash
cd ml-positional-encoding
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Training too slow**
```bash
python src/train.py --batch_size 16 --train_samples 1000
```

---

**Built by Varun Reddy** | 2026
