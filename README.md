# Learnable Positional Encoding Methods for Self-Attention Models

My implementation and comparison of different learnable positional encoding strategies for transformer-based models. I built this to explore how different position encoding methods affect model performance.

**Author:** Varun Reddy  
**GitHub:** [B-VARUN-REDDY](https://github.com/B-VARUN-REDDY)  
**Date:** February 2026

## 📋 Table of Contents

- [Overview](#overview)
- [Problem Analysis](#problem-analysis)
- [Methods I Implemented](#methods-i-implemented)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Experimental Results](#experimental-results)
- [Project Structure](#project-structure)
- [Technical Details](#technical-details)
- [Future Work](#future-work)

---

## 🎯 Overview

I designed this project to explore learnable positional encoding methods for deep self-attention architectures. My goal was to implement multiple approaches and empirically compare their effectiveness on position-aware tasks.

### What I Built

✅ **Three different learnable positional encoding methods:**
- Learned Absolute Positional Embeddings (inspired by BERT)
- Learned Relative Position Bias (inspired by T5)
- Continuous Positional Encoding using MLPs

✅ **A complete transformer architecture** with multi-head self-attention  
✅ **Position-aware datasets** to properly test each method  
✅ **Full training pipeline** with logging, checkpointing, and visualization  
✅ **Visualization tools** to analyze attention patterns  
✅ **Comprehensive test suite** with 20+ tests

---

## 📊 Problem Analysis

### Issues with Stacking Self-Attention Layers + Positional Encoding

When designing deep architectures that stack self-attention layers with positional encoding, several critical issues emerge. **First, positional information degradation** becomes a concern - since positional encodings are typically added only at the input layer, the explicit positional signal progressively dilutes as information flows through multiple self-attention layers, making it difficult for deeper layers to maintain position awareness. **Second, computational and memory complexity** scales quadratically O(n²) with sequence length due to attention's all-to-all comparison, making it expensive for long sequences and causing memory bottlenecks. **Third, training instability and gradient flow problems** arise because deep self-attention architectures can suffer from vanishing/exploding gradients without careful architecture choices like residual connections. **Fourth, the absence of inductive biases** means self-attention treats all positions equally initially, lacking the locality bias of CNNs or sequential bias of RNNs, which can make learning from limited data challenging. **Fifth, rank collapse and representation degeneration** can occur in deeper layers where attention distributions become overly uniform, causing all positions to attend similarly and losing diversity. **Finally, there are practical implementation challenges** including careful layer normalization placement, difficulties extrapolating to longer sequences than seen during training, and the computational cost of storing attention matrices for all layers. These issues require architectural innovations like residual connections, layer normalization, relative position representations, and sparse attention patterns to build effective deep self-attention networks.

---

## 📊 The Data

I generated a custom **Position-Aware Dataset** to rigorously test these methods. The task is to classify sequences based on values at specific positions.

**Now Included:** Pre-generated datasets in `data/` for easy inspection.
- `data/train/`: 5,000 samples (CSV + Pickle)
- `data/val/`: 1,000 samples
- `data/demo/`: 100 samples

You can open `data/train/train_data.csv` in Excel to see the patterns yourself!

**Note:** The training scripts (`src/train.py` and `scripts/quick_demo.py`) automatically detect and use this pre-generated data, maintaining consistency across experiments.

---

## 🛠️ Methods I Implemented

### 1. Learned Absolute Positional Embeddings

**My Approach:** Each position has its own learned embedding vector (similar to BERT)

```python
class LearnedAbsolutePositionalEncoding(nn.Module):
    """Position-specific learned embeddings"""
    def __init__(self, d_model, max_len=512):
        self.position_embeddings = nn.Embedding(max_len, d_model)
```

**Pros:**
- ✅ Simple and effective
- ✅ Position-specific learning
- ✅ Fast lookup

**Cons:**
- ❌ Fixed maximum length
- ❌ Cannot generalize beyond max_len
- ❌ Memory grows with max_len

### 2. Learned Relative Position Bias

**My Approach:** Learn biases based on relative distances (like T5)

```python
class LearnedRelativePositionalBias(nn.Module):
    """Relative position biases for attention"""
    def __init__(self, num_heads, max_distance=128):
        self.relative_attention_bias = nn.Embedding(2*max_distance+1, num_heads)
```

**Pros:**
- ✅ Generalizes to unseen lengths
- ✅ Captures pairwise relationships
- ✅ Applied at each layer

**Cons:**
- ❌ More complex to implement
- ❌ Requires attention integration

### 3. Continuous Positional Encoding with MLPs

**My Approach:** Map normalized positions through a neural network

```python
class ContinuousPositionalEncoding(nn.Module):
    """MLP-based continuous embeddings"""
    def __init__(self, d_model, hidden_dim=128):
        self.position_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
```

**Pros:**
- ✅ Handles arbitrary lengths
- ✅ Smooth interpolation
- ✅ Flexible

**Cons:**
- ❌ More parameters
- ❌ May overfit
- ❌ Needs careful tuning

### Comparison

| Method | Max Length | Generalization | Parameters | Speed |
|--------|-----------|----------------|------------|-------|
| Learned Absolute | Fixed | Poor | O(max_len × d) | Fast |
| Relative Bias | Flexible | Good | O(max_dist × heads) | Medium |
| Continuous MLP | Unlimited | Excellent | O(hidden × d) | Slow |
| Sinusoidal | Fixed | Poor | 0 | Fast |

---

## 💻 Installation

### Prerequisites
- Python 3.8+
- CUDA (optional)

### Setup

```bash
# Clone repository
git clone https://github.com/B-VARUN-REDDY/ml-positional-encoding.git
cd ml-positional-encoding

# Create environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### Train a Model

```bash
# Learned absolute
python src/train.py --pos_encoding learned_absolute --num_epochs 20

# Relative bias
python src/train.py --pos_encoding learned_relative --num_epochs 20

# Continuous
python src/train.py --pos_encoding continuous --num_epochs 20
```

### Compare All Methods

```bash
python scripts/compare_all.py
```

### Run Tests

```bash
python -m pytest
```

### Interactive Demo

Open `notebooks/demo.ipynb` in Jupyter/VS Code for an interactive walkthrough.

---

## 📖 Usage Examples

### Creating Positional Encodings

```python
import torch
from src.positional_encodings import *

# Input
batch_size, seq_len, d_model = 4, 32, 128
x = torch.randn(batch_size, seq_len, d_model)

# Learned absolute
pos_enc = LearnedAbsolutePositionalEncoding(d_model, max_len=512)
x_with_pos = pos_enc(x)

# Continuous
cont_enc = ContinuousPositionalEncoding(d_model, hidden_dim=128)
x_with_pos = cont_enc(x)

# Relative bias (for attention)
rel_bias = LearnedRelativePositionalBias(num_heads=8, max_distance=128)
bias = rel_bias(seq_len)
```

---

## 📈 Experimental Results

### My Dataset: Position-Aware Pattern Detection

**Task:** Classify sequences based on value-position patterns
- Class 0: value=5 at position 3 AND value=8 at position 7
- Class 1: value=3 at position 5 AND value=9 at position 10
- Class 2: value=7 at position 15 AND value=4 at position 20

**Settings:**
- Sequence Length: 32
- Vocabulary: 20
- Training: 5000 samples
- Validation: 1000 samples
- Model: 3-layer Transformer, d_model=128, 8 heads

### Results

| Method | Best Val Acc | Parameters | Training Time |
|--------|-------------|------------|---------------|
| **Learned Absolute** | ~95% | 478K | 8.3 min |
| **Relative Bias** | ~93% | 465K | 9.7 min |
| **Continuous MLP** | ~89% | 492K | 11.2 min |
| Sinusoidal | ~78% | 445K | 8.1 min |
| No Position | ~35% | 445K | 7.9 min |

### Key Findings

1. **Position encoding is critical** (35% → 95% improvement)
2. **Learned absolute works best** for fixed-length tasks
3. **Relative bias generalizes better**
4. **Continuous MLP needs more tuning**

---

## 📁 Project Structure

```
ml-positional-encoding/
├── README.md                     
├── requirements.txt              
├── src/
│   ├── positional_encodings.py   # My encoding implementations
│   ├── model.py                  # Transformer architecture
│   ├── dataset.py                # Dataset generators
│   └── train.py                  # Training script
├── tests/
│   └── test_positional_encoding.py
├── scripts/
│   └── compare_all.py
└── experiments/
    └── results/
```

---

## 🔬 Technical Details

### Architecture

- Token Embedding + Positional Encoding
- 3 Transformer Blocks
- Layer Normalization
- Global Average Pooling
- Classification Head

### Hyperparameters

**Model:**
- d_model: 128
- num_heads: 8
- num_layers: 3
- dropout: 0.1

**Training:**
- Optimizer: AdamW
- Learning Rate: 1e-3
- Batch Size: 32
- Epochs: 20

---

## 🔄 Future Work

### Potential Extensions

1. **RoPE** (Rotary Position Embeddings)
2. **ALiBi** (Attention with Linear Biases)
3. **2D Positional Encodings**
4. **Learned Fourier Features**

### Experiments

- Extrapolation tests
- Long-sequence benchmarks
- Multi-task evaluation

---

## 📚 References

1. **Attention Is All You Need** - Vaswani et al., 2017
2. **BERT** - Devlin et al., 2019
3. **T5** - Raffel et al., 2020
4. **RoFormer** - Su et al., 2021
5. **ALiBi** - Press et al., 2022

---

## 📄 License

MIT License

---

**Built by Varun Reddy** | February 2026
