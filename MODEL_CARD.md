# Model Documentation

Following ML best practices, I documented this model using the Model Card framework.

## Model Overview

**Type:** Transformer-based sequence classifier  
**Task:** Position-aware pattern detection  
**Author:** Varun Reddy  
**Date:** February 2026

## Architecture

I built a transformer with pluggable positional encodings:
- Token embeddings
- Learnable positional encoding (3 variants)
- 3 transformer blocks with multi-head attention
- Global pooling + classification head

**Hyperparameters:**
- d_model: 128
- num_heads: 8  
- num_layers: 3
- Parameters: ~478K

## Training

**Data:** Synthetic position-aware patterns
- 5,000 training samples
- 1,000 validation samples
- 3 classes based on value-position patterns

**Training Setup:**
- Optimizer: AdamW (lr=1e-3)
- Batch size: 32
- Epochs: 20
- Time: ~8-10 minutes

## Results

| Method | Validation Accuracy |
|--------|-------------------|
| Learned Absolute | 95.1% |
| Learned Relative | 93.2% |
| Continuous MLP | 89.3% |
| Sinusoidal | 78.1% |
| No Position | 35.8% |

**Key Finding:** Position encoding is critical (35% → 95%)

## Limitations

- Trained on synthetic data only
- Fixed maximum sequence length (512)
- Small model size (478K parameters)
- O(n²) complexity for long sequences

## Intended Use

✅ Research and education  
✅ Understanding positional encodings  
✅ Baseline for experimentation  
❌ Production without validation  
❌ High-stakes applications

## Repository

https://github.com/B-VARUN-REDDY/ml-positional-encoding

---

**Built by Varun Reddy** | February 2026
