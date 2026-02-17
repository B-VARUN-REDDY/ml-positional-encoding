# Interview Submission - Varun Reddy

## 1. Theoretical Question
**"Suppose that we design a deep architecture to represent a sequence by stacking self-attention layers with positional encoding. What could be issues?"**

When designing deep architectures that stack self-attention layers with positional encoding, several critical issues emerge that can significantly impact model performance and training efficiency. **First, positional information degradation** becomes a fundamental concern - since positional encodings are typically added only at the input layer, the explicit positional signal progressively dilutes as information flows through multiple self-attention layers, making it difficult for deeper layers to maintain fine-grained position awareness. **Second, computational and memory complexity** scales quadratically O(n²) with sequence length due to the attention mechanism's all-to-all comparison nature, making it prohibitively expensive for long sequences and causing GPU memory bottlenecks in production systems. **Third, training instability and gradient flow problems** arise because deep self-attention architectures lack the natural gradient highway that residual connections provide, leading to vanishing/exploding gradients, difficulty in optimization, and requiring careful initialization strategies and learning rate scheduling. **Fourth, the absence of inductive biases** for local patterns means self-attention treats all positions equally initially, lacking the locality bias of CNNs or the sequential bias of RNNs, which can make learning from limited data challenging and require massive datasets to learn basic patterns. **Fifth, rank collapse and representation degeneration** can occur in deeper layers where attention distributions become overly uniform (doubly exponential convergence) or overly peaked, causing all positions to attend similarly and losing representational diversity. **Finally, there are practical issues** including the need for careful layer normalization placement, difficulties in extrapolating to longer sequences than seen during training, challenges in maintaining meaningful attention patterns across many layers, and the computational cost of storing and computing attention matrices for all layers. These issues necessitate architectural innovations like residual connections, layer normalization, relative position representations, sparse attention patterns, and careful hyperparameter tuning to build effective deep self-attention networks.

## 2. Practical Implementation 
**"Can you design a learnable positional encoding method using pytorch? (Create dummy dataset)"**

Yes. I have implemented three different learnable approaches and a custom position-aware dataset.

### The Solution
*   **Learnable Encoding**: Implemented in [`src/positional_encodings.py`](src/positional_encodings.py) (Class `LearnedAbsolutePositionalEncoding` and `LearnedRelativePositionalBias`).
*   **Dummy Dataset**: Implemented in [`src/dataset.py`](src/dataset.py) (Class `PositionAwarePatternDataset`).
*   **Training Loop**: Complete training pipeline in [`src/train.py`](src/train.py).

### How to Verify (In < 60 Seconds)
I have created a quick verification script that generates the dataset, creates the model, and runs a training step:

```bash
python scripts/quick_demo.py
```

### Results
I compared the learnable methods against a baseline. The **Learned Absolute** encoding achieved **99.9% accuracy** on the dummy dataset, compared to 58% for the baseline (no position encoding).

See full results in [`experiments/results/comparison_summary.md`](experiments/results/comparison_summary.md).
