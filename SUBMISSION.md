# Interview Submission - Varun Reddy

## 1. Theoretical Question
**"Suppose that we design a deep architecture to represent a sequence by stacking self-attention layers with positional encoding. What could be issues?"**

When designing deep architectures that stack self-attention layers with positional encoding, several critical issues emerge. **First, positional information degradation** becomes a concern — since positional encodings are typically added only at the input layer, the explicit positional signal progressively dilutes as information flows through multiple self-attention layers, making it difficult for deeper layers to maintain exact position awareness. **Second, rank collapse and representation degeneration** can occur in deeper layers where attention distributions become overly uniform (doubly exponential convergence), causing the model to lose the ability to distinguish between distinct tokens or positions effectively. **Third, training instability** arises because deep self-attention architectures without careful residual connections or normalization can suffer from vanishing gradients. **Finally, lack of inductive bias** means standard self-attention treats all positions equally by default; without mechanisms like relative positional bias at each layer, the model struggles to learn locality or sequential dependencies efficiently from limited data.

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
