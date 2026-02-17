# Quick Start Guide

Get started with the ML Positional Encoding project in 5 minutes!

## ⚡ Setup (2 minutes)

```bash
# 1. Navigate to the project
cd ml-positional-encoding

# 2. Create virtual environment
python -m venv venv

# 3. Activate virtual environment
# On Mac/Linux:
source venv/bin/activate
# On Windows PowerShell:
venv\Scripts\Activate.ps1
# On Windows CMD:
# venv\Scripts\activate.bat

# 4. Install dependencies
pip install -r requirements.txt
```

## 🎯 Quick Test (1 minute)

Verify everything works:

```bash
# Run tests
python tests/test_positional_encoding.py
```

You should see: `✓ ALL TESTS PASSED!`

## 🚀 Train Your First Model (5 minutes)

```bash
# Train with learned absolute positional encoding
python src/train.py --pos_encoding learned_absolute --num_epochs 10
```

This will:
- ✅ Create a position-aware dataset
- ✅ Train a transformer model
- ✅ Save results to `experiments/results/`
- ✅ Generate training curves

## 📊 View Results

After training, check:
- **Training curves**: `experiments/results/learned_absolute_d128_h8_l3/visualizations/`
- **Checkpoints**: `experiments/results/learned_absolute_d128_h8_l3/checkpoints/`
- **Logs**: `experiments/results/learned_absolute_d128_h8_l3/`

## 🔍 Compare All Methods (30 minutes)

Train and compare all positional encoding methods:

```bash
python scripts/compare_all.py
```

This generates:
- Comparison plots
- Summary table (CSV)
- Detailed markdown report

Located in: `experiments/results/`

## 📋 Command Reference

### Training Options

```bash
# Different positional encodings
python src/train.py --pos_encoding learned_absolute
python src/train.py --pos_encoding learned_relative
python src/train.py --pos_encoding continuous
python src/train.py --pos_encoding sinusoidal
python src/train.py --pos_encoding none  # Ablation study

# Customize architecture
python src/train.py \
    --pos_encoding learned_absolute \
    --d_model 256 \
    --num_heads 8 \
    --num_layers 6 \
    --dropout 0.1

# Adjust training
python src/train.py \
    --num_epochs 50 \
    --batch_size 64 \
    --lr 5e-4 \
    --train_samples 10000

# Use GPU (if available)
python src/train.py --device cuda
```

### Testing

```bash
# Run all tests
python tests/test_positional_encoding.py
```

## 🆘 Troubleshooting

### Issue: Module not found

```bash
# Make sure you're in the right directory
cd ml-positional-encoding

# Make sure venv is activated (you should see (venv) in your terminal)
# On Windows PowerShell:
venv\Scripts\Activate.ps1

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: CUDA out of memory

```bash
# Reduce batch size
python src/train.py --batch_size 16

# Or use CPU
python src/train.py --device cpu
```

### Issue: Training too slow

```bash
# Reduce samples
python src/train.py --train_samples 1000 --val_samples 200

# Reduce epochs
python src/train.py --num_epochs 10
```

---

## 💡 Tips

1. **Start simple**: Run the quick test first to verify setup
2. **Compare methods**: The comparison script gives the best insights
3. **Visualize**: Training curves reveal what the model learned
4. **Document**: Take notes on what you observe

---

## 📚 Next Steps

Once you're comfortable:
- Review README.md for complete documentation
- Read START_HERE.md for submission instructions
- Experiment with different architectures
- Record your video walkthrough

---

**Good luck with your interview! 🚀**
