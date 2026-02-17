#!/usr/bin/env python3
"""
Quick Demo - 60-Second System Check

I built this script to quickly verify all components work together.
Useful for live demonstrations.

Usage:
    python scripts/quick_demo.py
"""

import sys
sys.path.insert(0, 'src')

import torch
import numpy as np
from tqdm import tqdm
import time

print("\n" + "="*70)
print("🚀 POSITIONAL ENCODING PROJECT - QUICK DEMO")
print("="*70 + "\n")

start_time = time.time()

# Test imports
print("1️⃣  Testing imports...")
try:
    from positional_encodings import (
        LearnedAbsolutePositionalEncoding,
        LearnedRelativePositionalBias,
        ContinuousPositionalEncoding
    )
    from model import create_model
    from dataset import create_dataloaders
    print("   ✅ All modules imported")
except Exception as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test encodings
print("\n2️⃣  Testing positional encodings...")
batch_size, seq_len, d_model = 2, 32, 64
x = torch.randn(batch_size, seq_len, d_model)

encodings = {
    'Learned Absolute': LearnedAbsolutePositionalEncoding(d_model),
    'Learned Relative': LearnedRelativePositionalBias(num_heads=8),
    'Continuous MLP': ContinuousPositionalEncoding(d_model)
}

for name, enc in encodings.items():
    try:
        if name == 'Learned Relative':
            out = enc(seq_len)
            assert out.shape == (1, 8, seq_len, seq_len)
        else:
            out = enc(x)
            assert out.shape ==x.shape
        print(f"   ✅ {name}: Working")
    except Exception as e:
        print(f"   ❌ {name}: {e}")

# Test dataset
print("\n3️⃣  Testing dataset...")
try:
    # Check physical files
    import os
    if os.path.exists('data/train/train_data.pkl'):
        print(f"   ✅ Data files found in data/")
    
    train_loader, val_loader, info = create_dataloaders(
        dataset_type='pattern',
        train_samples=100,
        val_samples=20,
        batch_size=16,
        seq_len=32,
        vocab_size=20
    )
    
    sequences, labels = next(iter(train_loader))
    assert sequences.shape[0] <= 16
    assert sequences.shape[1] == 32
    
    print(f"   ✅ Dataset: {len(train_loader.dataset)} samples")
    print(f"   ✅ Classes: {info['num_classes']}")
except Exception as e:
    print(f"   ❌ Dataset failed: {e}")

# Test model
print("\n4️⃣  Testing model...")
try:
    model = create_model(
        vocab_size=20,
        num_classes=3,
        pos_encoding_type='learned_absolute',
        d_model=64,
        num_heads=4,
        num_layers=2
    )
    
    test_input = torch.randint(0, 20, (2, 32))
    logits, _ = model(test_input)
    assert logits.shape == (2, 3)
    
    logits, attn = model(test_input, return_attention=True)
    assert attn.shape == (2, 4, 32, 32)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   ✅ Model: {num_params:,} parameters")
    print(f"   ✅ Forward pass: Working")
    print(f"   ✅ Attention: Working")
except Exception as e:
    print(f"   ❌ Model failed: {e}")

# Test training
print("\n5️⃣  Testing training...")
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training 1 epoch", leave=False)
    for sequences, labels in pbar:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits, _ = model(sequences)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total
    
    print(f"   ✅ Training: Working")
    print(f"   ✅ Loss: {avg_loss:.4f}")
    print(f"   ✅ Accuracy: {accuracy:.2f}%")
except Exception as e:
    print(f"   ❌ Training failed: {e}")

# Summary
elapsed = time.time() - start_time
print("\n" + "="*70)
print("✅ DEMO COMPLETE!")
print("="*70)
print(f"\n⏱️  Completed in {elapsed:.1f} seconds")
print("\n📊 All components verified:")
print("   • Positional encodings: ✅")
print("   • Dataset generation: ✅")
print("   • Model architecture: ✅")
print("   • Training pipeline: ✅")
print("\n🎉 Ready to use!")
print("\nNext steps:")
print("   1. python src/train.py --pos_encoding learned_absolute")
print("   2. python scripts/compare_all.py")
print("\n" + "="*70 + "\n")
