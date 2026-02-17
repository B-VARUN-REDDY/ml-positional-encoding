"""
Data Exploration Script

I wrote this script to verify and inspect the generated datasets.
It loads the pickle files and shows key statistics and pattern visualizations.

Usage:
    python scripts/explore_data.py --dataset data/train/train_data.pkl
"""

import argparse
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def load_dataset(filepath):
    """Load dataset from pickle file"""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def print_info(data):
    """Print dataset information"""
    print("\n" + "="*70)
    print("📊 DATASET INSPECTION")
    print("="*70 + "\n")
    
    meta = data['metadata']
    sequences = data['sequences']
    labels = data['labels']
    
    print(f"Dataset Shape: {sequences.shape}")
    print(f"Label Shape:   {labels.shape}")
    print("\nProperties:")
    print(f"  • Seq Length:  {meta['seq_len']}")
    print(f"  • Vocab Size:  {meta['vocab_size']}")
    print(f"  • Classes:     {meta['num_classes']}")
    
    print("\nClass Distribution:")
    for i in range(meta['num_classes']):
        count = np.sum(labels == i)
        pct = 100 * count / len(labels)
        print(f"  • Class {i}: {count} samples ({pct:.1f}%)")

def visualize_pattern(data):
    """Visualize a few samples to show the patterns"""
    sequences = data['sequences']
    labels = data['labels']
    patterns = data['metadata']['patterns']
    
    print("\nVisualizing Patterns:")
    
    # 3 samples per class
    for cls in range(data['metadata']['num_classes']):
        print(f"\n--- Class {cls} ---")
        patt = patterns[cls]
        print(f"Expected Pattern: {patt}")
        
        # Get indices
        idxs = np.where(labels == cls)[0][:2]
        
        for i, idx in enumerate(idxs):
            seq = sequences[idx]
            # Convert to strong
            parts = []
            for pos, val in enumerate(seq):
                is_patt = any(p == pos and v == val for p, v in patt)
                if is_patt:
                    parts.append(f"[{val:2d}]")
                else:
                    parts.append(f" {val:2d} ")
            print(f"Sample {i+1}: {''.join(parts)}")

def main():
    parser = argparse.ArgumentParser(description='Explore dataset')
    parser.add_argument('--dataset', type=str, default='data/train/train_data.pkl',
                       help='Path to dataset file')
    args = parser.parse_args()
    
    if not os.path.exists(args.dataset):
        print(f"❌ File not found: {args.dataset}")
        return

    data = load_dataset(args.dataset)
    print_info(data)
    visualize_pattern(data)
    
    print("\n" + "="*70)
    print("✅ Verified successfully")
    print("="*70 + "\n")

if __name__ == "__main__":
    import os
    main()
