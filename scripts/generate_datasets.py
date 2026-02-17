"""
Dataset Generation Script

I wrote this script to pre-generate the synthetic datasets for the project.
This ensures reproducibility and allows for data inspection without running the training loop.

It saves data in three formats:
1. CSV (for Excel/Pandas inspection)
2. Pickle (for fast loading in Python)
3. TXT (human-readable samples with visualizations)

Usage:
    python scripts/generate_datasets.py
"""

import sys
import os
import pickle
import json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append('src')
from dataset import PositionAwarePatternDataset

def save_dataset(dataset, output_dir, name):
    """Save dataset in multiple formats."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {name} set...")
    
    # 1. Extract data
    sequences = []
    labels = []
    for i in range(len(dataset)):
        seq, label = dataset[i]
        sequences.append(seq.numpy())
        labels.append(label.item())
    
    sequences = np.array(sequences)
    labels = np.array(labels)
    
    # 2. Save as Pickle (Standard numpy/python format)
    pkl_path = output_dir / f"{name}_data.pkl"
    with open(pkl_path, 'wb') as f:
        pickle.dump({
            'sequences': sequences,
            'labels': labels,
            'metadata': {
                'vocab_size': dataset.vocab_size,
                'seq_len': dataset.seq_len,
                'num_classes': dataset.num_classes,
                'patterns': dataset.patterns
            }
        }, f)
    print(f"  [OK] Saved pickle: {pkl_path}")

    # 3. Save as CSV (For inspection)
    df = pd.DataFrame(sequences, columns=[f'pos_{i}' for i in range(sequences.shape[1])])
    df.insert(0, 'label', labels)
    csv_path = output_dir / f"{name}_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"  [OK] Saved CSV: {csv_path}")

    # 4. Save Human-Readable Samples (TXT)
    txt_path = output_dir / "sample_sequences.txt"
    with open(txt_path, 'w') as f:
        f.write(f"# {name.capitalize()} Dataset Sample Sequences\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Samples: {len(dataset)}\n")
        f.write(f"Sequence Length: {dataset.seq_len}\n")
        f.write(f"Vocabulary: 0-{dataset.vocab_size-1}\n")
        f.write(f"Classes: {dataset.num_classes}\n\n")
        
        f.write("Patterns:\n")
        for cls_idx, pattern in enumerate(dataset.patterns):
            rule = " AND ".join([f"value={val} at pos={pos}" for pos, val in pattern])
            f.write(f"- Class {cls_idx}: {rule}\n")
        
        f.write("\nSample Sequences (pattern positions shown in [brackets]):\n")
        f.write("="*70 + "\n\n")
        
        # Group by class to show examples of each
        for class_id in range(dataset.num_classes):
            f.write(f"\n--- Class {class_id} ---\n")
            class_indices = np.where(labels == class_id)[0][:5] # Show 5 samples per class
            
            pattern = dataset.patterns[class_id]
            pattern_positions = [p for p, v in pattern]
            
            for idx in class_indices:
                seq = sequences[idx]
                seq_str = ""
                for pos, val in enumerate(seq):
                    val_str = f"{val:2d}"
                    if pos in pattern_positions:
                        seq_str += f"[{val_str}] "
                    else:
                        seq_str += f" {val_str}  "
                f.write(f"Sample {idx+1}: {seq_str}\n")

    print(f"  [OK] Saved description: {txt_path}")
    
    # Save Metadata
    meta_path = output_dir / "dataset_info.json"
    with open(meta_path, 'w') as f:
        json.dump({
            'num_samples': len(dataset),
            'num_classes': dataset.num_classes,
            'vocab_size': dataset.vocab_size,
            'seq_len': dataset.seq_len,
            'patterns': [[list(p) for p in pat] for pat in dataset.patterns]  # Convert tuples to lists
        }, f, indent=2)


def main():
    print("\n" + "="*70)
    print("📦 Generating Datasets")
    print("="*70 + "\n")
    
    # Configuration
    TRAIN_SAMPLES = 5000
    VAL_SAMPLES = 1000
    DEMO_SAMPLES = 100
    
    SEQ_LEN = 32
    VOCAB_SIZE = 20
    NUM_CLASSES = 3
    
    # 1. Generate Training Data
    train_dataset = PositionAwarePatternDataset(
        num_samples=TRAIN_SAMPLES,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seed=42
    )
    save_dataset(train_dataset, 'data/train', 'train')
    
    # 2. Generate Validation Data
    val_dataset = PositionAwarePatternDataset(
        num_samples=VAL_SAMPLES,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seed=43 # Different seed
    )
    save_dataset(val_dataset, 'data/val', 'val')

    # 3. Generate Small Demo Data
    demo_dataset = PositionAwarePatternDataset(
        num_samples=DEMO_SAMPLES,
        seq_len=SEQ_LEN,
        vocab_size=VOCAB_SIZE,
        num_classes=NUM_CLASSES,
        seed=44
    )
    save_dataset(demo_dataset, 'data/demo', 'demo')
    
    print("\n" + "="*70)
    print("🎉 Dataset generation complete!")
    print(f"📂 Saved to: {os.path.abspath('data')}")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
