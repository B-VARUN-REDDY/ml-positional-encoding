"""
Compare All Positional Encoding Methods

Trains and compares all positional encoding methods with the same hyperparameters.
Generates comparison plots and summary table.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import argparse

from train import main as train_main


def compare_all_methods(
    num_epochs: int = 20,
    device: str = 'auto',
    output_dir: str = 'experiments/results'
):
    """
    Train all positional encoding methods and compare results.
    
    Args:
        num_epochs: Number of epochs to train each model
        device: Device to use ('auto', 'cuda', 'cpu')
        output_dir: Output directory for results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Methods to compare
    methods = [
        'learned_absolute',
        'learned_relative',
        'continuous',
        'sinusoidal',
        'none'  # Ablation study
    ]
    
    # Shared hyperparameters
    base_args = [
        '--d_model', '128',
        '--num_heads', '8',
        '--num_layers', '3',
        '--batch_size', '32',
        '--lr', '0.001',
        '--train_samples', '5000',
        '--val_samples', '1000',
        '--seq_len', '32',
        '--vocab_size', '20',
        '--num_epochs', str(num_epochs),
        '--device', device,
        '--output_dir', output_dir
    ]
    
    results = {}
    
    print("="*80)
    print("COMPARING ALL POSITIONAL ENCODING METHODS")
    print("="*80)
    print(f"\nDevice: {device}")
    print(f"Epochs per method: {num_epochs}")
    print(f"Methods: {', '.join(methods)}")
    print("\n" + "="*80 + "\n")
    
    # Train each method
    for i, method in enumerate(methods, 1):
        print(f"\n{'#'*80}")
        print(f"# [{i}/{len(methods)}] Training: {method.upper()}")
        print(f"{'#'*80}\n")
        
        # Prepare arguments
        sys.argv = ['train.py', '--pos_encoding', method] + base_args
        
        try:
            # Run training
            train_main()
            
            # Load results
            exp_name = f"{method}_d128_h8_l3"
            result_dir = output_path / exp_name
            
            with open(result_dir / 'training_history.json', 'r') as f:
                history = json.load(f)
            
            with open(result_dir / 'config.json', 'r') as f:
                config = json.load(f)
            
            # Extract metrics
            best_val_acc = max(history['val_acc'])
            best_epoch = history['val_acc'].index(best_val_acc) + 1
            final_val_acc = history['val_acc'][-1]
            final_train_acc = history['train_acc'][-1]
            
            results[method] = {
                'best_val_acc': best_val_acc,
                'best_epoch': best_epoch,
                'final_val_acc': final_val_acc,
                'final_train_acc': final_train_acc,
                'history': history,
                'config': config
            }
            
            print(f"\n✓ {method}: Best Val Acc = {best_val_acc:.2f}% (epoch {best_epoch})")
            
        except Exception as e:
            print(f"\n✗ {method} failed: {e}")
            results[method] = None
    
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80 + "\n")
    
    # Generate comparison plots
    generate_comparison_plots(results, output_path)
    
    # Generate summary table
    generate_summary_table(results, output_path)
    
    print(f"\n✓ All results saved to: {output_path}")
    print("="*80 + "\n")


def generate_comparison_plots(results: Dict, output_dir: Path):
    """Generate comparison plots for all methods."""
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (16, 10)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Filter out failed methods
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # Plot 1: Validation Accuracy over epochs
    ax = axes[0, 0]
    for method, data in valid_results.items():
        epochs = data['history']['epochs']
        val_acc = data['history']['val_acc']
        ax.plot(epochs, val_acc, marker='o', label=method.replace('_', ' ').title(), linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Training Loss over epochs
    ax = axes[0, 1]
    for method, data in valid_results.items():
        epochs = data['history']['epochs']
        train_loss = data['history']['train_loss']
        ax.plot(epochs, train_loss, marker='s', label=method.replace('_', ' ').title(), linewidth=2)
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Training Loss', fontsize=12)
    ax.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Best validation accuracy bar chart
    ax = axes[1, 0]
    methods = list(valid_results.keys())
    best_accs = [valid_results[m]['best_val_acc'] for m in methods]
    
    # Color-code bars
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(methods)))
    bars = ax.bar(range(len(methods)), best_accs, color=colors, alpha=0.8, edgecolor='black')
    
    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, best_accs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Best Validation Accuracy (%)', fontsize=12)
    ax.set_title('Best Validation Accuracy by Method', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    
    # Plot 4: Train vs Val accuracy (final epoch)
    ax = axes[1, 1]
    train_accs = [valid_results[m]['final_train_acc'] for m in methods]
    val_accs = [valid_results[m]['final_val_acc'] for m in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, train_accs, width, label='Train Acc', alpha=0.8, color='skyblue', edgecolor='black')
    bars2 = ax.bar(x + width/2, val_accs, width, label='Val Acc', alpha=0.8, color='lightcoral', edgecolor='black')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Final Train vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ').title() for m in methods], rotation=45, ha='right')
    ax.legend(fontsize=10)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'comparison_plots.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved comparison plots to {save_path}")


def generate_summary_table(results: Dict, output_dir: Path):
    """Generate summary table of results."""
    
    # Filter out failed methods
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    # Create DataFrame
    data = []
    for method, res in valid_results.items():
        data.append({
            'Method': method.replace('_', ' ').title(),
            'Best Val Acc (%)': f"{res['best_val_acc']:.2f}",
            'Best Epoch': res['best_epoch'],
            'Final Val Acc (%)': f"{res['final_val_acc']:.2f}",
            'Final Train Acc (%)': f"{res['final_train_acc']:.2f}",
            'Improvement from No Position': f"{res['best_val_acc'] - valid_results.get('none', {'best_val_acc': 0})['best_val_acc']:.2f}" if 'none' in valid_results else 'N/A'
        })
    
    df = pd.DataFrame(data)
    
    # Sort by best validation accuracy
    df['_sort_key'] = df['Best Val Acc (%)'].astype(float)
    df = df.sort_values('_sort_key', ascending=False).drop('_sort_key', axis=1)
    
    # Save as CSV
    csv_path = output_dir / 'comparison_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved summary table to {csv_path}")
    
    # Save as markdown
    md_path = output_dir / 'comparison_summary.md'
    with open(md_path, 'w') as f:
        f.write("# Positional Encoding Methods Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n## Key Findings\n\n")
        
        if 'none' in valid_results:
            baseline_acc = valid_results['none']['best_val_acc']
            f.write(f"- **Baseline (No Position Encoding)**: {baseline_acc:.2f}%\n")
            
            best_method = max(valid_results.items(), 
                            key=lambda x: x[1]['best_val_acc'] if x[0] != 'none' else 0)
            best_acc = best_method[1]['best_val_acc']
            improvement = best_acc - baseline_acc
            
            f.write(f"- **Best Method**: {best_method[0].replace('_', ' ').title()} ({best_acc:.2f}%)\n")
            f.write(f"- **Improvement over No Position**: +{improvement:.2f}%\n")
            f.write(f"- **Relative Improvement**: {100*improvement/baseline_acc:.1f}%\n\n")
        
        f.write("## Insights\n\n")
        f.write("1. **Learned Absolute** typically performs best for fixed-length tasks\n")
        f.write("2. **Learned Relative** shows better generalization potential\n")
        f.write("3. **Continuous** provides flexibility for varying lengths\n")
        f.write("4. **Sinusoidal** offers parameter-free baseline\n")
        f.write("5. **No Position** demonstrates critical importance of positional information\n")
    
    print(f"✓ Saved markdown summary to {md_path}")
    
    # Print table to console
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80 + "\n")
    print(df.to_string(index=False))
    print("\n" + "="*80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare all positional encoding methods')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs per method')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                        help='Output directory')
    
    args = parser.parse_args()
    
    compare_all_methods(
        num_epochs=args.num_epochs,
        device=args.device,
        output_dir=args.output_dir
    )
