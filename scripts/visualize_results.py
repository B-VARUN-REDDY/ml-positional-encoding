"""
My Visualization Script

I wrote this script to load all training history files and generate 
publication-quality comparison plots for my report. It handles loading
results from multiple experiments and plotting them together.

Usage:
    python scripts/visualize_results.py
"""

import sys
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path if needed
sys.path.append('src')

def main():
    print("\n" + "="*70)
    print("📊 Generating Visualizations for My Report")
    print("="*70 + "\n")

    experiments_dir = Path('experiments/results')
    if not experiments_dir.exists():
        print(f"❌ '{experiments_dir}' not found. Run training first!")
        return

    # Find all training_history.json files
    history_files = list(experiments_dir.glob('**/training_history.json'))
    
    if not history_files:
        print(f"❌ No training history found in '{experiments_dir}'.")
        print("   Did you run: python src/train.py --pos_encoding learned_absolute --num_epochs 10?")
        return

    print(f"Found {len(history_files)} experiments to visualize.")

    # Data collection
    all_data = []
    summary_metrics = []

    for f in history_files:
        exp_name = f.parent.name
        try:
            with open(f, 'r') as json_file:
                hist = json.load(json_file)
            
            # Identify method name from folder or config
            method_name = "Unknown"
            if "learned_absolute" in exp_name: method_name = "Learned Absolute"
            elif "learned_relative" in exp_name: method_name = "Learned Relative"
            elif "continuous" in exp_name: method_name = "Continuous MLP"
            elif "sinusoidal" in exp_name: method_name = "Sinusoidal"
            elif "none" in exp_name: method_name = "No Position"

            # Check config file too
            config_path = f.parent / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as cf:
                    cfg = json.load(cf)
                    if 'pos_encoding' in cfg:
                        raw_name = cfg['pos_encoding'] 
                        method_mapping = {
                            'learned_absolute': 'Learned Absolute',
                            'learned_relative': 'Learned Relative',
                            'continuous': 'Continuous MLP',
                            'sinusoidal': 'Sinusoidal',
                            'none': 'No Position'
                        }
                        method_name = method_mapping.get(raw_name, raw_name)

            # Extract curves
            epochs = hist['epochs']
            val_acc = hist['val_acc']
            train_acc = hist['train_acc']
            
            for e, va, ta in zip(epochs, val_acc, train_acc):
                all_data.append({
                    'Epoch': e,
                    'Validation Accuracy': va,
                    'Train Accuracy': ta,
                    'Method': method_name
                })

            # Extract summary metrics
            best_val = max(val_acc)
            best_epoch = val_acc.index(best_val) + 1
            final_train = train_acc[-1]
            
            summary_metrics.append({
                'Method': method_name,
                'Best Validation Error': 100 - best_val, # for error comparison
                'Best Accuracy': best_val,
                'Best Epoch': best_epoch,
                'Final Train Accuracy': final_train
            })
            
        except Exception as e:
            print(f"⚠️ Could not process {exp_name}: {e}")

    if not all_data:
        print("No valid data extracted.")
        return

    df = pd.DataFrame(all_data)
    summary_df = pd.DataFrame(summary_metrics).sort_values('Best Accuracy', ascending=False)

    print("\nResults Summary:")
    print(summary_df[['Method', 'Best Accuracy', 'Best Epoch']].to_string(index=False))

    # Plot 1: Validation Accuracy Curves
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.lineplot(data=df, x='Epoch', y='Validation Accuracy', hue='Method', marker='o', linewidth=2.5)
    plt.title('Validation Accuracy Comparison: Positional Encoding Strategies', fontsize=14, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Training Epoch', fontsize=12)
    plt.legend(title='Encoding Method', loc='lower right')
    plt.tight_layout()
    plt.savefig('experiments/results/accuracy_comparison.png', dpi=300)
    print("\n✅ Saved accuracy_comparison.png")

    # Plot 2: Best Performance Bar Chart
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(data=summary_df, x='Method', y='Best Accuracy', hue='Method', palette='viridis', legend=False)
    plt.title('Peak Performance by Method', fontsize=14, fontweight='bold')
    plt.ylabel('Best Validation Accuracy (%)', fontsize=12)
    plt.ylim(0, 100)
    
    # Add labels
    for i, v in enumerate(summary_df['Best Accuracy']):
        ax.text(i, v + 2, f"{v:.1f}%", ha='center', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig('experiments/results/performance_bars.png', dpi=300)
    print("✅ Saved performance_bars.png")

    print(f"\n🎉 All visualizations generated in {experiments_dir}")

if __name__ == "__main__":
    main()
