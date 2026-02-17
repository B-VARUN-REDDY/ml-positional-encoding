"""
Training Script for Positional Encoding Comparison

Complete training pipeline with logging, visualization, and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import time

from model import create_model
from dataset import create_dataloaders


class Trainer:
    """
    Trainer class for positional encoding experiments.
    
    Args:
        model: PyTorch model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        output_dir: Directory for outputs
        config: Configuration dictionary
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        output_dir: Path,
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.output_dir = output_dir
        self.config = config
        
        # Create output directories
        self.checkpoint_dir = output_dir / 'checkpoints'
        self.vis_dir = output_dir / 'visualizations'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.vis_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'epochs': []
        }
        
        self.best_val_acc = 0.0
        self.best_epoch = 0
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for batch_idx, (sequences, labels) in enumerate(pbar):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            logits, _ = self.model(sequences)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.0 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    @torch.no_grad()
    def validate(self) -> Tuple[float, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for sequences, labels in tqdm(self.val_loader, desc='Validation', leave=False):
            sequences = sequences.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits, _ = self.model(sequences)
            loss = self.criterion(logits, labels)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, num_epochs: int) -> Dict:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
        
        Returns:
            Training history dictionary
        """
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Model: {self.config['pos_encoding_type']}")
        print(f"Output directory: {self.output_dir}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{num_epochs}")
            print(f"{'='*60}")
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate()
            
            # Update history
            self.history['epochs'].append(epoch)
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print statistics
            print(f"\nResults:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.save_checkpoint('best_model.pt', epoch)
                print(f"  ✓ New best model! (Val Acc: {val_acc:.2f}%)")
            
            # Save latest checkpoint
            if epoch % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pt', epoch)
        
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete!")
        print(f"Total time: {total_time/60:.2f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}% (epoch {self.best_epoch})")
        print(f"{'='*60}\n")
        
        # Save final results
        self.save_history()
        self.plot_training_curves()
        
        return self.history
    
    def save_checkpoint(self, filename: str, epoch: int):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'history': self.history,
            'config': self.config
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def save_history(self):
        """Save training history to JSON."""
        history_file = self.output_dir / 'training_history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"Saved training history to {history_file}")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = self.history['epochs']
        
        # Loss plot
        ax1.plot(epochs, self.history['train_loss'], label='Train Loss', marker='o')
        ax1.plot(epochs, self.history['val_loss'], label='Val Loss', marker='s')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy plot
        ax2.plot(epochs, self.history['train_acc'], label='Train Acc', marker='o')
        ax2.plot(epochs, self.history['val_acc'], label='Val Acc', marker='s')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Mark best epoch
        ax2.axvline(x=self.best_epoch, color='r', linestyle='--', alpha=0.5, label=f'Best (epoch {self.best_epoch})')
        ax2.legend()
        
        plt.tight_layout()
        save_path = self.vis_dir / 'training_curves.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved training curves to {save_path}")
    
    @torch.no_grad()
    def visualize_attention(self, num_samples: int = 3):
        """Visualize attention patterns."""
        self.model.eval()
        
        # Get a batch
        sequences, labels = next(iter(self.val_loader))
        sequences = sequences[:num_samples].to(self.device)
        labels = labels[:num_samples]
        
        # Forward pass with attention
        logits, attn_weights = self.model(sequences, return_attention=True)
        
        if attn_weights is None:
            print("Attention weights not available for this model")
            return
        
        # Plot attention for each sample
        for i in range(num_samples):
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.flatten()
            
            # Plot each head
            for head in range(min(8, attn_weights.size(1))):
                ax = axes[head]
                attn = attn_weights[i, head].cpu().numpy()
                
                sns.heatmap(
                    attn,
                    ax=ax,
                    cmap='viridis',
                    xticklabels=False,
                    yticklabels=False,
                    cbar=True
                )
                ax.set_title(f'Head {head+1}')
                ax.set_xlabel('Key Position')
                ax.set_ylabel('Query Position')
            
            plt.suptitle(f'Attention Patterns - Sample {i+1} (Label: {labels[i].item()})')
            plt.tight_layout()
            
            save_path = self.vis_dir / f'attention_sample_{i+1}.png'
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        
        print(f"Saved attention visualizations to {self.vis_dir}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train positional encoding models')
    
    # Model arguments
    parser.add_argument('--pos_encoding', type=str, default='learned_absolute',
                        choices=['learned_absolute', 'learned_relative', 'continuous', 'sinusoidal', 'none'],
                        help='Type of positional encoding')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=3,
                        help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout probability')
    
    # Dataset arguments
    parser.add_argument('--dataset_type', type=str, default='pattern',
                        choices=['pattern', 'sorting', 'distance'],
                        help='Type of dataset')
    parser.add_argument('--seq_len', type=int, default=32,
                        help='Sequence length')
    parser.add_argument('--vocab_size', type=int, default=20,
                        help='Vocabulary size')
    parser.add_argument('--train_samples', type=int, default=5000,
                        help='Number of training samples')
    parser.add_argument('--val_samples', type=int, default=1000,
                        help='Number of validation samples')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=20,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                        help='Output directory')
    parser.add_argument('--visualize_attention', action='store_true',
                        help='Visualize attention patterns after training')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Create output directory
    exp_name = f"{args.pos_encoding}_d{args.d_model}_h{args.num_heads}_l{args.num_layers}"
    output_dir = Path(args.output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    config = vars(args)
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    # Create dataloaders
    print("Creating datasets...")
    train_loader, val_loader, dataset_info = create_dataloaders(
        dataset_type=args.dataset_type,
        batch_size=args.batch_size,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seq_len=args.seq_len,
        vocab_size=args.vocab_size,
        seed=args.seed
    )
    
    print(f"Dataset: {args.dataset_type}")
    print(f"  Train samples: {args.train_samples}")
    print(f"  Val samples: {args.val_samples}")
    print(f"  Num classes: {dataset_info['num_classes']}")
    
    # Create model
    print("\nCreating model...")
    model = create_model(
        vocab_size=args.vocab_size,
        num_classes=dataset_info['num_classes'],
        pos_encoding_type=args.pos_encoding,
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        max_seq_len=args.seq_len,
        dropout=args.dropout
    )
    
    device = torch.device(args.device)
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")
    
    # Create optimizer and criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        output_dir=output_dir,
        config=config
    )
    
    # Train
    history = trainer.train(num_epochs=args.num_epochs)
    
    # Visualize attention if requested
    if args.visualize_attention:
        print("\nVisualizing attention patterns...")
        trainer.visualize_attention(num_samples=3)
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Model: {args.pos_encoding}")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best epoch: {trainer.best_epoch}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
