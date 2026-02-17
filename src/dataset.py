"""
Position-Aware Datasets for Testing Positional Encodings

Creates synthetic datasets where positional information is critical for solving the task.
This allows us to evaluate how well different positional encoding methods work.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, List


class PositionAwarePatternDataset(Dataset):
    """
    Dataset where classification depends on specific values at specific positions.
    
    The model must learn positional information to solve this task.
    
    Task: Classify sequences based on value-position patterns:
    - Class 0: value=5 at position 3 AND value=8 at position 7
    - Class 1: value=3 at position 5 AND value=9 at position 10  
    - Class 2: value=7 at position 15 AND value=4 at position 20
    
    Args:
        num_samples: Number of samples to generate
        seq_len: Sequence length
        vocab_size: Size of vocabulary (range of values)
        num_classes: Number of classes
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        num_samples: int = 5000,
        seq_len: int = 32,
        vocab_size: int = 20,
        num_classes: int = 3,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Define patterns for each class
        # Format: [(position, value), (position, value)]
        self.patterns = [
            [(3, 5), (7, 8)],    # Class 0
            [(5, 3), (10, 9)],   # Class 1
            [(15, 7), (20, 4)]   # Class 2
        ]
        
        # Generate data
        self.sequences, self.labels = self._generate_data()
    
    def _check_pattern(self, sequence: np.ndarray, pattern: List[Tuple[int, int]]) -> bool:
        """
        Check if sequence matches a pattern.
        
        Args:
            sequence: Sequence array
            pattern: List of (position, value) tuples
        
        Returns:
            True if all position-value pairs match
        """
        for pos, val in pattern:
            if pos >= len(sequence):
                return False
            if sequence[pos] != val:
                return False
        return True
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate sequences and labels.
        
        Returns:
            sequences: [num_samples, seq_len] tensor of token IDs
            labels: [num_samples] tensor of class labels
        """
        sequences = []
        labels = []
        
        # Calculate samples per class
        samples_per_class = self.num_samples // self.num_classes
        
        for class_idx in range(self.num_classes):
            pattern = self.patterns[class_idx]
            class_samples = 0
            
            while class_samples < samples_per_class:
                # Generate random sequence
                seq = np.random.randint(0, self.vocab_size, size=self.seq_len)
                
                # Check if it matches any pattern (avoid ambiguous samples)
                matches_any = any(
                    self._check_pattern(seq, self.patterns[i])
                    for i in range(self.num_classes)
                )
                
                if matches_any:
                    # If it matches this class pattern, use it
                    if self._check_pattern(seq, pattern):
                        sequences.append(seq)
                        labels.append(class_idx)
                        class_samples += 1
                else:
                    # If it doesn't match any pattern, inject the pattern
                    for pos, val in pattern:
                        if pos < self.seq_len:
                            seq[pos] = val
                    
                    sequences.append(seq)
                    labels.append(class_idx)
                    class_samples += 1
        
        # Convert to tensors
        sequences = torch.tensor(np.array(sequences), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Shuffle
        perm = torch.randperm(len(sequences))
        sequences = sequences[perm]
        labels = labels[perm]
        
        return sequences, labels
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]
    
    def get_pattern_info(self) -> Dict:
        """Get information about the patterns."""
        return {
            'num_classes': self.num_classes,
            'patterns': self.patterns,
            'seq_len': self.seq_len,
            'vocab_size': self.vocab_size
        }


class PositionSortingDataset(Dataset):
    """
    Dataset where the task is to predict if a sequence is sorted.
    
    This requires understanding relative positions.
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        vocab_size: Size of vocabulary
        seed: Random seed
    """
    
    def __init__(
        self,
        num_samples: int = 5000,
        seq_len: int = 32,
        vocab_size: int = 20,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.sequences, self.labels = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences and labels."""
        sequences = []
        labels = []
        
        for _ in range(self.num_samples):
            # Random choice: sorted or unsorted
            is_sorted = np.random.rand() > 0.5
            
            if is_sorted:
                # Generate sorted sequence
                seq = np.sort(np.random.randint(0, self.vocab_size, size=self.seq_len))
                label = 1
            else:
                # Generate random (likely unsorted) sequence
                seq = np.random.randint(0, self.vocab_size, size=self.seq_len)
                # Make sure it's not accidentally sorted
                if np.all(seq[:-1] <= seq[1:]):
                    # Swap two elements to unsort
                    idx = np.random.randint(0, self.seq_len - 1)
                    seq[idx], seq[idx + 1] = seq[idx + 1], seq[idx]
                label = 0
            
            sequences.append(seq)
            labels.append(label)
        
        sequences = torch.tensor(np.array(sequences), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return sequences, labels
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class PositionDistanceDataset(Dataset):
    """
    Dataset where the task is to predict the distance between two specific values.
    
    Args:
        num_samples: Number of samples
        seq_len: Sequence length
        vocab_size: Size of vocabulary
        target_value1: First target value to find
        target_value2: Second target value to find
        num_classes: Number of distance classes
        seed: Random seed
    """
    
    def __init__(
        self,
        num_samples: int = 5000,
        seq_len: int = 32,
        vocab_size: int = 20,
        target_value1: int = 5,
        target_value2: int = 8,
        num_classes: int = 5,
        seed: Optional[int] = None
    ):
        super().__init__()
        
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.target_value1 = target_value1
        self.target_value2 = target_value2
        self.num_classes = num_classes
        
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        self.sequences, self.labels = self._generate_data()
    
    def _generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate sequences and labels."""
        sequences = []
        labels = []
        
        for _ in range(self.num_samples):
            # Generate random sequence
            seq = np.random.randint(0, self.vocab_size, size=self.seq_len)
            
            # Place target values
            pos1 = np.random.randint(0, self.seq_len)
            pos2 = np.random.randint(0, self.seq_len)
            
            seq[pos1] = self.target_value1
            seq[pos2] = self.target_value2
            
            # Calculate distance class
            distance = abs(pos2 - pos1)
            # Bin distance into classes
            distance_class = min(distance // (self.seq_len // self.num_classes), self.num_classes - 1)
            
            sequences.append(seq)
            labels.append(distance_class)
        
        sequences = torch.tensor(np.array(sequences), dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        
        return sequences, labels
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


def create_dataloaders(
    dataset_type: str = 'pattern',
    batch_size: int = 32,
    train_samples: int = 5000,
    val_samples: int = 1000,
    seq_len: int = 32,
    vocab_size: int = 20,
    num_workers: int = 0,
    seed: Optional[int] = 42
) -> Tuple[DataLoader, DataLoader, Dict]:
    """
    Create train and validation dataloaders.
    
    Args:
        dataset_type: Type of dataset ('pattern', 'sorting', 'distance')
        batch_size: Batch size
        train_samples: Number of training samples
        val_samples: Number of validation samples
        seq_len: Sequence length
        vocab_size: Size of vocabulary
        num_workers: Number of dataloader workers
        seed: Random seed
    
    Returns:
        train_loader: Training dataloader
        val_loader: Validation dataloader
        info: Dataset information dictionary
    """
    dataset_type = dataset_type.lower()
    
    # Create datasets
    if dataset_type == 'pattern':
        train_dataset = PositionAwarePatternDataset(
            num_samples=train_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=3,
            seed=seed
        )
        val_dataset = PositionAwarePatternDataset(
            num_samples=val_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=3,
            seed=seed + 1 if seed is not None else None
        )
        num_classes = 3
        info = train_dataset.get_pattern_info()
    
    elif dataset_type == 'sorting':
        train_dataset = PositionSortingDataset(
            num_samples=train_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=seed
        )
        val_dataset = PositionSortingDataset(
            num_samples=val_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            seed=seed + 1 if seed is not None else None
        )
        num_classes = 2
        info = {'num_classes': 2, 'task': 'sorting'}
    
    elif dataset_type == 'distance':
        train_dataset = PositionDistanceDataset(
            num_samples=train_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=5,
            seed=seed
        )
        val_dataset = PositionDistanceDataset(
            num_samples=val_samples,
            seq_len=seq_len,
            vocab_size=vocab_size,
            num_classes=5,
            seed=seed + 1 if seed is not None else None
        )
        num_classes = 5
        info = {'num_classes': 5, 'task': 'distance'}
    
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    info.update({
        'train_samples': train_samples,
        'val_samples': val_samples,
        'seq_len': seq_len,
        'vocab_size': vocab_size,
        'batch_size': batch_size,
        'num_classes': num_classes
    })
    
    return train_loader, val_loader, info


if __name__ == "__main__":
    # Quick test
    print("Testing datasets...")
    
    # Test pattern dataset
    train_loader, val_loader, info = create_dataloaders(
        dataset_type='pattern',
        batch_size=8,
        train_samples=100,
        val_samples=20
    )
    
    print(f"\nDataset info: {info}")
    
    # Get a batch
    sequences, labels = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"  Sequences: {sequences.shape}")
    print(f"  Labels: {labels.shape}")
    print(f"  Label distribution: {torch.bincount(labels)}")
    
    # Check pattern
    dataset = train_loader.dataset
    print(f"\nPatterns:")
    for i, pattern in enumerate(dataset.patterns):
        print(f"  Class {i}: {pattern}")
    
    # Verify a sample
    seq, label = dataset[0]
    print(f"\nSample 0:")
    print(f"  Label: {label}")
    print(f"  Sequence: {seq.tolist()}")
    pattern = dataset.patterns[label.item()]
    print(f"  Expected pattern: {pattern}")
    for pos, val in pattern:
        print(f"    Position {pos}: expected {val}, got {seq[pos].item()}")
    
    print("\n✓ Datasets working!")
