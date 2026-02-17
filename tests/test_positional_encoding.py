"""
Comprehensive Test Suite for Positional Encoding Implementation

Tests all positional encoding methods, model components, and datasets.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
import numpy as np
from typing import List

from positional_encodings import (
    SinusoidalPositionalEncoding,
    LearnedAbsolutePositionalEncoding,
    LearnedRelativePositionalBias,
    ContinuousPositionalEncoding,
    create_positional_encoding
)
from model import (
    MultiHeadSelfAttention,
    TransformerBlock,
    PositionalTransformer,
    create_model
)
from dataset import (
    PositionAwarePatternDataset,
    PositionSortingDataset,
    PositionDistanceDataset,
    create_dataloaders
)


class TestSuite:
    """Comprehensive test suite."""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests_run = []
    
    def run_test(self, test_name: str, test_func, *args, **kwargs):
        """Run a single test."""
        try:
            test_func(*args, **kwargs)
            self.passed += 1
            self.tests_run.append((test_name, True, None))
            print(f"✓ {test_name}")
        except Exception as e:
            self.failed += 1
            self.tests_run.append((test_name, False, str(e)))
            print(f"✗ {test_name}: {e}")
    
    def print_summary(self):
        """Print test summary."""
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"Passed: {self.passed} ({100*self.passed/total:.1f}%)")
        print(f"Failed: {self.failed}")
        print(f"{'='*60}")
        
        if self.failed > 0:
            print("\nFailed tests:")
            for name, passed, error in self.tests_run:
                if not passed:
                    print(f"  - {name}: {error}")
        else:
            print("\n✓ ALL TESTS PASSED!")


# ============================================================================
# Positional Encoding Tests
# ============================================================================

def test_sinusoidal_shape():
    """Test sinusoidal encoding output shape."""
    d_model = 128
    batch_size, seq_len = 4, 32
    
    enc = SinusoidalPositionalEncoding(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    out = enc(x)
    
    assert out.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"


def test_learned_absolute_shape():
    """Test learned absolute encoding output shape."""
    d_model = 128
    batch_size, seq_len = 4, 32
    
    enc = LearnedAbsolutePositionalEncoding(d_model, max_len=512)
    x = torch.randn(batch_size, seq_len, d_model)
    out = enc(x)
    
    assert out.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"


def test_learned_relative_shape():
    """Test learned relative encoding output shape."""
    num_heads = 8
    seq_len = 32
    
    enc = LearnedRelativePositionalBias(num_heads, max_distance=128)
    bias = enc(seq_len)
    
    expected_shape = (num_heads, seq_len, seq_len)
    assert bias.shape == expected_shape, f"Expected {expected_shape}, got {bias.shape}"


def test_continuous_shape():
    """Test continuous encoding output shape."""
    d_model = 128
    batch_size, seq_len = 4, 32
    
    enc = ContinuousPositionalEncoding(d_model, hidden_dim=128)
    x = torch.randn(batch_size, seq_len, d_model)
    out = enc(x)
    
    assert out.shape == (batch_size, seq_len, d_model), f"Expected {(batch_size, seq_len, d_model)}, got {out.shape}"


def test_max_length_error():
    """Test that learned absolute raises error for sequences longer than max_len."""
    d_model = 128
    max_len = 16
    
    enc = LearnedAbsolutePositionalEncoding(d_model, max_len=max_len)
    x = torch.randn(2, max_len + 5, d_model)
    
    try:
        out = enc(x)
        raise AssertionError("Should have raised ValueError for sequence longer than max_len")
    except ValueError:
        pass  # Expected


def test_positional_encoding_gradient_flow():
    """Test that gradients flow through positional encodings."""
    d_model = 128
    batch_size, seq_len = 4, 32
    
    for enc_type in ['learned_absolute', 'continuous']:
        enc = create_positional_encoding(enc_type, d_model, dropout=0.0)
        x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None, f"Gradients not flowing for {enc_type}"


def test_create_factory():
    """Test positional encoding factory function."""
    d_model = 128
    
    # Test all types
    for enc_type in ['sinusoidal', 'learned_absolute', 'continuous', 'none']:
        enc = create_positional_encoding(enc_type, d_model)
        assert enc is not None, f"Failed to create {enc_type}"
    
    # Test learned_relative with num_heads
    enc = create_positional_encoding('learned_relative', d_model, num_heads=8)
    assert enc is not None, "Failed to create learned_relative"


# ============================================================================
# Model Tests
# ============================================================================

def test_multi_head_attention_shape():
    """Test multi-head attention output shape."""
    d_model = 128
    num_heads = 8
    batch_size, seq_len = 4, 32
    
    attn = MultiHeadSelfAttention(d_model, num_heads)
    x = torch.randn(batch_size, seq_len, d_model)
    out, weights = attn(x, return_attention=True)
    
    assert out.shape == (batch_size, seq_len, d_model), f"Output shape mismatch"
    assert weights.shape == (batch_size, num_heads, seq_len, seq_len), f"Attention weights shape mismatch"


def test_transformer_block_shape():
    """Test transformer block output shape."""
    d_model = 128
    num_heads = 8
    d_ff = 512
    batch_size, seq_len = 4, 32
    
    block = TransformerBlock(d_model, num_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    out, _ = block(x)
    
    assert out.shape == (batch_size, seq_len, d_model), f"Output shape mismatch"


def test_model_forward():
    """Test full model forward pass."""
    vocab_size = 20
    num_classes = 3
    batch_size, seq_len = 4, 32
    
    for pos_type in ['learned_absolute', 'learned_relative', 'continuous', 'sinusoidal', 'none']:
        model = create_model(
            vocab_size=vocab_size,
            num_classes=num_classes,
            pos_encoding_type=pos_type,
            d_model=128,
            num_heads=8,
            num_layers=3
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, _ = model(input_ids)
        
        expected_shape = (batch_size, num_classes)
        assert logits.shape == expected_shape, f"{pos_type}: Expected {expected_shape}, got {logits.shape}"


def test_model_backward():
    """Test that gradients flow through model."""
    vocab_size = 20
    num_classes = 3
    batch_size, seq_len = 4, 32
    
    model = create_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pos_encoding_type='learned_absolute',
        d_model=128,
        num_heads=8,
        num_layers=3
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    labels = torch.randint(0, num_classes, (batch_size,))
    
    logits, _ = model(input_ids)
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    
    # Check that some parameters have gradients
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            break
    
    assert has_grad, "No gradients found in model parameters"


def test_attention_mask():
    """Test attention mask functionality."""
    vocab_size = 20
    num_classes = 3
    batch_size, seq_len = 4, 32
    
    model = create_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pos_encoding_type='learned_absolute',
        d_model=128,
        num_heads=8,
        num_layers=2
    )
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Create mask (mask out second half)
    mask = torch.ones(batch_size, seq_len)
    mask[:, seq_len//2:] = 0
    
    logits, _ = model(input_ids, attention_mask=mask)
    
    assert logits.shape == (batch_size, num_classes), "Output shape with mask incorrect"


# ============================================================================
# Dataset Tests
# ============================================================================

def test_pattern_dataset_creation():
    """Test pattern dataset creation."""
    dataset = PositionAwarePatternDataset(
        num_samples=100,
        seq_len=32,
        vocab_size=20,
        num_classes=3,
        seed=42
    )
    
    assert len(dataset) == 100, f"Expected 100 samples, got {len(dataset)}"
    
    seq, label = dataset[0]
    assert seq.shape == (32,), f"Sequence shape mismatch"
    assert 0 <= label < 3, f"Label out of range: {label}"


def test_pattern_dataset_patterns():
    """Test that pattern dataset actually has the patterns."""
    dataset = PositionAwarePatternDataset(
        num_samples=300,
        seq_len=32,
        vocab_size=20,
        num_classes=3,
        seed=42
    )
    
    # Check that each sample matches its pattern
    for idx in range(min(10, len(dataset))):
        seq, label = dataset[idx]
        pattern = dataset.patterns[label.item()]
        
        for pos, val in pattern:
            if pos < len(seq):
                assert seq[pos].item() == val, f"Sample {idx}: Pattern mismatch at position {pos}"


def test_sorting_dataset():
    """Test sorting dataset creation."""
    dataset = PositionSortingDataset(
        num_samples=100,
        seq_len=32,
        vocab_size=20,
        seed=42
    )
    
    assert len(dataset) == 100, f"Expected 100 samples"
    
    seq, label = dataset[0]
    assert seq.shape == (32,), "Sequence shape mismatch"
    assert label in [0, 1], f"Label should be 0 or 1, got {label}"


def test_distance_dataset():
    """Test distance dataset creation."""
    dataset = PositionDistanceDataset(
        num_samples=100,
        seq_len=32,
        vocab_size=20,
        num_classes=5,
        seed=42
    )
    
    assert len(dataset) == 100, "Expected 100 samples"
    
    seq, label = dataset[0]
    assert seq.shape == (32,), "Sequence shape mismatch"
    assert 0 <= label < 5, f"Label out of range: {label}"


def test_create_dataloaders():
    """Test dataloader creation."""
    train_loader, val_loader, info = create_dataloaders(
        dataset_type='pattern',
        batch_size=8,
        train_samples=100,
        val_samples=20,
        seq_len=32,
        vocab_size=20,
        seed=42
    )
    
    # Check loaders exist
    assert train_loader is not None, "Train loader is None"
    assert val_loader is not None, "Val loader is None"
    
    # Check info
    assert 'num_classes' in info, "num_classes not in info"
    assert 'seq_len' in info, "seq_len not in info"
    
    # Get a batch
    sequences, labels = next(iter(train_loader))
    assert sequences.shape[0] <= 8, "Batch size too large"
    assert sequences.shape[1] == 32, "Sequence length mismatch"


# ============================================================================
# Integration Tests
# ============================================================================

def test_end_to_end_training_step():
    """Test a single training step end-to-end."""
    # Create data
    train_loader, val_loader, info = create_dataloaders(
        dataset_type='pattern',
        batch_size=8,
        train_samples=32,
        val_samples=16,
        seq_len=32,
        vocab_size=20,
        seed=42
    )
    
    # Create model
    model = create_model(
        vocab_size=20,
        num_classes=info['num_classes'],
        pos_encoding_type='learned_absolute',
        d_model=64,
        num_heads=4,
        num_layers=2
    )
    
    # Training step
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    sequences, labels = next(iter(train_loader))
    
    optimizer.zero_grad()
    logits, _ = model(sequences)
    loss = criterion(logits, labels)
    loss.backward()
    optimizer.step()
    
    assert loss.item() > 0, "Loss should be positive"


def test_different_batch_sizes():
    """Test model with different batch sizes."""
    vocab_size = 20
    num_classes = 3
    seq_len = 32
    
    model = create_model(
        vocab_size=vocab_size,
        num_classes=num_classes,
        pos_encoding_type='learned_absolute',
        d_model=128,
        num_heads=8,
        num_layers=2
    )
    
    for batch_size in [1, 4, 16]:
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        logits, _ = model(input_ids)
        assert logits.shape == (batch_size, num_classes), f"Failed for batch_size={batch_size}"


# ============================================================================
# Run All Tests
# ============================================================================

def main():
    """Run all tests."""
    print("="*60)
    print("RUNNING COMPREHENSIVE TEST SUITE")
    print("="*60)
    print()
    
    suite = TestSuite()
    
    print("Testing Positional Encodings...")
    print("-" * 60)
    suite.run_test("Sinusoidal shape", test_sinusoidal_shape)
    suite.run_test("Learned absolute shape", test_learned_absolute_shape)
    suite.run_test("Learned relative shape", test_learned_relative_shape)
    suite.run_test("Continuous shape", test_continuous_shape)
    suite.run_test("Max length error handling", test_max_length_error)
    suite.run_test("Gradient flow", test_positional_encoding_gradient_flow)
    suite.run_test("Factory function", test_create_factory)
    
    print("\nTesting Model Components...")
    print("-" * 60)
    suite.run_test("Multi-head attention shape", test_multi_head_attention_shape)
    suite.run_test("Transformer block shape", test_transformer_block_shape)
    suite.run_test("Model forward pass", test_model_forward)
    suite.run_test("Model backward pass", test_model_backward)
    suite.run_test("Attention mask", test_attention_mask)
    
    print("\nTesting Datasets...")
    print("-" * 60)
    suite.run_test("Pattern dataset creation", test_pattern_dataset_creation)
    suite.run_test("Pattern dataset patterns", test_pattern_dataset_patterns)
    suite.run_test("Sorting dataset", test_sorting_dataset)
    suite.run_test("Distance dataset", test_distance_dataset)
    suite.run_test("Dataloader creation", test_create_dataloaders)
    
    print("\nIntegration Tests...")
    print("-" * 60)
    suite.run_test("End-to-end training step", test_end_to_end_training_step)
    suite.run_test("Different batch sizes", test_different_batch_sizes)
    
    # Print summary
    suite.print_summary()
    
    return suite.failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
