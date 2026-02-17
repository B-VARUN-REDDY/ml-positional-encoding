"""
ML Positional Encoding Project

Learnable positional encoding methods for transformer models.
"""

__version__ = "1.0.0"
__author__ = "ML Engineering Interview Submission"

from .positional_encodings import (
    SinusoidalPositionalEncoding,
    LearnedAbsolutePositionalEncoding,
    LearnedRelativePositionalBias,
    ContinuousPositionalEncoding,
    create_positional_encoding
)

from .model import (
    MultiHeadSelfAttention,
    TransformerBlock,
    PositionalTransformer,
    create_model
)

from .dataset import (
    PositionAwarePatternDataset,
    PositionSortingDataset,
    PositionDistanceDataset,
    create_dataloaders
)

__all__ = [
    'SinusoidalPositionalEncoding',
    'LearnedAbsolutePositionalEncoding',
    'LearnedRelativePositionalBias',
    'ContinuousPositionalEncoding',
    'create_positional_encoding',
    'MultiHeadSelfAttention',
    'TransformerBlock',
    'PositionalTransformer',
    'create_model',
    'PositionAwarePatternDataset',
    'PositionSortingDataset',
    'PositionDistanceDataset',
    'create_dataloaders',
]
