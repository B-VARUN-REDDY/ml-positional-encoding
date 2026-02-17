"""
Learnable Positional Encoding Methods for Transformers

This module implements various positional encoding strategies:
1. Learned Absolute Positional Embeddings (BERT-style)
2. Learned Relative Position Bias (T5-style)
3. Continuous Positional Encoding with MLPs
4. Sinusoidal Positional Encoding (baseline)
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding from 'Attention Is All You Need'.
    
    Uses sine and cosine functions of different frequencies to encode positions.
    This is the baseline method - no learnable parameters.
    
    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length to pre-compute
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but part of state)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class LearnedAbsolutePositionalEncoding(nn.Module):
    """
    Learned absolute positional embeddings (BERT-style).
    
    Each position has a unique learnable embedding vector.
    Simple and effective, but cannot generalize beyond max_len.
    
    Args:
        d_model: Dimension of the model embeddings
        max_len: Maximum sequence length (fixed)
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # Learnable position embeddings
        self.position_embeddings = nn.Embedding(max_len, d_model)
        
        # Initialize with small values
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        if seq_len > self.max_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds maximum length {self.max_len}"
            )
        
        # Create position indices
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)  # [1, seq_len]
        
        # Get position embeddings
        pos_embeddings = self.position_embeddings(positions)  # [1, seq_len, d_model]
        
        # Add to input
        x = x + pos_embeddings
        return self.dropout(x)


class LearnedRelativePositionalBias(nn.Module):
    """
    Learned relative position bias (T5-style).
    
    Instead of adding to embeddings, this adds learnable biases to attention scores
    based on relative distances between positions. Better generalization to unseen lengths.
    
    Args:
        num_heads: Number of attention heads
        max_distance: Maximum relative distance to consider
        bidirectional: Whether to distinguish forward/backward relative positions
    """
    
    def __init__(
        self, 
        num_heads: int, 
        max_distance: int = 128,
        bidirectional: bool = True
    ):
        super().__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.bidirectional = bidirectional
        
        # Number of relative position buckets
        if bidirectional:
            num_buckets = 2 * max_distance + 1  # -max_distance to +max_distance
        else:
            num_buckets = max_distance + 1  # 0 to max_distance
        
        # Learnable relative attention bias
        self.relative_attention_bias = nn.Embedding(num_buckets, num_heads)
        
        # Initialize
        nn.init.normal_(self.relative_attention_bias.weight, mean=0.0, std=0.02)
    
    def _get_relative_position_bucket(
        self, 
        relative_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert relative positions to bucket indices.
        
        Args:
            relative_position: Relative position matrix [seq_len, seq_len]
        
        Returns:
            Bucket indices [seq_len, seq_len]
        """
        if self.bidirectional:
            # Clip to [-max_distance, max_distance]
            relative_buckets = torch.clamp(
                relative_position,
                -self.max_distance,
                self.max_distance
            )
            # Shift to [0, 2*max_distance]
            relative_buckets = relative_buckets + self.max_distance
        else:
            # Clip to [0, max_distance]
            relative_buckets = torch.clamp(
                relative_position,
                0,
                self.max_distance
            )
        
        return relative_buckets
    
    def forward(self, seq_len: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Compute relative position bias matrix.
        
        Args:
            seq_len: Sequence length
            device: Device to create tensor on
        
        Returns:
            Bias matrix [num_heads, seq_len, seq_len]
        """
        if device is None:
            device = self.relative_attention_bias.weight.device
        
        # Create relative position matrix
        # relative_position[i, j] = i - j
        positions = torch.arange(seq_len, device=device)
        relative_position = positions.unsqueeze(0) - positions.unsqueeze(1)  # [seq_len, seq_len]
        
        # Convert to bucket indices
        relative_buckets = self._get_relative_position_bucket(relative_position)
        
        # Get bias values
        bias = self.relative_attention_bias(relative_buckets)  # [seq_len, seq_len, num_heads]
        
        # Transpose to [num_heads, seq_len, seq_len]
        bias = bias.permute(2, 0, 1)
        
        return bias


class ContinuousPositionalEncoding(nn.Module):
    """
    Continuous positional encoding using MLPs.
    
    Maps normalized position indices through a neural network to generate
    position embeddings. Can theoretically handle arbitrary sequence lengths.
    
    Args:
        d_model: Dimension of the model embeddings
        hidden_dim: Hidden dimension of the MLP
        dropout: Dropout probability
        max_len: Maximum length for normalization (used for scaling)
    """
    
    def __init__(
        self, 
        d_model: int, 
        hidden_dim: int = 128,
        dropout: float = 0.1,
        max_len: int = 512
    ):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # MLP to transform position index to embedding
        self.position_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, d_model)
        )
        
        # Initialize
        for layer in self.position_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Tensor with positional encoding added [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, _ = x.size()
        
        # Create normalized position indices [0, 1]
        positions = torch.arange(seq_len, device=x.device, dtype=torch.float32)
        positions = positions / self.max_len  # Normalize
        positions = positions.unsqueeze(0).unsqueeze(2)  # [1, seq_len, 1]
        
        # Generate position embeddings through MLP
        pos_embeddings = self.position_mlp(positions)  # [1, seq_len, d_model]
        
        # Add to input
        x = x + pos_embeddings
        return self.dropout(x)


def create_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_len: int = 512,
    dropout: float = 0.1,
    num_heads: Optional[int] = None,
    **kwargs
) -> nn.Module:
    """
    Factory function to create positional encoding modules.
    
    Args:
        encoding_type: Type of encoding ('sinusoidal', 'learned_absolute', 
                       'learned_relative', 'continuous', 'none')
        d_model: Model dimension
        max_len: Maximum sequence length
        dropout: Dropout probability
        num_heads: Number of attention heads (required for learned_relative)
        **kwargs: Additional arguments for specific encoding types
    
    Returns:
        Positional encoding module
    """
    encoding_type = encoding_type.lower()
    
    if encoding_type == 'sinusoidal':
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)
    
    elif encoding_type == 'learned_absolute':
        return LearnedAbsolutePositionalEncoding(d_model, max_len, dropout)
    
    elif encoding_type == 'learned_relative':
        if num_heads is None:
            raise ValueError("num_heads is required for learned_relative encoding")
        max_distance = kwargs.get('max_distance', 128)
        bidirectional = kwargs.get('bidirectional', True)
        return LearnedRelativePositionalBias(num_heads, max_distance, bidirectional)
    
    elif encoding_type == 'continuous':
        hidden_dim = kwargs.get('hidden_dim', 128)
        return ContinuousPositionalEncoding(d_model, hidden_dim, dropout, max_len)
    
    elif encoding_type == 'none':
        # Identity module (no positional encoding)
        return nn.Identity()
    
    else:
        raise ValueError(
            f"Unknown encoding type: {encoding_type}. "
            f"Choose from: sinusoidal, learned_absolute, learned_relative, continuous, none"
        )


if __name__ == "__main__":
    # Quick test
    print("Testing Positional Encoding Modules...")
    
    batch_size, seq_len, d_model = 4, 32, 128
    num_heads = 8
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test sinusoidal
    sin_enc = SinusoidalPositionalEncoding(d_model)
    out = sin_enc(x)
    print(f"✓ Sinusoidal: {out.shape}")
    
    # Test learned absolute
    abs_enc = LearnedAbsolutePositionalEncoding(d_model, max_len=512)
    out = abs_enc(x)
    print(f"✓ Learned Absolute: {out.shape}")
    
    # Test learned relative
    rel_enc = LearnedRelativePositionalBias(num_heads)
    bias = rel_enc(seq_len, x.device)
    print(f"✓ Learned Relative: {bias.shape}")
    
    # Test continuous
    cont_enc = ContinuousPositionalEncoding(d_model)
    out = cont_enc(x)
    print(f"✓ Continuous: {out.shape}")
    
    print("\n✓ All positional encoding modules working!")
