"""
Transformer Model with Pluggable Positional Encodings

Implements a transformer-based sequence classifier that can use different
positional encoding strategies. Designed for evaluating positional encoding methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math

from positional_encodings import (
    create_positional_encoding,
    LearnedRelativePositionalBias
)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        dropout: Dropout probability
        use_relative_bias: Whether to use relative position bias
        relative_bias_module: Optional pre-created relative bias module
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_relative_bias: bool = False,
        relative_bias_module: Optional[LearnedRelativePositionalBias] = None
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_relative_bias = use_relative_bias
        
        # Linear projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        
        # Relative position bias (optional)
        self.relative_bias = relative_bias_module
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.zeros_(self.v_proj.bias)
        nn.init.zeros_(self.out_proj.bias)
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask [batch_size, seq_len] or [batch_size, seq_len, seq_len]
            return_attention: Whether to return attention weights
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
            Optional attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.size()
        
        # Linear projections and reshape to [batch_size, num_heads, seq_len, d_k]
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        # scores: [batch_size, num_heads, seq_len, seq_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Add relative position bias if using
        if self.use_relative_bias and self.relative_bias is not None:
            # Get relative position bias [num_heads, seq_len, seq_len]
            rel_bias = self.relative_bias(seq_len, device=x.device)
            # Expand for batch: [1, num_heads, seq_len, seq_len]
            rel_bias = rel_bias.unsqueeze(0)
            scores = scores + rel_bias
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                # Expand mask: [batch_size, 1, 1, seq_len]
                mask = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        
        if return_attention:
            return output, attn_weights
        return output, None


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    
    Args:
        d_model: Model dimension
        d_ff: Feed-forward dimension
        dropout: Dropout probability
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        x = self.linear1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single transformer block with self-attention and feed-forward.
    
    Args:
        d_model: Model dimension
        num_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout probability
        use_relative_bias: Whether to use relative position bias
        relative_bias_module: Optional pre-created relative bias module
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        use_relative_bias: bool = False,
        relative_bias_module: Optional[LearnedRelativePositionalBias] = None
    ):
        super().__init__()
        
        # Self-attention
        self.self_attn = MultiHeadSelfAttention(
            d_model, num_heads, dropout, use_relative_bias, relative_bias_module
        )
        
        # Feed-forward
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
        
        Returns:
            Output tensor [batch_size, seq_len, d_model]
            Optional attention weights
        """
        # Self-attention with residual
        attn_output, attn_weights = self.self_attn(
            self.norm1(x), mask, return_attention
        )
        x = x + attn_output
        
        # Feed-forward with residual
        ff_output = self.feed_forward(self.norm2(x))
        x = x + ff_output
        
        return x, attn_weights


class PositionalTransformer(nn.Module):
    """
    Transformer model with configurable positional encoding.
    
    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of output classes
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward dimension  
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pos_encoding_type: Type of positional encoding
        pad_token_id: ID of padding token
    """
    
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        d_ff: int = 512,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pos_encoding_type: str = 'learned_absolute',
        pad_token_id: int = 0
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.pos_encoding_type = pos_encoding_type
        self.pad_token_id = pad_token_id
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        # Special handling for relative position bias
        if pos_encoding_type == 'learned_relative':
            self.pos_encoding = create_positional_encoding(
                pos_encoding_type, d_model, max_seq_len, dropout, num_heads
            )
            use_relative_bias = True
            relative_bias_module = self.pos_encoding
        else:
            self.pos_encoding = create_positional_encoding(
                pos_encoding_type, d_model, max_seq_len, dropout
            )
            use_relative_bias = False
            relative_bias_module = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model, num_heads, d_ff, dropout,
                use_relative_bias, relative_bias_module
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters."""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            return_attention: Whether to return attention weights
        
        Returns:
            Logits [batch_size, num_classes]
            Optional attention weights from last layer
        """
        # Token embeddings
        x = self.token_embedding(input_ids)  # [batch_size, seq_len, d_model]
        
        # Add positional encoding
        # Note: For relative bias, this is a no-op (Identity module)
        if self.pos_encoding_type != 'learned_relative':
            x = self.pos_encoding(x)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()
        
        # Apply transformer blocks
        attn_weights = None
        for block in self.blocks:
            x, attn_weights = block(x, attention_mask, return_attention)
        
        # Final layer norm
        x = self.norm(x)
        
        # Global average pooling over sequence dimension
        # Mask out padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()
        x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        
        # Classification
        logits = self.classifier(x)
        
        return logits, attn_weights


def create_model(
    vocab_size: int,
    num_classes: int,
    pos_encoding_type: str = 'learned_absolute',
    d_model: int = 128,
    num_heads: int = 8,
    num_layers: int = 3,
    d_ff: Optional[int] = None,
    max_seq_len: int = 512,
    dropout: float = 0.1,
    pad_token_id: int = 0
) -> PositionalTransformer:
    """
    Factory function to create a PositionalTransformer model.
    
    Args:
        vocab_size: Size of vocabulary
        num_classes: Number of output classes
        pos_encoding_type: Type of positional encoding
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer blocks
        d_ff: Feed-forward dimension (default: 4 * d_model)
        max_seq_len: Maximum sequence length
        dropout: Dropout probability
        pad_token_id: ID of padding token
    
    Returns:
        PositionalTransformer model
    """
    if d_ff is None:
        d_ff = 4 * d_model
    
    model = PositionalTransformer(
        vocab_size=vocab_size,
        num_classes=num_classes,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        pos_encoding_type=pos_encoding_type,
        pad_token_id=pad_token_id
    )
    
    return model


if __name__ == "__main__":
    # Quick test
    print("Testing PositionalTransformer...")
    
    batch_size, seq_len = 4, 32
    vocab_size, num_classes = 20, 3
    
    # Create dummy input
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Test with different positional encodings
    for pos_type in ['learned_absolute', 'learned_relative', 'continuous', 'sinusoidal', 'none']:
        model = create_model(
            vocab_size=vocab_size,
            num_classes=num_classes,
            pos_encoding_type=pos_type,
            d_model=128,
            num_heads=8,
            num_layers=3
        )
        
        logits, _ = model(input_ids)
        print(f"✓ {pos_type:20s}: logits shape = {logits.shape}")
    
    print("\n✓ All models working!")
