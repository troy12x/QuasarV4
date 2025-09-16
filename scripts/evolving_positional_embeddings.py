#!/usr/bin/env python3
"""
Evolving Positional Embeddings
==============================

Dynamic positional embeddings that evolve with sequence length and content.
No fixed tables - everything is generated on-the-fly.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple

class EvolvingPositionalEmbedding(nn.Module):
    """
    Evolving positional embeddings: E(i, L, content) = g(i, L, h_tokens)
    
    Features:
    - No maximum length limit
    - Continuous interpolation 
    - Content-aware positioning
    - Adaptive to sequence structure
    """
    
    def __init__(self, d_model: int, max_wavelength: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.max_wavelength = max_wavelength
        
        # Content-aware position generator
        self.content_encoder = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Sequence-aware position generator  
        self.sequence_encoder = nn.Sequential(
            nn.Linear(3, d_model // 2),  # [relative_pos, seq_len, content_influence]
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # Adaptive frequency generator (evolves with sequence)
        self.frequency_generator = nn.Sequential(
            nn.Linear(2, d_model // 4),  # [seq_len, content_variance]
            nn.GELU(),
            nn.Linear(d_model // 4, d_model // 2)
        )
        
        # Fusion layer
        self.position_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, token_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Generate evolving positional embeddings
        
        Args:
            token_embeddings: [batch, seq_len, d_model]
            
        Returns:
            positional_embeddings: [batch, seq_len, d_model]
        """
        batch_size, seq_len, d_model = token_embeddings.shape
        device = token_embeddings.device
        
        # === STEP 1: CONTENT-AWARE FEATURES ===
        # Encode content information for each token
        content_features = self.content_encoder(token_embeddings)  # [batch, seq_len, d_model]
        
        # Global content statistics
        content_mean = token_embeddings.mean(dim=1, keepdim=True)  # [batch, 1, d_model]
        content_variance = token_embeddings.var(dim=1, keepdim=True)  # [batch, 1, d_model]
        
        # === STEP 2: SEQUENCE-AWARE FEATURES ===
        # Relative positions (normalized by sequence length)
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        relative_positions = positions / seq_len  # [0, 1] range
        
        # Content influence at each position
        content_influence = torch.norm(token_embeddings, dim=-1) / torch.norm(token_embeddings, dim=-1).max(dim=1, keepdim=True)[0]
        
        # Sequence features for each position
        seq_len_tensor = torch.full((batch_size, seq_len, 1), seq_len, device=device, dtype=torch.float32)
        relative_pos_tensor = relative_positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)
        content_influence_tensor = content_influence.unsqueeze(-1)
        
        sequence_input = torch.cat([
            relative_pos_tensor,
            seq_len_tensor / 1000.0,  # Normalize sequence length
            content_influence_tensor
        ], dim=-1)  # [batch, seq_len, 3]
        
        sequence_features = self.sequence_encoder(sequence_input)  # [batch, seq_len, d_model]
        
        # === STEP 3: ADAPTIVE FREQUENCIES ===
        # Generate frequencies that adapt to sequence characteristics
        seq_stats = torch.cat([
            torch.full((batch_size, 1), seq_len / 1000.0, device=device),  # Normalized seq length
            content_variance.mean(dim=-1)  # Content complexity
        ], dim=-1)  # [batch, 2]
        
        adaptive_freqs = self.frequency_generator(seq_stats)  # [batch, d_model//2]
        
        # Create evolving sinusoidal patterns
        freq_indices = torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        base_freqs = 1.0 / (self.max_wavelength ** (freq_indices / d_model))
        
        # Modulate base frequencies with adaptive component
        evolved_freqs = base_freqs.unsqueeze(0) * (1.0 + 0.1 * adaptive_freqs)  # [batch, d_model//2]
        
        # Generate sinusoidal embeddings with evolved frequencies
        pos_args = relative_pos_tensor * evolved_freqs.unsqueeze(1)  # [batch, seq_len, d_model//2]
        
        sinusoidal_pos = torch.zeros(batch_size, seq_len, d_model, device=device)
        sinusoidal_pos[:, :, 0::2] = torch.sin(pos_args)
        sinusoidal_pos[:, :, 1::2] = torch.cos(pos_args)
        
        # === STEP 4: FUSION ===
        # Combine content-aware and sequence-aware features
        combined_features = torch.cat([content_features, sequence_features], dim=-1)
        fused_positions = self.position_fusion(combined_features)
        
        # Final evolving positional embedding
        evolving_embeddings = fused_positions + sinusoidal_pos
        
        return evolving_embeddings

class EvolvingMultiHeadAttention(nn.Module):
    """Multi-head attention with evolving positional embeddings"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        # Evolving positional embeddings
        self.evolving_pos = EvolvingPositionalEmbedding(d_model)
        
        # Standard attention projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                causal_mask: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with evolving positional embeddings
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_mask: Optional attention mask
            causal_mask: Whether to apply causal masking
            
        Returns:
            output: [batch, seq_len, d_model]
            attention_weights: [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # === STEP 1: GENERATE EVOLVING POSITIONAL EMBEDDINGS ===
        pos_embeddings = self.evolving_pos(hidden_states)
        
        # Add positional information to tokens
        enhanced_states = hidden_states + pos_embeddings
        
        # === STEP 2: STANDARD MULTI-HEAD ATTENTION ===
        # Project to Q, K, V
        q = self.q_proj(enhanced_states)
        k = self.k_proj(enhanced_states)
        v = self.v_proj(enhanced_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply masks
        if causal_mask:
            causal_mask_tensor = torch.triu(
                torch.ones(seq_len, seq_len, device=hidden_states.device), diagonal=1
            ).bool()
            attention_scores.masked_fill_(causal_mask_tensor, float('-inf'))
        
        if attention_mask is not None:
            attention_scores += attention_mask
        
        # Apply attention
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, d_model)
        output = self.out_proj(attention_output)
        
        return output, attention_weights

class EvolvingTransformerLayer(nn.Module):
    """Complete transformer layer with evolving positional embeddings"""
    
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.attention = EvolvingMultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention with evolving positions
        attn_output, _ = self.attention(hidden_states)
        hidden_states = self.norm1(hidden_states + attn_output)
        
        # Feed forward
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)
        
        return hidden_states

# Test the evolving embeddings
if __name__ == "__main__":
    print("🌟 Testing Evolving Positional Embeddings")
    
    # Test configuration
    batch_size = 4
    d_model = 256
    num_heads = 8
    
    # Test different sequence lengths
    for seq_len in [16, 64, 128, 512, 1024]:
        print(f"\n📏 Testing sequence length: {seq_len}")
        
        # Random token embeddings
        token_embeddings = torch.randn(batch_size, seq_len, d_model)
        
        # Create evolving positional embeddings
        evolving_pos = EvolvingPositionalEmbedding(d_model)
        pos_embeddings = evolving_pos(token_embeddings)
        
        print(f"   Token embeddings: {token_embeddings.shape}")
        print(f"   Positional embeddings: {pos_embeddings.shape}")
        
        # Test attention layer
        attention_layer = EvolvingMultiHeadAttention(d_model, num_heads)
        output, attn_weights = attention_layer(token_embeddings)
        
        print(f"   Attention output: {output.shape}")
        print(f"   Attention weights: {attn_weights.shape}")
        
        # Test full transformer layer
        transformer_layer = EvolvingTransformerLayer(d_model, num_heads, d_model * 4)
        layer_output = transformer_layer(token_embeddings)
        
        print(f"   Transformer output: {layer_output.shape}")
        print(f"   ✅ Success!")
    
    print(f"\n🎉 Evolving positional embeddings working for all sequence lengths!")
    print(f"🌟 No fixed tables - everything evolves dynamically!")
