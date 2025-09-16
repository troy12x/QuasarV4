#!/usr/bin/env python3
"""
Evolving Attention Transformer
=============================

A Transformer architecture with continuous-time evolving attention mechanisms
that maintains parallel efficiency while incorporating temporal dynamics.

Key Innovation: Attention weights evolve continuously over layers using differential equations,
but computation remains fully parallelizable.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

@dataclass
class EvolvingAttentionConfig:
    """Configuration for Evolving Attention Transformer"""
    vocab_size: int = 50257
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    max_position_embeddings: int = 2048
    
    # Evolving attention parameters
    evolution_rate: float = 0.1  # How fast attention evolves
    attention_memory_decay: float = 0.9  # Decay rate for attention memory
    temporal_smoothing: float = 0.1  # Smoothing factor for attention evolution
    
    # Standard parameters
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02

class EvolvingMultiHeadAttention(nn.Module):
    """
    Multi-head attention with evolving attention patterns.
    
    Key Innovation: Attention weights evolve based on:
    1. Previous layer's attention state (memory)
    2. Current input dynamics 
    3. Learned evolution parameters
    
    Maintains parallelizability by computing evolution for entire sequence at once.
    """
    
    def __init__(self, config: EvolvingAttentionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        
        assert self.hidden_size % self.num_heads == 0
        
        # Standard attention projections
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        
        # Evolving attention components
        self.attention_evolution = nn.Linear(config.hidden_size * 2, config.num_heads, bias=True)
        self.memory_gate = nn.Linear(config.hidden_size, config.num_heads, bias=True)
        self.temporal_dynamics = nn.Linear(config.hidden_size, config.num_heads, bias=True)
        
        # Learnable evolution parameters per head
        self.evolution_rate = nn.Parameter(torch.full((config.num_heads,), config.evolution_rate))
        self.memory_decay = nn.Parameter(torch.full((config.num_heads,), config.attention_memory_decay))
        
        # Attention memory buffer (will be passed between layers)
        self.register_buffer('attention_memory', torch.zeros(1, 1, config.num_heads, 1, 1))
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def evolve_attention_weights(self, 
                               attention_scores: torch.Tensor,
                               hidden_states: torch.Tensor,
                               attention_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Evolve attention weights based on temporal dynamics.
        
        Args:
            attention_scores: [batch, heads, seq_len, seq_len]
            hidden_states: [batch, seq_len, hidden_size]
            attention_memory: Previous layer's attention state
            
        Returns:
            evolved_attention: [batch, heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Compute temporal dynamics for each position
        temporal_signal = self.temporal_dynamics(hidden_states)  # [batch, seq_len, num_heads]
        temporal_signal = temporal_signal.transpose(1, 2).unsqueeze(-1)  # [batch, heads, seq_len, 1]
        
        # Memory gating - how much to retain from previous layers
        if attention_memory is not None and attention_memory.size(-1) == seq_len:
            # Compute memory influence
            memory_influence = torch.sigmoid(self.memory_gate(hidden_states))  # [batch, seq_len, heads]
            memory_influence = memory_influence.transpose(1, 2).unsqueeze(-1)  # [batch, heads, seq_len, 1]
            
            # Decay previous attention memory
            decayed_memory = attention_memory * self.memory_decay.view(1, -1, 1, 1)
            
            # Evolve attention: combine current scores with evolved memory
            evolution_factor = self.evolution_rate.view(1, -1, 1, 1)
            
            evolved_attention = (
                attention_scores * (1 - evolution_factor) +  # Current attention
                decayed_memory * evolution_factor * memory_influence +  # Memory influence
                temporal_signal * evolution_factor * 0.1  # Temporal dynamics
            )
        else:
            # First layer or size mismatch - just apply temporal dynamics
            evolution_factor = self.evolution_rate.view(1, -1, 1, 1)
            evolved_attention = attention_scores + temporal_signal * evolution_factor * 0.1
        
        return evolved_attention
    
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                attention_memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with evolving attention.
        
        Returns:
            output: Attention output
            new_attention_memory: Updated attention memory for next layer
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Standard Q, K, V projections
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Evolve attention weights based on temporal dynamics
        evolved_attention_scores = self.evolve_attention_weights(
            attention_scores, hidden_states, attention_memory
        )
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(evolved_attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, v)
        
        # Reshape and project output
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_size)
        attention_output = self.out_proj(attention_output)
        
        # Store attention weights as memory for next layer
        new_attention_memory = attention_weights.detach()
        
        return attention_output, new_attention_memory

class EvolvingTransformerBlock(nn.Module):
    """Transformer block with evolving attention"""
    
    def __init__(self, config: EvolvingAttentionConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        
        # Evolving attention
        self.attention = EvolvingMultiHeadAttention(config, layer_idx)
        
        # Feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_size, 4 * config.hidden_size),
            nn.GELU(),
            nn.Linear(4 * config.hidden_size, config.hidden_size),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                attention_memory: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Pre-norm attention with evolving dynamics
        normed_hidden_states = self.ln_1(hidden_states)
        attention_output, new_attention_memory = self.attention(
            normed_hidden_states, attention_mask, attention_memory
        )
        
        # Residual connection
        hidden_states = hidden_states + attention_output
        
        # Pre-norm MLP
        normed_hidden_states = self.ln_2(hidden_states)
        mlp_output = self.mlp(normed_hidden_states)
        
        # Residual connection
        hidden_states = hidden_states + mlp_output
        
        return hidden_states, new_attention_memory

class EvolvingAttentionTransformer(nn.Module):
    """
    Complete Evolving Attention Transformer model.
    
    Key Features:
    1. Attention weights evolve across layers using continuous-time dynamics
    2. Maintains full parallelizability (no sequential bottlenecks)
    3. Memory mechanism allows attention patterns to build up over layers
    4. Learnable evolution parameters adapt to different tasks
    """
    
    def __init__(self, config: EvolvingAttentionConfig):
        super().__init__()
        self.config = config
        
        # Embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        # Transformer layers with evolving attention
        self.layers = nn.ModuleList([
            EvolvingTransformerBlock(config, i) for i in range(config.num_layers)
        ])
        
        # Output
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize weights following GPT-2 style"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def create_attention_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Create causal attention mask"""
        batch_size, seq_len = input_ids.shape
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.float32))
        mask = mask.view(1, 1, seq_len, seq_len)
        mask = (1.0 - mask) * torch.finfo(torch.float32).min
        return mask
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through evolving attention transformer.
        
        The key innovation: attention patterns evolve across layers while
        maintaining full parallelizability.
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create embeddings
        token_emb = self.token_embedding(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(pos_ids)
        
        hidden_states = token_emb + pos_emb
        
        # Create attention mask
        attention_mask = self.create_attention_mask(input_ids)
        
        # Pass through layers with evolving attention
        attention_memory = None
        for layer in self.layers:
            hidden_states, attention_memory = layer(
                hidden_states, attention_mask, attention_memory
            )
        
        # Final layer norm and output projection
        hidden_states = self.ln_f(hidden_states)
        logits = self.lm_head(hidden_states)
        
        return logits
    
    def get_attention_evolution_stats(self) -> dict:
        """Get statistics about attention evolution parameters"""
        stats = {}
        for i, layer in enumerate(self.layers):
            stats[f'layer_{i}'] = {
                'evolution_rate': layer.attention.evolution_rate.detach().cpu().numpy(),
                'memory_decay': layer.attention.memory_decay.detach().cpu().numpy(),
            }
        return stats

def create_evolving_attention_model(
    vocab_size: int = 50257,
    hidden_size: int = 768,
    num_layers: int = 12,
    num_heads: int = 12,
    evolution_rate: float = 0.1,
    memory_decay: float = 0.9
) -> EvolvingAttentionTransformer:
    """Create an evolving attention transformer with specified parameters"""
    
    config = EvolvingAttentionConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_heads=num_heads,
        evolution_rate=evolution_rate,
        attention_memory_decay=memory_decay
    )
    
    return EvolvingAttentionTransformer(config)

# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_evolving_attention_model(
        vocab_size=1000,  # Small vocab for testing
        hidden_size=256,
        num_layers=6,
        num_heads=8,
        evolution_rate=0.15,
        memory_decay=0.85
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Input shape: {input_ids.shape}")
    
    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)
        print(f"Output shape: {logits.shape}")
    
    # Check attention evolution stats
    evolution_stats = model.get_attention_evolution_stats()
    print(f"\nAttention Evolution Stats:")
    for layer_name, stats in evolution_stats.items():
        print(f"{layer_name}: evolution_rate={stats['evolution_rate'].mean():.3f}, "
              f"memory_decay={stats['memory_decay'].mean():.3f}")
    
    print("\n✅ Evolving Attention Transformer created successfully!")
    print("Key features:")
    print("- Attention weights evolve across layers using continuous-time dynamics")
    print("- Fully parallelizable (no sequential bottlenecks like LNNs)")
    print("- Memory mechanism builds attention patterns over layers")
    print("- Learnable evolution parameters adapt to different tasks")
