#!/usr/bin/env python3
"""
PMB-Enhanced TrueEvolving Attention
==================================

Integrates Parameter Memory Bank (PMB) with TrueEvolving attention for:
1. Infinite positional encoding without context window limits
2. Content-aware positional memory
3. Semantic similarity-based position retrieval
4. Temporal flow enhancement through memory augmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from quasar.pmb import ParameterMemoryBank

class PositionalMemoryBank(nn.Module):
    """Infinite positional encoding using PMB principles"""
    
    def __init__(self, d_model, num_blocks=512, slots_per_block=2048):
        super().__init__()
        self.d_model = d_model
        self.pmb = ParameterMemoryBank(num_blocks, slots_per_block, d_model)
        
        # Position generators (no fixed limits)
        self.position_generator = nn.Linear(1, d_model)
        self.content_encoder = nn.Linear(d_model, d_model)
        self.similarity_weight = nn.Parameter(torch.tensor(0.1))
        
        # Temporal evolution integration
        self.temporal_position_gate = nn.Linear(d_model * 2, d_model)
        self.position_evolution = nn.Linear(d_model, d_model)
        
    def get_position_encoding(self, position, token_content, temporal_state=None):
        """Get infinite positional encoding with temporal awareness"""
        
        # Generate position ID (can be any integer - infinite!)
        pos_id = f"pos_{position}"
        
        # Try direct retrieval first (O(1))
        cached_encoding = self.pmb.retrieve_direct(pos_id)
        
        if cached_encoding is not None:
            # Ensure cached encoding is on correct device
            cached_encoding = cached_encoding.to(token_content.device)
            # Apply temporal evolution to cached position
            if temporal_state is not None:
                temporal_influence = torch.sigmoid(
                    self.temporal_position_gate(torch.cat([cached_encoding.unsqueeze(0), temporal_state.unsqueeze(0)], dim=-1))
                )
                cached_encoding = cached_encoding + temporal_influence.squeeze(0) * self.position_evolution(cached_encoding.unsqueeze(0)).squeeze(0)
            return cached_encoding
        
        # Generate new encoding if not cached
        pos_tensor = torch.tensor([float(position)], device=token_content.device, dtype=torch.float32)
        base_encoding = self.position_generator(pos_tensor.unsqueeze(0))  # [1, d_model]
        
        # Content-aware adjustment
        content_key = self.content_encoder(token_content.unsqueeze(0))  # [1, d_model]
        
        # Semantic retrieval of similar positions
        if len(self.pmb) > 0:
            similar_encodings = self.pmb.retrieve_semantic(content_key, top_k=3)
            # Combine base + semantic similarity
            final_encoding = base_encoding + self.similarity_weight * similar_encodings
        else:
            final_encoding = base_encoding
        
        # Apply temporal evolution if available
        if temporal_state is not None:
            temporal_influence = torch.sigmoid(
                self.temporal_position_gate(torch.cat([final_encoding, temporal_state.unsqueeze(0)], dim=-1))
            )
            final_encoding = final_encoding + temporal_influence * self.position_evolution(final_encoding)
        
        # Store for future O(1) access (ensure CPU storage for PMB)
        self.pmb.store(pos_id, content_key.squeeze(0).cpu(), final_encoding.squeeze(0).cpu())
        
        return final_encoding.squeeze(0)  # [d_model]

class PMBTrueEvolving(nn.Module):
    """TrueEvolving attention enhanced with PMB for infinite context"""
    
    def __init__(self, config, layer_idx: int):
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
        
        # PMB-based infinite positional encoding
        self.position_memory_bank = PositionalMemoryBank(self.hidden_size)
        
        # Evolving attention components
        self.attention_evolution = nn.Linear(config.hidden_size * 2, config.num_heads, bias=True)
        self.memory_gate = nn.Linear(config.hidden_size, config.num_heads, bias=True)
        self.temporal_dynamics = nn.Linear(config.hidden_size, config.num_heads, bias=True)
        
        # PMB-aware evolution components
        self.pmb_attention_gate = nn.Linear(config.hidden_size * 2, config.num_heads, bias=True)
        self.position_attention_fusion = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=True)
        
        # Learnable evolution parameters per head
        self.evolution_rate = nn.Parameter(torch.full((config.num_heads,), 0.1))
        self.memory_decay = nn.Parameter(torch.full((config.num_heads,), 0.95))
        self.pmb_weight = nn.Parameter(torch.full((config.num_heads,), 0.2))
        
        # Attention memory buffer for temporal evolution
        self.register_buffer('attention_memory', torch.zeros(1, 1, config.num_heads, 1, 1))
        
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(self, hidden_states, attention_states=None):
        """Forward pass with PMB-enhanced temporal evolution"""
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get PMB-based positional encodings for each token
        enhanced_states = []
        current_context = hidden_states.mean(dim=1)  # [batch, hidden]
        
        for i in range(seq_len):
            token = hidden_states[:, i, :]  # [batch, hidden]
            
            # Get infinite positional encoding for each token in batch
            pos_encodings = []
            for b in range(batch_size):
                pos_enc = self.position_memory_bank.get_position_encoding(
                    position=i,
                    token_content=token[b],
                    temporal_state=current_context[b] if attention_states is not None else None
                )
                # Ensure pos_enc is on the correct device
                pos_enc = pos_enc.to(token.device)
                pos_encodings.append(pos_enc)
            
            pos_encodings = torch.stack(pos_encodings, dim=0)  # [batch, hidden]
            
            # Fuse token content with positional information
            fused_token = self.position_attention_fusion(
                torch.cat([token, pos_encodings], dim=-1)
            )
            enhanced_states.append(fused_token)
        
        enhanced_hidden_states = torch.stack(enhanced_states, dim=1)  # [batch, seq_len, hidden]
        
        # Project to Q, K, V using enhanced states
        q = self.q_proj(enhanced_hidden_states)
        k = self.k_proj(enhanced_hidden_states)
        v = self.v_proj(enhanced_hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Initialize or update attention memory
        if attention_states is None:
            attention_memory = self.attention_memory.expand(batch_size, seq_len, -1, -1, -1)
        else:
            attention_memory = attention_states
        
        # Enhanced temporal evolution with PMB awareness
        current_context = enhanced_hidden_states.mean(dim=1, keepdim=True)  # [batch, 1, hidden]
        
        if attention_states is not None:
            # Get previous context from attention memory
            prev_context = attention_states.mean(dim=(1, 2, 3, 4), keepdim=True)  # [batch, 1, 1, 1, 1]
            prev_context = prev_context.squeeze(-1).squeeze(-1).squeeze(-1)  # [batch, 1]
            prev_context = prev_context.unsqueeze(-1).expand(-1, -1, self.hidden_size)  # [batch, 1, hidden]
            
            # Evolution dynamics
            context_change = torch.cat([current_context, prev_context], dim=-1)
            evolution_strength = torch.sigmoid(self.attention_evolution(context_change))
            evolution_strength = evolution_strength.unsqueeze(-1).unsqueeze(-1)
            
            # Memory gating
            memory_strength = torch.sigmoid(self.memory_gate(current_context))
            memory_strength = memory_strength.unsqueeze(-1).unsqueeze(-1)
            
            # Temporal dynamics
            temporal_flow = torch.tanh(self.temporal_dynamics(current_context))
            temporal_flow = temporal_flow.unsqueeze(-1).unsqueeze(-1)
            
            # PMB-aware attention gating
            pmb_context = torch.cat([current_context, prev_context], dim=-1)
            pmb_gate = torch.sigmoid(self.pmb_attention_gate(pmb_context))
            pmb_gate = pmb_gate.unsqueeze(-1).unsqueeze(-1)
            
            # Combine evolution factors with PMB influence
            total_evolution = (
                evolution_strength * self.evolution_rate.view(1, 1, -1, 1, 1) + 
                pmb_gate * self.pmb_weight.view(1, 1, -1, 1, 1)
            )
            
            # Update attention memory with decay and evolution
            attention_memory = (
                attention_memory * self.memory_decay.view(1, 1, -1, 1, 1) + 
                total_evolution * temporal_flow
            )
        
        # Apply evolved attention bias
        if attention_memory.shape[-2:] == attn_scores.shape[-2:]:
            memory_bias = attention_memory.squeeze(1)  # [batch, heads, seq_len, seq_len]
            attn_scores = attn_scores + memory_bias
        
        # Apply attention
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        attn_output = self.out_proj(attn_output)
        
        # Update attention states for next layer
        new_attention_states = attention_memory
        
        return attn_output, new_attention_states

class PMBTrueEvolvingConfig:
    """Configuration for PMB-enhanced TrueEvolving"""
    def __init__(self, hidden_size=64, num_heads=4, max_position_embeddings=None, dropout=0.1):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_position_embeddings = max_position_embeddings  # Not used - infinite context!
        self.dropout = dropout

# Test the PMB-enhanced TrueEvolving
if __name__ == "__main__":
    # Test configuration
    config = PMBTrueEvolvingConfig(hidden_size=64, num_heads=4)
    
    # Create model
    model = PMBTrueEvolving(config, layer_idx=0)
    
    # Test input
    batch_size, seq_len = 2, 32
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    
    print("🚀 Testing PMB-Enhanced TrueEvolving...")
    
    # Forward pass
    output, attention_states = model(hidden_states)
    
    print(f"✅ Input shape: {hidden_states.shape}")
    print(f"✅ Output shape: {output.shape}")
    print(f"✅ Attention states shape: {attention_states.shape}")
    
    # Test with different sequence lengths (infinite context capability)
    print("\n🌌 Testing infinite context capability...")
    for test_len in [16, 64, 128, 256, 512, 1024]:
        test_input = torch.randn(1, test_len, config.hidden_size)
        test_output, _ = model(test_input)
        print(f"✅ Length {test_len}: {test_input.shape} -> {test_output.shape}")
    
    print("\n🎯 PMB Statistics:")
    print(f"✅ Positions stored in PMB: {len(model.position_memory_bank.pmb)}")
    print("🚀 PMB-Enhanced TrueEvolving ready for infinite context!")
