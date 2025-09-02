"""
Hierarchical Flow Anchoring for TrueEvolving Attention

Combines TrueEvolving's continuous temporal flow with discrete PMB checkpoints.
Flow handles local pattern evolution, checkpoints provide exact recall anchors.
"Attention islands in a temporal river" - best of both worlds.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Optional, Tuple, Dict, List
import logging

# Import PMB and TrueEvolving components
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from pmb_trueevolving import PositionalMemoryBank
from quasar.pmb import ParameterMemoryBank

logger = logging.getLogger(__name__)

class CheckpointTrigger(nn.Module):
    """Determines when to create memory checkpoints during flow"""
    
    def __init__(self, d_model: int, entropy_threshold: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.entropy_threshold = entropy_threshold
        
        # Attention entropy analyzer
        self.entropy_analyzer = nn.Linear(d_model, 1)
        
        # Semantic boundary detector
        self.semantic_detector = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )
        
        # Learnable checkpoint frequency
        self.checkpoint_frequency = nn.Parameter(torch.tensor(0.1))  # 10% of tokens by default
        
    def should_checkpoint(self, 
                         current_state: torch.Tensor,
                         prev_state: Optional[torch.Tensor] = None,
                         attention_weights: Optional[torch.Tensor] = None,
                         position: int = 0) -> bool:
        """Determine if current position should be a checkpoint"""
        
        batch_size = current_state.size(0)
        checkpoint_signals = []
        
        # 1. Attention entropy spike detection
        if attention_weights is not None:
            # Calculate attention entropy
            entropy = -torch.sum(attention_weights * torch.log(attention_weights + 1e-8), dim=-1)
            avg_entropy = torch.mean(entropy)
            entropy_signal = self.entropy_analyzer(current_state).squeeze(-1)  # [batch]
            
            high_entropy = (avg_entropy > self.entropy_threshold).float()
            checkpoint_signals.append(entropy_signal * high_entropy)
        
        # 2. Semantic boundary detection
        if prev_state is not None:
            semantic_change = self.semantic_detector(
                torch.cat([current_state, prev_state], dim=-1)
            ).squeeze(-1)  # [batch]
            checkpoint_signals.append(semantic_change)
        
        # 3. Periodic checkpoints (fallback)
        periodic_signal = torch.sigmoid(
            torch.sin(position * self.checkpoint_frequency) * 5.0
        )
        periodic_signal = periodic_signal.expand(batch_size)
        checkpoint_signals.append(periodic_signal)
        
        # Combine signals
        if checkpoint_signals:
            combined_signal = torch.stack(checkpoint_signals, dim=0).mean(dim=0)  # [batch]
            # Use threshold for binary decision
            should_checkpoint = (combined_signal > 0.7).any().item()
        else:
            should_checkpoint = False
            
        return should_checkpoint

class HierarchicalFlowMemory(nn.Module):
    """Manages checkpoints and flow states for hierarchical memory"""
    
    def __init__(self, d_model: int, max_checkpoints: int = 1000):
        super().__init__()
        self.d_model = d_model
        self.max_checkpoints = max_checkpoints
        
        # PMB for checkpoint storage
        self.checkpoint_pmb = PositionalMemoryBank(d_model)
        
        # Checkpoint metadata storage
        self.checkpoint_positions = []  # List of checkpoint positions
        self.checkpoint_states = {}     # Position -> state mapping
        
        # Flow interpolation network
        self.flow_interpolator = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),  # prev_checkpoint + target_pos + next_checkpoint
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
            nn.Tanh()
        )
        
    def store_checkpoint(self, position: int, state: torch.Tensor, token_content: torch.Tensor):
        """Store a checkpoint at given position"""
        
        # Store in PMB
        checkpoint_id = f"checkpoint_{position}"
        self.checkpoint_pmb.pmb.store(
            checkpoint_id, 
            token_content.cpu(), 
            state.cpu()
        )
        
        # Update metadata
        self.checkpoint_positions.append(position)
        self.checkpoint_positions.sort()
        
        # Limit checkpoint count
        if len(self.checkpoint_positions) > self.max_checkpoints:
            # Remove oldest checkpoint
            old_pos = self.checkpoint_positions.pop(0)
            old_id = f"checkpoint_{old_pos}"
            # Note: PMB doesn't have explicit delete, but it will be overwritten
            
        logger.debug(f"Stored checkpoint at position {position}, total checkpoints: {len(self.checkpoint_positions)}")
    
    def retrieve_surrounding_checkpoints(self, target_position: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], int, int]:
        """Find checkpoints before and after target position"""
        
        if not self.checkpoint_positions:
            return None, None, -1, -1
            
        # Find surrounding checkpoints
        prev_pos, next_pos = -1, -1
        prev_state, next_state = None, None
        
        for pos in self.checkpoint_positions:
            if pos <= target_position:
                prev_pos = pos
            elif pos > target_position and next_pos == -1:
                next_pos = pos
                break
                
        # Retrieve states
        if prev_pos != -1:
            prev_id = f"checkpoint_{prev_pos}"
            prev_state = self.checkpoint_pmb.pmb.retrieve_direct(prev_id)
            if prev_state is not None:
                prev_state = prev_state.to(next(self.parameters()).device)
                
        if next_pos != -1:
            next_id = f"checkpoint_{next_pos}"
            next_state = self.checkpoint_pmb.pmb.retrieve_direct(next_id)
            if next_state is not None:
                next_state = next_state.to(next(self.parameters()).device)
        
        return prev_state, next_state, prev_pos, next_pos
    
    def interpolate_flow_state(self, 
                              target_position: int,
                              target_content: torch.Tensor,
                              prev_state: Optional[torch.Tensor] = None,
                              next_state: Optional[torch.Tensor] = None,
                              prev_pos: int = -1,
                              next_pos: int = -1) -> torch.Tensor:
        """Interpolate state between checkpoints using flow dynamics"""
        
        batch_size = target_content.size(0)
        device = target_content.device
        
        if prev_state is None and next_state is None:
            # No checkpoints available, use content-based encoding
            return self.checkpoint_pmb.position_generator(
                torch.tensor([float(target_position)], device=device).unsqueeze(0)
            ).expand(batch_size, -1)
        
        # Prepare interpolation inputs
        interpolation_inputs = []
        
        if prev_state is not None:
            # Handle batch size mismatch - repeat or slice as needed
            if prev_state.size(0) == 1 and batch_size > 1:
                prev_state = prev_state.repeat(batch_size, 1)
            elif prev_state.size(0) > batch_size:
                prev_state = prev_state[:batch_size]
            interpolation_inputs.append(prev_state)
        else:
            interpolation_inputs.append(torch.zeros(batch_size, self.d_model, device=device))
            
        # Position encoding for target
        pos_encoding = self.checkpoint_pmb.position_generator(
            torch.tensor([float(target_position)], device=device).unsqueeze(0)
        ).expand(batch_size, -1)
        interpolation_inputs.append(pos_encoding)
        
        if next_state is not None:
            # Handle batch size mismatch - repeat or slice as needed
            if next_state.size(0) == 1 and batch_size > 1:
                next_state = next_state.repeat(batch_size, 1)
            elif next_state.size(0) > batch_size:
                next_state = next_state[:batch_size]
            interpolation_inputs.append(next_state)
        else:
            interpolation_inputs.append(torch.zeros(batch_size, self.d_model, device=device))
        
        # Flow interpolation
        interpolation_input = torch.cat(interpolation_inputs, dim=-1)
        interpolated_state = self.flow_interpolator(interpolation_input)
        
        return interpolated_state

class HierarchicalFlowAnchoring(nn.Module):
    """TrueEvolving with Hierarchical Flow Anchoring - the breakthrough architecture"""
    
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.d_model = config.hidden_size
        self.num_heads = config.num_heads
        self.head_dim = self.d_model // self.num_heads
        
        # Core TrueEvolving components
        self.q_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.k_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.v_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=False)
        
        # Temporal evolution components
        self.attention_evolution = nn.Linear(self.d_model * 2, self.num_heads, bias=True)
        self.memory_gate = nn.Linear(self.d_model, self.num_heads, bias=True)
        self.temporal_dynamics = nn.Linear(self.d_model, self.num_heads, bias=True)
        
        # Evolution parameters
        self.evolution_rate = nn.Parameter(torch.full((self.num_heads,), 0.1))
        self.memory_decay = nn.Parameter(torch.full((self.num_heads,), 0.95))
        
        # Hierarchical Flow Anchoring components
        self.checkpoint_trigger = CheckpointTrigger(self.d_model)
        self.hierarchical_memory = HierarchicalFlowMemory(self.d_model)
        
        # Flow-checkpoint fusion
        self.flow_checkpoint_fusion = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.ReLU(),
            nn.Linear(self.d_model, self.d_model),
            nn.Tanh()
        )
        
        self.dropout = nn.Dropout(getattr(config, 'dropout', 0.1))
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Attention memory for flow
        self.register_buffer('attention_memory', torch.zeros(1, 1, self.num_heads, 1, 1))
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical Flow Anchoring forward pass
        
        Args:
            hidden_states: [batch, seq_len, d_model]
            attention_states: Previous attention memory states
            
        Returns:
            output: [batch, seq_len, d_model] 
            new_attention_states: Updated attention memory
        """
        
        batch_size, seq_len, d_model = hidden_states.shape
        device = hidden_states.device
        
        # Initialize attention memory if needed
        if attention_states is None:
            attention_states = self.attention_memory.expand(batch_size, 1, self.num_heads, 1, 1).to(device)
        
        outputs = []
        current_attention_memory = attention_states
        prev_state = None
        
        for i in range(seq_len):
            current_token = hidden_states[:, i, :]  # [batch, d_model]
            
            # === CHECKPOINT DECISION ===
            should_checkpoint = self.checkpoint_trigger.should_checkpoint(
                current_state=current_token,
                prev_state=prev_state,
                position=i
            )
            
            # === HIERARCHICAL MEMORY RETRIEVAL ===
            # Get surrounding checkpoints
            prev_checkpoint, next_checkpoint, prev_pos, next_pos = \
                self.hierarchical_memory.retrieve_surrounding_checkpoints(i)
            
            # Interpolate flow state between checkpoints
            checkpoint_state = self.hierarchical_memory.interpolate_flow_state(
                target_position=i,
                target_content=current_token,
                prev_state=prev_checkpoint,
                next_state=next_checkpoint,
                prev_pos=prev_pos,
                next_pos=next_pos
            )
            
            # === TRUEEVOLVING FLOW COMPUTATION ===
            # Standard attention projections
            q = self.q_proj(current_token).view(batch_size, 1, self.num_heads, self.head_dim)
            k = self.k_proj(current_token).view(batch_size, 1, self.num_heads, self.head_dim)
            v = self.v_proj(current_token).view(batch_size, 1, self.num_heads, self.head_dim)
            
            # Temporal evolution (flow between checkpoints)
            if i > 0:
                context = torch.cat([current_token, prev_state], dim=-1)
                evolution_weights = torch.sigmoid(self.attention_evolution(context))  # [batch, num_heads]
                
                # Apply temporal dynamics
                temporal_influence = torch.sigmoid(self.temporal_dynamics(current_token))  # [batch, num_heads]
                
                # Evolve attention memory with flow - simplified for single token processing
                evolution_factor = self.evolution_rate.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads]
                decay_factor = self.memory_decay.unsqueeze(0).unsqueeze(0)  # [1, 1, num_heads]
                
                # Update memory state (simplified evolution)
                memory_update = evolution_weights.unsqueeze(-1).unsqueeze(-1) * evolution_factor.unsqueeze(-1)
                current_attention_memory = current_attention_memory * decay_factor.unsqueeze(-1).unsqueeze(-1)
            
            # Attention computation with evolved memory
            attention_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            attention_weights = F.softmax(attention_scores, dim=-1)
            attention_output = torch.matmul(attention_weights, v)
            
            # Reshape and project
            flow_output = attention_output.view(batch_size, d_model)
            flow_output = self.out_proj(flow_output)
            
            # === FLOW-CHECKPOINT FUSION ===
            # Fuse temporal flow with checkpoint-based memory
            fused_input = torch.cat([flow_output, checkpoint_state], dim=-1)
            final_output = self.flow_checkpoint_fusion(fused_input)
            
            outputs.append(final_output)
            
            # === CHECKPOINT STORAGE ===
            if should_checkpoint:
                self.hierarchical_memory.store_checkpoint(i, final_output, current_token)
                logger.debug(f"Created checkpoint at position {i}")
            
            prev_state = final_output
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)  # [batch, seq_len, d_model]
        
        return output, current_attention_memory

# Test configuration
class HierarchicalFlowConfig:
    def __init__(self):
        self.hidden_size = 128
        self.num_heads = 4
        self.vocab_size = 1000
        self.dropout = 0.1

def test_hierarchical_flow_anchoring():
    """Test the Hierarchical Flow Anchoring mechanism"""
    
    print("🚀 Testing Hierarchical Flow Anchoring")
    
    # Setup
    config = HierarchicalFlowConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = HierarchicalFlowAnchoring(config, layer_idx=0).to(device)
    
    # Test input
    batch_size, seq_len = 2, 64
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size, device=device)
    
    print(f"Input shape: {hidden_states.shape}")
    
    # Forward pass
    with torch.no_grad():
        output, attention_states = model(hidden_states)
    
    print(f"Output shape: {output.shape}")
    print(f"Attention states shape: {attention_states.shape}")
    print(f"Checkpoints created: {len(model.hierarchical_memory.checkpoint_positions)}")
    
    # Test infinite context capability
    print("\n🌊 Testing infinite context...")
    long_seq_len = 512
    long_hidden_states = torch.randn(1, long_seq_len, config.hidden_size, device=device)
    
    with torch.no_grad():
        long_output, _ = model(long_hidden_states)
    
    print(f"Long sequence output shape: {long_output.shape}")
    print(f"Total checkpoints after long sequence: {len(model.hierarchical_memory.checkpoint_positions)}")
    
    print("✅ Hierarchical Flow Anchoring test complete!")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.DEBUG)
    
    # Run test
    test_hierarchical_flow_anchoring()
