# c:\quasarv4\quasar\moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Expert(nn.Module):
    """An expert network. For Quasar, this could be an LNN layer followed by a feed-forward network."""
    def __init__(self, embedding_dim, expert_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, expert_dim),
            nn.GELU(),
            nn.Linear(expert_dim, embedding_dim)
        )

    def forward(self, x):
        return self.net(x)

class MoERouter(nn.Module):
    """A simple router that learns to dispatch tokens to experts."""
    def __init__(self, embedding_dim, num_experts, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.gate = nn.Linear(embedding_dim, num_experts)

    def forward(self, x):
        # x: [batch_size, seq_len, embedding_dim]
        # gate_logits: [batch_size * seq_len, num_experts]
        gate_logits = self.gate(x.view(-1, x.shape[-1]))
        
        # Find top-k experts for each token
        # top_k_weights: [batch_size * seq_len, top_k]
        # top_k_indices: [batch_size * seq_len, top_k]
        top_k_weights, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        
        # Apply softmax to get normalized weights
        top_k_weights = F.softmax(top_k_weights, dim=-1, dtype=torch.float).to(x.dtype)
        
        # Create a sparse routing matrix for combining expert outputs
        # routing_weights: [batch_size * seq_len, num_experts]
        routing_weights = torch.zeros_like(gate_logits, dtype=x.dtype)
        routing_weights.scatter_(1, top_k_indices, top_k_weights)
        
        return routing_weights, top_k_indices

class MoELayer(nn.Module):
    """A Mixture of Experts layer."""
    def __init__(self, embedding_dim, num_experts, expert_dim, top_k=2):
        super().__init__()
        self.router = MoERouter(embedding_dim, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(embedding_dim, expert_dim) for _ in range(num_experts)])
        self.num_experts = num_experts

    def forward(self, x):
        # Get routing decisions
        routing_weights, top_k_indices = self.router(x)
        
        # Flatten input for expert processing
        flat_x = x.view(-1, x.shape[-1])
        
        # Initialize final output tensor
        final_output = torch.zeros_like(flat_x)
        
        # Loop through each expert
        for i, expert in enumerate(self.experts):
            # Find which tokens are routed to this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            expert_indices = torch.where(expert_mask)[0]
            
            if expert_indices.numel() > 0:
                # Select tokens and corresponding weights for this expert
                expert_inputs = flat_x[expert_indices]
                expert_routing_weights = routing_weights[expert_indices, i]
                
                # Get expert output and apply routing weight
                expert_output = expert(expert_inputs)
                weighted_output = expert_output * expert_routing_weights.unsqueeze(1)
                
                # Add to final output (scatter_add_ is inplace)
                final_output.index_add_(0, expert_indices, weighted_output)

        # Reshape to original input shape
        return final_output.view(x.shape)

    def get_load_balancing_loss(self, routing_weights):
        """Calculate the load balancing loss."""
        # routing_weights: [num_tokens, num_experts]
        num_tokens = routing_weights.shape[0]
        
        # Per-expert load: fraction of tokens routed to each expert
        load = routing_weights.sum(dim=0) / num_tokens
        
        # Per-expert importance: mean routing probability to each expert
        importance = routing_weights.mean(dim=0)
        
        # Loss is based on the square of the coefficient of variation
        loss = self.num_experts * torch.sum(load * importance)
        return loss
