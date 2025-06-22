# c:\quasarv4\quasar\moe.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

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
        """ Returns the top-k weights and indices for each token. """
        gate_logits = self.gate(x.reshape(-1, x.shape[-1]))
        top_k_logits, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_logits, dim=-1, dtype=torch.float).to(x.dtype)
        return top_k_weights, top_k_indices

class MoELayer(nn.Module):
    """A Mixture of Experts layer."""
    def __init__(self, embedding_dim, num_experts, expert_dim, top_k=2):
        super().__init__()
        self.router = MoERouter(embedding_dim, num_experts, top_k)
        self.num_experts = num_experts

        # Create experts
        experts = [Expert(embedding_dim, expert_dim) for _ in range(self.num_experts)]
        self.experts = nn.ModuleList(experts)

    def forward(self, x):
        """Forward pass for the MoE layer."""
        original_shape = x.shape
        flat_x = x.reshape(-1, x.shape[-1])

        # Create the final output tensor on the correct device, avoiding meta-device issues.
        final_output = torch.zeros(flat_x.shape, dtype=x.dtype, device=self.router.gate.weight.device)

        # Get routing decisions from the router
        top_k_weights, top_k_indices = self.router(x)

        # Calculate load balancing loss using one_hot to be meta-tensor compatible
        num_tokens = top_k_indices.size(0)
        one_hot_indices = F.one_hot(top_k_indices, num_classes=self.num_experts).float()
        tokens_per_expert = one_hot_indices.sum(dim=[0, 1])
        router_probs_per_expert = torch.mean(F.softmax(self.router.gate.weight, dim=0), dim=1)
        load_balancing_loss = self.num_experts * torch.dot(tokens_per_expert / num_tokens, router_probs_per_expert)

        # Dispatch tokens to experts and aggregate outputs
        for i in range(self.num_experts):
            # Find which tokens are routed to this expert
            expert_mask = (top_k_indices == i).any(dim=1)
            expert_indices_for_expert = torch.where(expert_mask)[0]

            if expert_indices_for_expert.numel() == 0:
                continue

            # Get the tokens for this expert
            expert_tokens = flat_x[expert_indices_for_expert]

            # Find the specific weight for this expert for each token
            top_k_weights_for_expert = top_k_weights[expert_indices_for_expert]
            is_expert_in_top_k = (top_k_indices[expert_indices_for_expert] == i)
            weights_for_expert = torch.sum(top_k_weights_for_expert * is_expert_in_top_k, dim=1, keepdim=True)

            # Process with expert and apply routing weight
            expert_output = self.experts[i](expert_tokens)
            weighted_output = expert_output * weights_for_expert

            # Add the weighted output to the final output tensor at the correct positions
            final_output.index_add_(0, expert_indices_for_expert, weighted_output)

        return final_output.reshape(original_shape), load_balancing_loss
