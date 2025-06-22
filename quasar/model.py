# c:\quasarv4\quasar\model.py

import torch
import torch.nn as nn
import uuid

from .lnn import LNN
from transformers import PreTrainedModel, PretrainedConfig
from .pmb import ParameterMemoryBank
from .chunker import SemanticChunker
from .moe import MoELayer
from torch.utils.checkpoint import checkpoint

class QuasarConfig(PretrainedConfig):
    model_type = "quasar"

    def __init__(
        self, 
        vocab_size=129280, 
        embedding_dim=8192, 
        hidden_dim=8192, 
        num_experts=128, 
        expert_dim=2048, 
        top_k=4, 
        lnn_config=None, 
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dim = expert_dim
        self.top_k = top_k
        self.lnn_config = lnn_config or {}
        super().__init__(**kwargs)

class Quasar(PreTrainedModel):
    config_class = QuasarConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: QuasarConfig):
        super().__init__(config)
        self.embedding_dim = config.embedding_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        # Token embedding layer
        self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=0)

        # Core LNN for processing sequences
        # Note: input_size for LNN is embedding_dim, and it outputs hidden_dim states.
        self.lnn = LNN(input_size=config.embedding_dim, hidden_size=config.hidden_dim, **config.lnn_config)

        self.use_moe = config.num_experts > 0
        if self.use_moe:
            self.moe_layer = MoELayer(
                embedding_dim=config.hidden_dim, 
                num_experts=config.num_experts, 
                expert_dim=config.expert_dim,
                top_k=config.top_k
            )


        # Output layer for next-token prediction
        self.output_head = nn.Linear(config.hidden_dim, config.vocab_size)


    def forward(self, input_ids):
        """
        Defines the forward pass of the Quasar model for pre-training.
        This version supports gradient checkpointing to save memory.
        """
        # 1. Embedding
        embedded_input = self.embedding(input_ids)

        # 2. LNN processing
        # LNN expects (seq_len, batch_size, dim)
        hidden_states = embedded_input.transpose(0, 1)

        if self.gradient_checkpointing and self.training:
            # use_reentrant=False is more memory-efficient
            hidden_states = checkpoint(self.lnn, hidden_states, use_reentrant=False)
        else:
            hidden_states = self.lnn(hidden_states)
        
        hidden_states = hidden_states.transpose(0, 1)

        # 3. MoE Layer (if enabled)
        load_balancing_loss = torch.tensor(0.0, device=hidden_states.device)
        if self.use_moe:
            if self.gradient_checkpointing and self.training:
                # moe_layer returns two tensors, which is supported by checkpoint
                hidden_states, load_balancing_loss = checkpoint(self.moe_layer, hidden_states, use_reentrant=False)
            else:
                hidden_states, load_balancing_loss = self.moe_layer(hidden_states)

        # 4. Output Head
        logits = self.output_head(hidden_states)

        return logits, load_balancing_loss
