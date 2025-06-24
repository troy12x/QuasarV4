# Copyright 2024 Quasar AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.utils.generic import ModelOutput
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .pmb import ParameterMemoryBank

# --- 1. Configuration Class ---
class LNNConfig(PretrainedConfig):
    """
    Configuration class for the Liquid Neural Network (LNN) model.
    Inherits from HuggingFace's PretrainedConfig.
    """
    model_type = "lnn"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=8192,
        num_hidden_layers=96,
        activation='gelu',
        lambda_res=0.0,
        dt=1.0,
        initializer_range=0.02,
        use_pmb=False,
        pmb_num_blocks=1024,
        pmb_slots_per_block=4096,
        pmb_top_k=1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.lambda_res = lambda_res
        self.activation = activation
        self.dt = dt
        self.initializer_range = initializer_range
        self.use_pmb = use_pmb
        self.pmb_num_blocks = pmb_num_blocks
        self.pmb_slots_per_block = pmb_slots_per_block
        self.pmb_top_k = pmb_top_k
        super().__init__(**kwargs)

# --- 2. Custom Model Output ---
@dataclass
class LNNModelOutput(ModelOutput):
    """
    Base class for LNN model's outputs, ensuring compatibility with HuggingFace.
    """
    last_hidden_state: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[List[torch.FloatTensor]] = None # List of hidden states per layer

# --- 3. Core LNN Cell ---
class LNNCell(nn.Module):
    """A single Liquid Neural Network cell with continuous-time dynamics."""
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.lambda_res = config.lambda_res

        # Learnable parameters
        self.W = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.U = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.b = nn.Parameter(torch.empty(config.hidden_size))
        self.tau = nn.Parameter(torch.empty(config.hidden_size)) # Learnable time-constant

        if config.activation == 'gelu':
            self.sigma = nn.GELU()
        else:
            self.sigma = torch.tanh

    def forward(self, h, u):
        """
        The ODE function dx/dt = f(t, x, u).
        h: hidden state (batch_size, hidden_size)
        u: input (batch_size, hidden_size)
        """
        # Ensure tau is positive
        tau_positive = F.softplus(self.tau)
        
        # dX/dt = -1/τ * X + σ(W·X + U·u + b)
        decay_term = -h / tau_positive
        activation_input = F.linear(h, self.W) + F.linear(u, self.U) + self.b
        activation_output = self.sigma(activation_input)
        
        dx_dt = decay_term + activation_output
        
        # Optional residual connection on the input
        if self.lambda_res > 0:
            dx_dt = dx_dt + self.lambda_res * u
            
        return dx_dt

# --- 4. LNN Block (Layer + Residual) ---
class LNNBlock(nn.Module):
    """A recurrent block using an LNN layer with a residual connection."""
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.cell = LNNCell(config)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dt = config.dt

    def forward(self, h, u, pmb=None, pmb_top_k=0):
        """
        h: hidden state (batch_size, hidden_size)
        u: input (batch_size, hidden_size)
        pmb: ParameterMemoryBank instance
        pmb_top_k: Number of memories to retrieve
        """
        # Get the change in hidden state from the cell
        dx_dt = self.cell(h, u)
        
        # Euler integration step
        h_next = h + self.dt * dx_dt
        
        # --- PMB Integration ---
        retrieved_memory = 0
        if pmb is not None and len(pmb) > 0 and pmb_top_k > 0:
            # Query PMB with the new hidden state
            # Note: This assumes the PMB's value dimension matches the hidden_size
            retrieved_memory_batch = pmb.retrieve_semantic(h_next, top_k=pmb_top_k)
            # Average the retrieved memories (B, K, D) -> (B, D)
            retrieved_memory = retrieved_memory_batch.mean(dim=1)

        # Residual connection and LayerNorm, now with memory
        output = self.norm(h_next + u + retrieved_memory)
        
        return output

# --- 5. Full LNN Model ---
class LNNModel(PreTrainedModel):
    """
    A full Liquid Neural Network model composed of stacked LNNBlocks.
    Compatible with HuggingFace's PreTrainedModel.
    """
    config_class = LNNConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: LNNConfig):
        super().__init__(config)
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LNNBlock(config) for _ in range(config.num_hidden_layers)])
        
        # Initialize Parameter Memory Bank if enabled
        self.pmb = ParameterMemoryBank(
            num_blocks=config.pmb_num_blocks,
            slots_per_block=config.pmb_slots_per_block
        ) if config.use_pmb else None

        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # This attribute is needed for HF's gradient checkpointing
        self.gradient_checkpointing = False
        
        self.post_init() # HF-specific method to finalize model initialization

    def _init_weights(self, module):
        """Initializes weights of the model."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, LNNCell):
            # Initialize LNNCell-specific parameters
            nn.init.kaiming_uniform_(module.W, a=0.01)
            nn.init.kaiming_uniform_(module.U, a=0.01)
            nn.init.zeros_(module.b)
            # Initialize tau to be around 1.0 after softplus
            nn.init.uniform_(module.tau, 0.5, 1.5)

    def forward(
        self,
        input_ids: torch.LongTensor,
        hidden_states: Optional[List[torch.FloatTensor]] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> LNNModelOutput:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 1. Get token embeddings
        x = self.embedding(input_ids)
        
        # 2. Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.config.hidden_size, device=device)
                for _ in range(self.config.num_hidden_layers)
            ]

        all_layer_outputs = [] if output_hidden_states else None
        outputs_over_time = []

        # --- PMB Store Operation (Conceptual) ---
        # A real implementation would need a more robust strategy for when and what to store.
        # For this demonstration, we'll populate the PMB with the input embeddings so it has memory to retrieve.
        if self.pmb is not None and self.training:
            with torch.no_grad():
                # Store each token embedding from the batch. This is inefficient but populates the PMB.
                for i in range(input_ids.size(1)):
                    for j in range(input_ids.size(0)):
                        # Using a simple, non-unique ID for demonstration.
                        # A robust solution needs a proper content-addressable or unique ID scheme.
                        item_id = f"batch_{j}_token_{i}"
                        embedding_vector = x[j, i, :].detach()
                        # In this simple case, the key and value are the same.
                        self.pmb.store(item_id, embedding_vector, embedding_vector)

        # 3. Process sequence token by token (recurrently)
        for t in range(seq_len):
            current_input_token = x[:, t, :]
            
            layer_input = current_input_token
            next_hidden_states = []

            for i, layer in enumerate(self.layers):
                h_current = hidden_states[i]
                
                if self.gradient_checkpointing and self.training:
                    # use_reentrant=False is more modern and memory-efficient
                    layer_output = torch.utils.checkpoint.checkpoint(
                        layer, h_current, layer_input, self.pmb, self.config.pmb_top_k, use_reentrant=False
                    )
                else:
                    layer_output = layer(h_current, layer_input, self.pmb, self.config.pmb_top_k)

                next_hidden_states.append(layer_output)
                layer_input = layer_output # Input to next layer is output of current

            hidden_states = next_hidden_states
            
            # 4. Get final output from the last layer for this time step
            final_output = hidden_states[-1]
            outputs_over_time.append(final_output)

            if output_hidden_states:
                all_layer_outputs.append(hidden_states)

        # 5. Stack outputs over time
        last_hidden_state = torch.stack(outputs_over_time, dim=1)
        
        # 6. Final LayerNorm and output projection
        last_hidden_state = self.final_ln(last_hidden_state)
        logits = self.output_head(last_hidden_state)

        if not return_dict:
            return (logits, last_hidden_state, all_layer_outputs)

        return LNNModelOutput(
            last_hidden_state=last_hidden_state,
            logits=logits,
            hidden_states=all_layer_outputs,
        )
