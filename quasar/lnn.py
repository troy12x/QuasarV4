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
from .moe import MoELayer, Expert

from tqdm import tqdm

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
        num_hidden_layers=96,  # 96 layers to keep active parameters manageable
        activation='gelu',
        lambda_res=0.0,
        dt=1.0,
        initializer_range=0.02,
        use_pmb=False,
        pmb_num_blocks=1024,
        pmb_slots_per_block=4096,
        pmb_top_k=1,
        # MoE parameters
        use_moe: bool = False,
        num_experts: int = 407,   # 407 experts to reach 440B total parameters
        num_experts_per_tok: int = 4,  # 4 active experts per token to maintain 25B active params
        expert_dim: int = 32768,  # 32K expert dimension for capacity
        moe_load_balance_loss_weight: float = 0.01,
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
        # MoE
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_dim = expert_dim
        self.moe_load_balance_loss_weight = moe_load_balance_loss_weight
        super().__init__(**kwargs)

# --- 2. Custom Model Output ---
@dataclass
class LNNModelOutput(ModelOutput):
    """
    Base class for LNN model's outputs, ensuring compatibility with HuggingFace.
    """
    logits: torch.FloatTensor = None
    last_hidden_state: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    load_balancing_loss: Optional[torch.FloatTensor] = None


# --- 3. Core LNN Cell ---
class LNNCell(nn.Module):
    """A single Liquid Neural Network cell with continuous-time dynamics."""
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.lambda_res = config.lambda_res
        self.use_moe = config.use_moe
        self.expert_dim = config.expert_dim
        
        # Core LNN parameters
        self.W = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.U = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.b = nn.Parameter(torch.empty(config.hidden_size))
        self.tau = nn.Parameter(torch.empty(config.hidden_size))
        
        # Initialize weights
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.b)
        nn.init.ones_(self.tau)

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
    """ A single block of the LNN, optionally with a Mixture of Experts layer. """
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.config = config
        
        # Initialize cell and layer norm first
        self.cell = LNNCell(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size)
        
        self.use_moe = config.use_moe
        if self.use_moe:
            self.moe_layer = MoELayer(
                embedding_dim=config.hidden_size,
                num_experts=config.num_experts,
                expert_dim=config.expert_dim,
                top_k=config.num_experts_per_tok
            )
            self.ln_2 = nn.LayerNorm(config.hidden_size)

    def forward(self, x, hidden_state, pmb=None):
        pmb_out = torch.zeros_like(x)  # Initialize with zeros to match x's shape
        
        if pmb is not None and self.config.use_pmb:
            try:
                # Query PMB with current input
                retrieved_memories = pmb.retrieve_semantic(x, top_k=self.config.pmb_top_k)
                # Ensure retrieved memories have the same shape as x
                if retrieved_memories.shape == x.shape:
                    pmb_out = retrieved_memories
                else:
                    # If shapes don't match, project or reshape as needed
                    print(f"Warning: PMB output shape {retrieved_memories.shape} doesn't match input shape {x.shape}")
                    pmb_out = torch.zeros_like(x)
            except Exception as e:
                print(f"Warning: PMB retrieval failed: {e}")
                pmb_out = torch.zeros_like(x)

        # Recurrent part
        residual = x
        dx_dt = self.cell(hidden_state, x)
        x = hidden_state + self.config.dt * dx_dt # Euler integration
        x = self.ln_1(x + residual + pmb_out)

        # MoE part (replaces the FFN in a standard Transformer block)
        if self.use_moe:
            residual_moe = x
            x, load_balancing_loss = self.moe_layer(x)
            x = self.ln_2(x + residual_moe)
            return x, load_balancing_loss
        
        return x, None # Return a tuple to maintain a consistent signature

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

        self.blocks = nn.ModuleList()
        for i in tqdm(range(config.num_hidden_layers), desc="Creating LNN blocks"):
            self.blocks.append(LNNBlock(config))
        
        # --- Final Initialization Steps ---
        print("  [LNNModel] Creating final model components...")
        if config.use_pmb:
            self.pmb = ParameterMemoryBank(
                num_blocks=config.pmb_num_blocks,
                slots_per_block=config.pmb_slots_per_block,
                embedding_dim=config.hidden_size
            )
        else:
            self.pmb = None
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.gradient_checkpointing = False
        print("  [LNNModel] Final components created.")

        # Since post_init is bypassed, we manually initialize the embedding weights.
        print("  [LNNModel] Initializing embedding weights...")
        self.embedding.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

        # Manually tie the weights directly, bypassing post_init and _init_weights completely.
        print("  [LNNModel] Manually tying output and embedding weights...")
        self.proj_out.weight = self.embedding.weight
        print("  [LNNModel] Weights tied successfully.")

        print("  [LNNModel] Model initialization complete.")
 


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
        all_load_balancing_losses = []
        outputs_over_time = []

        # --- PMB Store Operation (Improved) ---
        if self.pmb is not None and self.training:
            with torch.no_grad():
                # Store embeddings more efficiently
                for t in range(seq_len):
                    for b in range(batch_size):
                        item_id = f"step_{self.training_step if hasattr(self, 'training_step') else 0}_batch_{b}_token_{t}"
                        embedding_vector = x[b, t, :].detach()
                        self.pmb.store(item_id, embedding_vector, embedding_vector)

        # 3. Process sequence token by token (recurrently)
        for t in range(seq_len):
            current_input_token = x[:, t, :]
            
            layer_input = current_input_token
            next_hidden_states = []

            for i, block in enumerate(self.blocks):
                h_current = hidden_states[i]
                
                if self.gradient_checkpointing and self.training:
                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs)
                        return custom_forward

                    layer_output, load_balancing_loss = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        layer_input,
                        h_current,
                        self.pmb if self.config.use_pmb else None,
                        use_reentrant=False
                    )
                else:
                    layer_output, load_balancing_loss = block(layer_input, h_current, pmb=self.pmb if self.config.use_pmb else None)
                
                if load_balancing_loss is not None:
                    all_load_balancing_losses.append(load_balancing_loss)

                next_hidden_states.append(layer_output)
                layer_input = layer_output # Input to next layer is output of current

            hidden_states = next_hidden_states
            
            # 4. Get final output from the last layer for this time step
            final_output = hidden_states[-1]
            outputs_over_time.append(final_output)

            if output_hidden_states:
                all_layer_outputs.append(tuple(hidden_states))

        # 5. Stack outputs over time
        last_hidden_state = torch.stack(outputs_over_time, dim=1)
        
        # 6. Final LayerNorm and output projection
        last_hidden_state = self.final_ln(last_hidden_state)
        logits = self.proj_out(last_hidden_state)

        # Calculate total load balancing loss
        total_load_balancing_loss = torch.sum(torch.stack(all_load_balancing_losses)) if all_load_balancing_losses else None

        if not return_dict:
            return (logits, last_hidden_state, tuple(all_layer_outputs) if all_layer_outputs else None, total_load_balancing_loss)

        return LNNModelOutput(
            last_hidden_state=last_hidden_state,
            logits=logits,
            hidden_states=tuple(all_layer_outputs) if all_layer_outputs else None,
            load_balancing_loss=total_load_balancing_loss
        )