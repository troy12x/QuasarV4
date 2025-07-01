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
import math
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.generic import ModelOutput
from typing import Optional, Tuple, List
from dataclasses import dataclass
from .pmb import ParameterMemoryBank
from .moe import MoELayer, Expert


from tqdm import tqdm

try:
    from torchdiffeq import odeint
except ImportError:
    raise ImportError("torchdiffeq is not installed. Please install it with `pip install torchdiffeq`")

# --- 1. Configuration Class ---
class LNNConfig(PretrainedConfig):
    """
    Configuration class for the Liquid Neural Network (LNN) model.
    Inherits from HuggingFace's PretrainedConfig.
    """
    model_type = "quasar"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=8192,
        num_hidden_layers=96,  # 96 layers to keep active parameters manageable
        activation='gelu',
        lambda_res=0.0,
        dt=0.2, # Step size for the fixed-step Euler solver.
        initializer_range=0.02,
        dropout=0.1,
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
        self.dt = dt
        self.activation = activation
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
    loss: Optional[torch.FloatTensor] = None
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
        
        # Core LNN parameters
        self.W = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.U = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.b = nn.Parameter(torch.empty(config.hidden_size))

        # Input-Dependent Dynamics
        self.tau_w_h = nn.Linear(config.hidden_size, config.hidden_size)
        self.tau_w_u = nn.Linear(config.hidden_size, config.hidden_size)
        self.tau_b = nn.Parameter(torch.empty(config.hidden_size))

        # Initialize weights
        nn.init.orthogonal_(self.W) # Orthogonal init for recurrent weights
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.b)
        self.tau_b.data.uniform_(-2, 2)

        self.sigma = nn.Tanh() # Use Tanh for bounded output and stability

    def forward(self, h, u):
        """Core ODE dynamics calculation for a single discrete step."""
        # 1. Compute Input-Dependent Time Constant (tau)
        tau_control = self.tau_w_h(h) + self.tau_w_u(u) + self.tau_b
        tau_positive = F.softplus(tau_control) + 0.01 # Add a floor for stability

        # 2. Compute State Update
        decay_term = -h / tau_positive
        activation_input = F.linear(h, self.W) + F.linear(u, self.U) + self.b
        activation_output = self.sigma(activation_input)
        dx_dt = decay_term + activation_output

        if self.lambda_res > 0:
            dx_dt = dx_dt + self.lambda_res * u

        # 3. Stability: Clip the derivative
        dx_dt = torch.clamp(dx_dt, -10, 10)
        return dx_dt

# --- 4. LNN Block (Layer + Residual) ---
class LNNBlock(nn.Module):
    """ A single block of the LNN, using a fixed-step Euler loop. """
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.dt = config.dt
        self.cell = LNNCell(config)
        self.ln = nn.LayerNorm(config.hidden_size)

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes the entire sequence using a fixed-step Euler integration loop,
        starting from a given hidden state h.
        This version is optimized to be JIT-friendly by pre-allocating the output tensor.
        """
        seq_len = x.size(1)
        # Pre-allocate tensor for outputs to avoid slow list appends
        outputs = torch.empty(x.size(0), seq_len, self.hidden_size, device=x.device)

        for t in range(seq_len):
            u = x[:, t, :]
            dx_dt = self.cell(h, u)
            h = h + self.dt * dx_dt
            outputs[:, t, :] = h

        # Add residual connection and layer norm
        output = self.ln(outputs + x)
        return output, h

# --- 5. Full LNN Model ---
class LNNModel(PreTrainedModel):
    """
    The Liquid Neural Network Model.
    This version restores the architecture from the high-performing `old_lnn.py`.
    It uses stacked LNNBlocks to process the sequence and a Transformer-based
    attention readout for global context before prediction.
    """
    config_class = LNNConfig

    def __init__(self, config: LNNConfig):
        super().__init__(config)
        self.config = config

        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.blocks = nn.ModuleList([LNNBlock(config) for _ in range(config.num_hidden_layers)])
        
        # JIT-compile the LNNBlocks for a significant performance boost
        for i in range(len(self.blocks)):
            self.blocks[i] = torch.jit.script(self.blocks[i])

        self.ln_final = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # Restore the attention-based readout
        self.readout = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=8, # A reasonable default
            dim_feedforward=config.hidden_size * 4, # Standard practice
            dropout=0.1,
            activation=F.gelu,
            batch_first=True,
            norm_first=True
        )
        
        self.proj_out = nn.Linear(config.hidden_size, config.vocab_size)

    def get_input_embeddings(self):
        return self.embedding

    def set_input_embeddings(self, value):
        self.embedding = value

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        hidden_states: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,  # Accept attention_mask
        **kwargs,  # Accept other arguments
    ) -> LNNModelOutput:
        """
        Processes a sequence, calculates loss, and handles unexpected arguments.
        The `attention_mask` is accepted but not used, as the LNN processes
        the sequence recurrently.
        """
        # 1. Get Embeddings
        x = self.embedding(input_ids)
        batch_size = input_ids.shape[0]

        # 2. Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.config.hidden_size, device=x.device)
                for _ in range(self.config.num_hidden_layers)
            ]

        # 3. Process sequence through LNN blocks
        new_hidden_states = []
        layer_output = x
        for i, block in enumerate(self.blocks):
            h_initial = hidden_states[i]
            layer_output, h_final = block(layer_output, h_initial)
            new_hidden_states.append(h_final)

        # 4. Final Readout and Projection
        readout_output = self.readout(layer_output)
        readout_output = self.ln_final(readout_output)
        logits = self.proj_out(readout_output)

        # 5. Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Flatten the tokens and compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        return LNNModelOutput(
            loss=loss,
            logits=logits,
            hidden_states=tuple(new_hidden_states),
        )