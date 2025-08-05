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
from transformers.generation.utils import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils.generic import ModelOutput
from typing import Optional, Tuple, List
from torch.utils.checkpoint import checkpoint
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
        self.dropout = dropout
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
        # Increased the floor from 0.01 to 1.0 to prevent division by a near-zero
        # number, which is a common cause of NaN in bf16.
        tau_positive = F.softplus(tau_control) + 1.0

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
            # Clamp the hidden state to prevent runaway values, a common
            # source of instability in recurrent models.
            h = torch.clamp(h, -100, 100)
            outputs[:, t, :] = h

        # Add residual connection and layer norm
        output = self.ln(outputs + x)
        return output, h

# --- 5. Full LNN Model ---
class LNNModel(PreTrainedModel, GenerationMixin):
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
        # Disabling JIT as a test, as it can sometimes cause unexpected memory allocation issues with recurrent loops.
        # for i in range(len(self.blocks)):
        #     self.blocks[i] = torch.jit.script(self.blocks[i])

        self.ln_final = nn.LayerNorm(config.hidden_size, eps=1e-5)
        
        # The attention-based readout is removed to prevent the model from "cheating"
        # by using self-attention on the whole sequence instead of relying on its
        # recurrent state. This forces the LNN to learn more robust representations.
        # self.readout = nn.TransformerEncoderLayer(...)
        
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
                torch.zeros(batch_size, self.config.hidden_size, device=x.device, dtype=x.dtype)
                for _ in range(self.config.num_hidden_layers)
            ]

        # 3. Process sequence through LNN blocks
        new_hidden_states = []
        layer_output = x

        # Check if gradient checkpointing is enabled and if we are in training mode
        use_checkpointing = self.training and getattr(self.config, 'gradient_checkpointing', False)

        for i, block in enumerate(self.blocks):
            h_initial = hidden_states[i]

            if use_checkpointing:
                # Pass a function that calls the block's forward method
                layer_output, h_final = checkpoint(
                    block, 
                    layer_output, 
                    h_initial, 
                    use_reentrant=False
                )
            else:
                layer_output, h_final = block(layer_output, h_initial)
            
            new_hidden_states.append(h_final)

        # 4. Final Projection (without attention readout)
        final_output = self.ln_final(layer_output)
        logits = self.proj_out(final_output)

        # 5. Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift so that logits at time t predict token at time t+1
            # This is the standard procedure for training causal language models.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Flatten the tokens and compute loss
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        return LNNModelOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=final_output,
            hidden_states=tuple(new_hidden_states),
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        max_new_tokens: int = None,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
        repetition_penalty: float = 1.0,
        **kwargs
    ) -> torch.LongTensor:
        """
        Generate text using the LNN model with improved repetition handling.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Determine actual max length
        if max_new_tokens is not None:
            max_length = input_ids.shape[1] + max_new_tokens
        
        # Initialize hidden states
        hidden_states = [
            torch.zeros(batch_size, self.config.hidden_size, device=device)
            for _ in range(self.config.num_hidden_layers)
        ]
        
        # Initialize output with input_ids
        generated = input_ids.clone()
        
        # Set model to evaluation mode
        self.eval()
        
        for step in range(max_length - input_ids.shape[1]):
            # Get model output - only pass the last few tokens to avoid recomputing everything
            context_length = min(generated.shape[1], 512)  # Limit context to prevent memory issues
            context_ids = generated[:, -context_length:]
            
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=context_ids,
                    hidden_states=hidden_states if step == 0 else None  # Only use initial hidden states
                )
                
                # Get logits for the last token
                logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for i in range(batch_size):
                        for token_id in set(generated[i].tolist()):
                            # If logit is positive, divide by penalty, else multiply
                            if logits[i, token_id] > 0:
                                logits[i, token_id] /= repetition_penalty
                            else:
                                logits[i, token_id] *= repetition_penalty
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_values, _ = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1)
                    indices_to_remove = logits < top_k_values[..., -1, None]
                    logits[indices_to_remove] = -float('inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    # Convert back to original indices
                    indices_to_remove = sorted_indices_to_remove.gather(dim=-1, index=sorted_indices.argsort(dim=-1))
                    logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated

    def generate_simple(
        self,
        input_ids: torch.LongTensor,
        max_length: int = 100,
        temperature: float = 1.0,
        do_sample: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
        hidden_states: Optional[List[torch.Tensor]] = None,
        **kwargs
    ) -> torch.LongTensor:
        """
        Simple generate method without top-k/top-p sampling to avoid dimension issues.
        """
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize hidden states if not provided
        if hidden_states is None:
            hidden_states = [
                torch.zeros(batch_size, self.config.hidden_size, device=device)
                for _ in range(self.config.num_hidden_layers)
            ]
        
        # Initialize output with input_ids
        generated = input_ids.clone()
        
        # Set model to evaluation mode
        self.eval()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Get model output
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated,
                    hidden_states=hidden_states
                )
                
                # Get logits for the last token
                logits = outputs.logits[:, -1, :]  # Shape: [batch_size, vocab_size]
                hidden_states = list(outputs.hidden_states)
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = True,
        **kwargs
    ) -> dict:
        """
        Prepare inputs for generation. For LNN, we use hidden_states instead of past_key_values.
        """
        # For LNN, we don't use past_key_values in the traditional sense
        # Instead, we rely on the recurrent nature of the model
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        return model_inputs

    def _reorder_cache(self, past_key_values: List[torch.Tensor], beam_idx: torch.Tensor) -> List[torch.Tensor]:
        """
        Reorder hidden states for beam search.
        """
        if past_key_values is None:
            return None
        
        reordered_past = []
        for hidden_state in past_key_values:
            reordered_past.append(hidden_state.index_select(0, beam_idx))
        return reordered_past

# --- 6. For Causal LM compatibility ---
class LNNForCausalLM(LNNModel):
    """
    Wrapper class for compatibility with HuggingFace's CausalLM interface.
    """
    def __init__(self, config: LNNConfig):
        super().__init__(config)
        self.lm_head = self.proj_out  # Alias for compatibility
        
    @property
    def model(self):
        """Return self for compatibility with some HF utilities."""
        return self

    def get_output_embeddings(self):
        return self.proj_out

    def set_output_embeddings(self, new_embeddings):
        self.proj_out = new_embeddings

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        hidden_states: Optional[List[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.Tensor]] = None,
        use_cache: bool = True,
        **kwargs,
    ) -> LNNModelOutput:
        """Forward pass that's compatible with CausalLM interface."""
        return super().forward(
            input_ids=input_ids,
            labels=labels,
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs
        )

# --- 7. Model registration ---
# Register the model with transformers
try:
    from transformers import AutoModel, AutoModelForCausalLM
    AutoModel.register(LNNConfig, LNNModel)
    AutoModelForCausalLM.register(LNNConfig, LNNForCausalLM)
except ImportError:
    pass  # transformers not available or version doesn't support registration
