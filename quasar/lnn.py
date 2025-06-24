# c:\quasarv4\quasar\lnn.py

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

# --- 1. LNN Configuration Class ---
class LNNConfig(PretrainedConfig):
    """Configuration class for the LNNModel."""
    model_type = "lnn"

    def __init__(
        self,
        vocab_size=151552,
        hidden_size=8192,
        num_hidden_layers=96,
        activation='gelu',
        lambda_res=0.1,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.lambda_res = lambda_res
        super().__init__(**kwargs)

# --- 2. Core LNN Cell (The "Real" Implementation) ---
class LNNCell(nn.Module):
    """A single Liquid Neural Network cell with continuous-time dynamics."""
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.lambda_res = config.lambda_res

        self.alpha = nn.Parameter(torch.empty(config.hidden_size))
        self.W = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.U = nn.Parameter(torch.empty(config.hidden_size, config.hidden_size))
        self.b = nn.Parameter(torch.empty(config.hidden_size))

        if config.activation == 'gelu':
            self.sigma = nn.GELU()
        else:
            self.sigma = torch.tanh

    def forward(self, h, u):
        """The ODE function dx/dt = f(t, x, u)."""
        dx_dt = -self.alpha * h + self.sigma(h @ self.W.T + u @ self.U.T + self.b)
        if self.lambda_res > 0:
            dx_dt = dx_dt + self.lambda_res * u
        return dx_dt

# --- 3. LNN Block (Layer + Residual) ---
class LNNBlock(nn.Module):
    """A purely recurrent block using an LNN layer with a residual connection."""
    def __init__(self, config: LNNConfig):
        super().__init__()
        self.cell = LNNCell(config)
        self.norm = nn.LayerNorm(config.hidden_size)
        self.dt = 1.0 # Time step for Euler integration

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.cell.hidden_size, device=x.device, dtype=x.dtype)
        outputs = []
        for i in range(seq_len):
            dx_dt = self.cell(h=h, u=x[:, i, :])
            h = h + self.dt * dx_dt # Euler integration
            outputs.append(h.unsqueeze(1))
        
        return self.norm(x + torch.cat(outputs, dim=1))

# --- 4. Full, HF-Compatible LNN Model ---
class LNNModel(PreTrainedModel):
    """The complete LNN model, compatible with Hugging Face's `save_pretrained`."""
    config_class = LNNConfig
    _supports_gradient_checkpointing = True

    def __init__(self, config: LNNConfig):
        super().__init__(config)
        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([LNNBlock(config) for _ in range(config.num_hidden_layers)])
        self.final_norm = nn.LayerNorm(config.hidden_size)
        self.output_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embeddings(input_ids)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                 x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)

        x = self.final_norm(x)
        logits = self.output_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            logits_flat = logits.view(-1, self.config.vocab_size)
            labels_flat = labels.view(-1)
            loss = loss_fct(logits_flat, labels_flat)

        if loss is not None:
            return (loss, logits)
        else:
            return (logits,)
