import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class Quasar1ParameterLNNCell(nn.Module):
    """
    A Liquid Neural Network cell where the dynamics are governed by fixed,
    random matrices, and modulated by a single learnable scalar parameter.
    This cell is designed to be part of the Quasar1ParameterLNN.
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        
        # --- Fixed, Non-Trainable Components ---
        # Create matrices but immediately freeze them.
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=False)
        self.U = nn.Parameter(torch.empty(hidden_size, hidden_size), requires_grad=False)
        self.b = nn.Parameter(torch.empty(hidden_size), requires_grad=False)
        
        # Fixed layers for input-dependent time constant
        self.tau_w_h = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_w_u = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau_b = nn.Parameter(torch.empty(hidden_size), requires_grad=False)

        for param in self.tau_w_h.parameters():
            param.requires_grad = False
        for param in self.tau_w_u.parameters():
            param.requires_grad = False

        # --- Initialize Fixed Weights ---
        nn.init.orthogonal_(self.W)
        nn.init.xavier_uniform_(self.U)
        nn.init.zeros_(self.b)
        nn.init.xavier_uniform_(self.tau_w_h.weight)
        nn.init.xavier_uniform_(self.tau_w_u.weight)
        self.tau_b.data.uniform_(-2, 2)
        
        self.sigma = nn.Tanh()

    def forward(self, h, u, lambda_param):
        """
        The lambda_param is passed in from the main model.
        It's the single learnable parameter that controls the dynamics.
        """
        with torch.no_grad():
            tau_control = F.linear(h, self.tau_w_h.weight) + F.linear(u, self.tau_w_u.weight) + self.tau_b
            tau_positive = F.softplus(tau_control) + 0.01

        decay_term = -h / tau_positive
        activation_input = F.linear(h, self.W) + F.linear(u, self.U) + self.b
        activation_output = self.sigma(activation_input)
        
        dx_dt = decay_term + activation_output
        
        positive_lambda = F.softplus(lambda_param)
        dx_dt = positive_lambda * dx_dt
        
        return torch.clamp(dx_dt, -10, 10)

class Quasar1ParameterLNN(nn.Module):
    """
    An LNN where the recurrent core has its dynamics controlled by a single
    learnable parameter. The embedding and output layers remain trainable.
    """
    def __init__(self, vocab_size: int, hidden_size: int, num_layers: int, dt: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dt = dt

        # --- The Single Learnable Parameter for the LNN core ---
        self.lambda_param = nn.Parameter(torch.tensor(0.0))

        # --- Trainable Input/Output Layers ---
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.proj_out = nn.Linear(hidden_size, vocab_size)

        # --- Non-Trainable Recurrent Core ---
        self.cells = nn.ModuleList([Quasar1ParameterLNNCell(hidden_size) for _ in range(num_layers)])

    def forward(self, input_ids: torch.LongTensor, hidden_states: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len = input_ids.shape
        x = self.embedding(input_ids)

        if hidden_states is None:
            hidden_states = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=input_ids.device)

        all_hidden_outputs = []
        for t in range(seq_len):
            token_embedding = x[:, t, :]
            
            layer_input = token_embedding
            next_hidden_states = []
            for i in range(self.num_layers):
                h_prev = hidden_states[i]
                dx_dt = self.cells[i](h_prev, layer_input, self.lambda_param)
                h_next = h_prev + self.dt * dx_dt
                next_hidden_states.append(h_next)
                layer_input = h_next
            
            hidden_states = torch.stack(next_hidden_states)
            all_hidden_outputs.append(layer_input)

        output_sequence = torch.stack(all_hidden_outputs, dim=1)
        final_logits = self.proj_out(output_sequence)
        return final_logits, hidden_states

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
