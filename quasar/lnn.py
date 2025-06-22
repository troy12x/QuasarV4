# c:\quasarv4\quasar\lnn.py

import torch
import torch.nn as nn
from torchdiffeq import odeint

class LNNCell(nn.Module):
    """
    A single Liquid Neural Network cell that defines the continuous-time dynamics
    based on the equation:
    dx/dt = -alpha * x + sigma(W*x + U*u + b) + lambda * u
    """
    def __init__(self, input_size, hidden_size, activation='tanh', lambda_res=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lambda_res = lambda_res

        # Learnable parameters are created with torch.empty to be compatible with `init_empty_weights`.
        # The `accelerate` framework will handle the actual initialization.
        self.alpha = nn.Parameter(torch.empty(hidden_size))          # Time constants per neuron
        self.W = nn.Parameter(torch.empty(hidden_size, hidden_size)) # Recurrent weights
        self.U = nn.Parameter(torch.empty(hidden_size, input_size))  # Input weights
        self.b = nn.Parameter(torch.empty(hidden_size))              # Bias

        # Activation function
        if activation == 'tanh':
            self.sigma = torch.tanh
        elif activation == 'gelu':
            self.sigma = nn.GELU()
        else:
            raise NotImplementedError(f"Activation '{activation}' not supported.")
            
        # Note on block sparsity for W:
        # For a real implementation, W would be initialized with a block-sparse
        # structure and a mask would be applied during the forward pass to maintain sparsity.
        # For this initial version, we use a dense matrix for simplicity.

        # Note on residual connection:
        # The term `lambda * u` is added to the state dynamics. This is only
        # directly possible if input_size == hidden_size. If not, a projection
        # layer would be required. We will proceed assuming this condition holds
        # when lambda_res > 0.


    def forward(self, t, x, u):
        """
        The ODE function dx/dt = f(t, x, u).
        
        Args:
            t (torch.Tensor): Current time (unused, for compatibility with odeint).
            x (torch.Tensor): Hidden state of shape (batch_size, hidden_size).
            u (torch.Tensor): Input of shape (batch_size, input_size).
            
        Returns:
            torch.Tensor: The derivative of the hidden state, dx/dt.
        """
        # Main LNN dynamics
        dx_dt = -self.alpha * x + self.sigma(
            x @ self.W.T + u @ self.U.T + self.b
        )

        # Add residual connection from the input
        # Add residual connection only if dimensions match
        if self.lambda_res > 0 and x.shape[-1] == u.shape[-1]:
            dx_dt = dx_dt + self.lambda_res * u

        return dx_dt

class LNN(nn.Module):
    """
    Liquid Neural Network layer that processes a sequence of inputs by solving
    the underlying ODE for each step.
    """
    def __init__(self, input_size, hidden_size, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LNNCell(input_size, hidden_size, **kwargs)



    def forward(self, u, h0=None):
        """
        Processes a sequence of inputs through the LNN.
        
        Args:
            u (torch.Tensor): Input sequence of shape (seq_len, batch_size, input_size).
            h0 (torch.Tensor, optional): Initial hidden state. Defaults to zeros.
            
        Returns:
            torch.Tensor: Output sequence of hidden states of shape (seq_len, batch_size, hidden_size).
        """
        seq_len, batch_size, _ = u.shape

        # Meta device guard for `torchdiffeq.odeint` compatibility.
        if u.device.type == 'meta':
            return torch.zeros(seq_len, batch_size, self.hidden_size, device='meta')

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=self.cell.b.device)

        # Create t_span on the correct device, right before it's needed.
        t_span = torch.tensor([0.0, 1.0], device=h0.device)

        outputs = []
        h = h0

        for i in range(seq_len):
            ode_func = lambda t, x: self.cell(t, x, u[i])
            h_next = odeint(ode_func, h, t_span, method='rk4')[1]
            outputs.append(h_next)
            h = h_next

        return torch.stack(outputs, dim=0)
