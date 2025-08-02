# Quasar: A Liquid Neural Network Language Model

This repository contains the implementation of Quasar, a novel language model architecture centered around Liquid Neural Networks (LNNs). It's designed to process sequential data with continuous-time dynamics, offering an alternative to traditional attention-based Transformer models. The architecture is augmented with a Mixture of Experts (MoE) layer for efficient scaling and a Parameter Memory Bank (PMB) for long-term memory.

## Technical Deep Dive: The Liquid Neural Network (LNN) Core

The LNN is the heart of the model. Unlike traditional RNNs that operate on discrete time steps, LNNs are defined by Ordinary Differential Equations (ODEs), allowing them to handle sequences with a continuous-time perspective.

### 1. The Core Concept: Continuous-Time Dynamics via ODEs

Traditional Recurrent Neural Networks (RNNs) update their hidden state at discrete time steps (e.g., `h_t = f(h_{t-1}, x_t)`). The LNN, however, is based on an **Ordinary Differential Equation (ODE)**. Its state `h` evolves *continuously* over time, governed by the equation:

`dh/dt = f(h, u)`

-   `h`: The hidden state of the neuron (a vector).
-   `u`: The input to the neuron (a vector).
-   `dh/dt`: The instantaneous rate of change of the hidden state.
-   `f(...)`: A function, learned by the network, that defines the dynamics.

This means the LNN doesn't just jump from state to state; it models the *flow* between them.

### 2. `LNNCell`: The Heart of the Dynamics

The `LNNCell` class in `quasar/lnn.py` defines the function `f(...)` from the ODE. Its `forward` method calculates `dh/dt` (named `dx_dt` in the code):

```python
# From LNNCell.forward
def forward(self, h, u):
    # ... (calculations for time constants) ...

    # Core ODE dynamics calculation
    dx_dt = tau_w_inv * (self.sigma(torch.matmul(h, self.W.T) + torch.matmul(u, self.U.T) + self.b) - h) \
            - tau_b_inv * h

    return dx_dt
```

This equation combines the current hidden state `h` and input `u` with learned weights (`W`, `U`, `b`) and input-dependent time constants (`tau_w_inv`, `tau_b_inv`) to determine the state's instantaneous rate of change.

### 3. `LNNBlock`: Processing a Sequence

To process a sequence, the `LNNBlock` integrates the ODE over time using the **fixed-step Euler method**. It iterates through the input sequence, updating the hidden state at each step.

```python
# From LNNBlock.forward
def forward(self, x: torch.Tensor, h: torch.Tensor):
    # ...
    for t in range(seq_len):
        u = x[:, t, :]          # Get the input for this time step
        dx_dt = self.cell(h, u)   # 1. Calculate the rate of change
        h = h + self.dt * dx_dt   # 2. Update the state via Euler's method
        outputs[:, t, :] = h
    # ...
    output = self.ln(outputs + x) # 3. Apply residual connection and LayerNorm
    return output, h
```

In summary, the LNN core works by:
1.  Defining the continuous-time dynamics of a neuron as an ODE within the **`LNNCell`**.
2.  Using the **`LNNBlock`** to simulate these dynamics over a discrete input sequence by repeatedly calculating the rate of change and updating the state with a numerical solver.

## Other Key Architectural Components

### Mixture of Experts (MoE) (`quasar/moe.py`)

To scale the model to a very high parameter count without a proportional increase in computational cost, Quasar employs a Mixture of Experts layer.

-   **`Expert`**: A simple feed-forward neural network.
-   **`MoERouter`**: A lightweight gating network that learns to dynamically dispatch each input token to a small subset (`top_k`) of the best-suited experts.
-   **`MoELayer`**: This layer combines the router and the experts. For any given token, only the `top_k` selected experts are activated. This allows the model to have a massive number of total parameters while keeping the number of active parameters for each token much lower.

### Parameter Memory Bank (PMB) (`quasar/pmb.py`)

The PMB is a sophisticated memory system designed to provide the model with access to a vast, queryable knowledge base.

-   **Direct Access (O(1) Retrieval)**: It uses a two-level hashing system to store and retrieve key-value pairs in constant time using a unique ID.
-   **Semantic Search**: The PMB stores embeddings for keys and can retrieve the `top_k` most semantically similar items to a given query embedding.

## Model Implementations

This repository contains two primary model implementations:

1.  **`LNNModel` (`quasar/lnn.py`)**: The main Liquid Neural Network model. It stacks `LNNBlock`s to process sequences recurrently.

2.  **`Quasar` (`quasar/model.py`)**: A powerful language model based on the more traditional Transformer architecture, using the `MoELayer` in each block to scale efficiently.

## Project Structure

```
quasar/
├── quasar/
│   ├── __init__.py
│   ├── lnn.py           # Liquid Neural Network Model (LNNModel)
│   ├── model.py         # Transformer + MoE Model (Quasar)
│   ├── moe.py           # Mixture of Experts Layer
│   ├── pmb.py           # Parameter Memory Bank
│   └── utils.py         # Utility functions
├── scripts/
│   ├── generate_text.py
│   └── lnn_pretrain.py
├── .gitignore
├── requirements.txt
└── README.md
```

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd quasar
    ```

2.  Install the required packages. It is highly recommended to do this in a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## Usage

Scripts for training and generation are located in the `scripts/` directory.

### Training

To pre-train the LNN model, you can use the `lnn_pretrain.py` script:

```bash
python scripts/lnn_pretrain.py
```

### Text Generation

To generate text with a trained model, use the `generate_text.py` script:

```bash
python scripts/generate_text.py --prompt "Your starting prompt here"
```
