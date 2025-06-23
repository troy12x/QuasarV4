import argparse
import torch
import torch.nn as nn
import pandas as pd
import math
import os
import sys
import inspect

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the canonical LNN model and config from our creation script
from create_lnn_model import LNNModel, ModelConfig as LNNConfig

# --- Transformer-specific definitions ---

class TransformerConfig:
    VOCAB_SIZE = 32000
    MAX_SEQ_LEN = 512
    HIDDEN_SIZE = 768
    NUM_LAYERS = 12
    NUM_HEADS = 12
    FFN_DIM = 3072  # 4 * HIDDEN_SIZE

class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = nn.MultiheadAttention(config.HIDDEN_SIZE, config.NUM_HEADS, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(config.HIDDEN_SIZE, config.FFN_DIM),
            nn.GELU(),
            nn.Linear(config.FFN_DIM, config.HIDDEN_SIZE)
        )
        self.norm1 = nn.LayerNorm(config.HIDDEN_SIZE)
        self.norm2 = nn.LayerNorm(config.HIDDEN_SIZE)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

class TransformerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_SIZE)
        self.pos_embeddings = nn.Embedding(config.MAX_SEQ_LEN, config.HIDDEN_SIZE)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.NUM_LAYERS)])
        self.output_head = nn.Linear(config.HIDDEN_SIZE, config.VOCAB_SIZE, bias=False)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embeddings(input_ids) + self.pos_embeddings(positions)
        for layer in self.layers:
            x = layer(x)
        return self.output_head(x)

# --- Analysis Functions ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_flops(model, input_shape):
    """A simplified FLOPs calculation."""
    batch_size, seq_len = input_shape
    C = model.config.HIDDEN_SIZE
    L = model.config.NUM_LAYERS

    if isinstance(model, TransformerModel):
        H = model.config.NUM_HEADS
        # Attention FLOPs (per block for Transformer)
        attn_flops = L * (4 * C**2 * seq_len + 2 * seq_len**2 * C)
        # FFN FLOPs (per block)
        ffn_flops = L * (2 * C * model.config.FFN_DIM) * seq_len
        total_flops = attn_flops + ffn_flops
        model_type = "Transformer"
    elif isinstance(model, LNNModel):
        # LNN FLOPs (per block)
        # W*h + U*u: 2*C*C + 2*C*C = 4*C^2 per step
        lnn_flops = L * (4 * C**2) * seq_len
        total_flops = lnn_flops
        model_type = "LNN"
    else:
        raise TypeError(f"Unknown model type: {type(model)}")
    return total_flops, model_type

def analyze_gradient_flow(model, input_shape):
    """Performs a forward/backward pass and returns gradient norms."""
    input_tensor = torch.randint(0, model.config.VOCAB_SIZE, input_shape, device='cpu')
    model.train()
    output = model(input_tensor)
    loss = output.mean()
    loss.backward()

    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
    
    return grad_norms

# --- Main Execution ---

def main():
    # --- Define Model Configurations ---
    VOCAB_SIZE = 32000 # Define a shared vocab size for a fair comparison

    # 1. Base Transformer Configuration (the original comparison)
    transformer_base_config = TransformerConfig()
    transformer_base_config.VOCAB_SIZE = VOCAB_SIZE
    transformer_base = TransformerModel(transformer_base_config)

    # 2. LNN Configuration (our efficiency target)
    lnn_config = LNNConfig(vocab_size=VOCAB_SIZE)
    lnn_model = LNNModel(lnn_config)

    # 3. Small Transformer (parameter-matched to the LNN)
    transformer_small_config = TransformerConfig()
    transformer_small_config.VOCAB_SIZE = VOCAB_SIZE
    transformer_small_config.HIDDEN_SIZE = 512
    transformer_small_config.NUM_HEADS = 8
    transformer_small_config.FFN_DIM = 2048 # 4 * 512
    transformer_small = TransformerModel(transformer_small_config)

    models = {
        "Transformer (Base)": transformer_base,
        "Transformer (Small)": transformer_small,
        "LNN (Base)": lnn_model
    }

    results = []
    # Since MAX_SEQ_LEN is the same for all, we can use any config
    input_shape = (1, transformer_base_config.MAX_SEQ_LEN) # Batch size of 1

    print("--- Architectural Analysis (Fair Comparison) ---")
    for name, model in models.items():
        params = count_parameters(model) / 1e6 # In millions
        flops, _ = calculate_flops(model, input_shape)
        flops_g = flops / 1e9 # In GFLOPs
        
        grad_norms = analyze_gradient_flow(model, input_shape)
        avg_grad_norm = sum(grad_norms.values()) / len(grad_norms)

        results.append({
            "Architecture": name,
            "Parameters (M)": f"{params:.2f}",
            "Forward FLOPs (G)": f"{flops_g:.2f}",
            "Avg. Grad Norm": f"{avg_grad_norm:.2e}"
        })

    # Print results table
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("\nNotes:")
    print("- FLOPs are an estimate for a single forward pass.")
    print("- Avg. Grad Norm indicates initial trainability. Higher is not always better, but very low values can indicate vanishing gradients.")

if __name__ == "__main__":
    try:
        main()
    except ImportError:
        print("\nERROR: Could not import LNNCell.")
        print("Please ensure 'quasar/lnn.py' exists and contains the LNNCell class.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
