"""
Script to create and save a pure LNN-based model.
"""

import torch
import os
import argparse
import sys
import pandas as pd
import torch.nn as nn
from transformers import AutoTokenizer

# Add project root to the Python path to allow importing from 'quasar'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the necessary components from the analysis script's logic
# This assumes the LNNCell is defined in 'quasar.lnn'
from quasar.lnn import LNNCell

# --- Model Definitions (Copied from analyze_architectures.py for self-containment) ---

class ModelConfig:
    def __init__(self, vocab_size, hidden_size=768, num_layers=12):
        self.VOCAB_SIZE = vocab_size
        self.HIDDEN_SIZE = hidden_size
        self.NUM_LAYERS = num_layers
        self.MAX_SEQ_LEN = 512 # Standard value

class LNNLayer(nn.Module):
    """A layer that wraps the LNNCell to process a sequence."""
    def __init__(self, hidden_size, activation='gelu'):
        super().__init__()
        self.cell = LNNCell(input_size=hidden_size, hidden_size=hidden_size, activation=activation)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        h = torch.zeros(batch_size, self.cell.hidden_size, device=x.device)
        outputs = []
        for i in range(seq_len):
            h = self.cell(t=i, x=h, u=x[:, i, :])
            outputs.append(h.unsqueeze(1))
        return torch.cat(outputs, dim=1)

class LNNBlock(nn.Module):
    """A purely recurrent block using an LNN layer with a residual connection."""
    def __init__(self, config):
        super().__init__()
        self.lnn = LNNLayer(config.HIDDEN_SIZE)
        self.norm = nn.LayerNorm(config.HIDDEN_SIZE)

    def forward(self, x):
        lnn_output = self.lnn(x)
        x = self.norm(x + lnn_output)
        return x

class LNNModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.VOCAB_SIZE, config.HIDDEN_SIZE)
        self.pos_embeddings = nn.Embedding(config.MAX_SEQ_LEN, config.HIDDEN_SIZE)
        self.layers = nn.ModuleList([LNNBlock(config) for _ in range(config.NUM_LAYERS)])
        self.output_head = nn.Linear(config.HIDDEN_SIZE, config.VOCAB_SIZE, bias=False)

    def forward(self, input_ids):
        seq_len = input_ids.shape[1]
        positions = torch.arange(0, seq_len, device=input_ids.device).unsqueeze(0)
        x = self.embeddings(input_ids) + self.pos_embeddings(positions)
        for layer in self.layers:
            x = layer(x)
        return self.output_head(x)

# --- Main Creation Function ---

def create_and_save_lnn_model(args):
    # Load tokenizer to get vocab size
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Create model configuration
    print("Creating LNN model configuration...")
    model_config = ModelConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers
    )

    # Initialize model
    print("Initializing LNN model...")
    model = LNNModel(model_config)
    model.eval() # Set to evaluation mode

    # --- Save the model --- 
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model to {output_dir}...")

    # This uses the 'transformers' library's save_pretrained method,
    # which handles sharding and creating the necessary config files.
    model.save_pretrained(output_dir, max_shard_size=args.max_shard_size)
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_dir)

    print("\n--- Model Parameter-Count ---")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params/1e6:.2f}M")
    print("-----------------------------")
    print(f"\nLNN Model created and saved successfully at: {output_dir}")

# --- Argument Parsing ---

def main():
    parser = argparse.ArgumentParser(description="Create and save a pure LNN-based model.")
    # Model Args
    parser.add_argument('--hidden_size', type=int, default=768, help='Hidden size of the model.')
    parser.add_argument('--num_layers', type=int, default=12, help='Number of LNN layers.')
    
    # Tokenizer and Output Args
    parser.add_argument('--tokenizer_path', type=str, default='gpt2', help='Path or Hub ID for the tokenizer.')
    parser.add_argument('--output_dir', type=str, default='./lnn_model', help='Directory to save the model.')
    parser.add_argument('--max_shard_size', type=str, default='1GB', help='Maximum size of each model shard.')

    args = parser.parse_args()
    create_and_save_lnn_model(args)

if __name__ == '__main__':
    main()
