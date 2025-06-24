"""
Script to create and save a pure LNN-based model using the standardized Quasar LNN module.
"""

import torch
import os
import argparse
import sys
from transformers import AutoTokenizer

# Add project root to the Python path to allow importing from 'quasar'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the new, standardized LNN components from quasar.lnn
from quasar.lnn import LNNConfig, LNNModel

def create_and_save_lnn_model(args):
    """
    Creates an LNN model using the specified configuration and saves it to disk.
    """
    # Load tokenizer to get vocab size
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Create model configuration using the new LNNConfig
    print("Creating LNN model configuration...")
    model_config = LNNConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        activation=args.activation,
        lambda_res=args.lambda_res,
        dt=args.dt,
        use_pmb=args.use_pmb
    )

    # Initialize model using the new LNNModel
    print("Initializing LNN model...")
    model = LNNModel(model_config)
    model.eval() # Set to evaluation mode

    # --- Save the model and tokenizer ---
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving model to {output_dir}...")

    # Use the 'transformers' library's save_pretrained method
    model.save_pretrained(output_dir, max_shard_size=args.max_shard_size)
    tokenizer.save_pretrained(output_dir)
    
    print(f"\nModel successfully saved to {output_dir}")
    print("Configuration (config.json), model weights, and tokenizer files have been written.")

    # --- Print Model Parameter-Count ---
    total_params = sum(p.numel() for p in model.parameters())
    print("\n--- Model Parameter-Count ---")
    print(f"Total Parameters: {total_params/1e6:.2f}M")
    print("-----------------------------")
    print(f"\nLNN Model created and saved successfully at: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Create and save a pure LNN model using the Quasar LNN module.")
    
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the pretrained tokenizer directory.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the new model.")
    
    # Model architecture arguments
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size of the model.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of LNN layers.")
    parser.add_argument("--activation", type=str, default='tanh', choices=['tanh', 'gelu'], help="Activation function to use.")
    parser.add_argument("--lambda_res", type=float, default=0.0, help="Strength of the residual connection in the LNN cell.")
    parser.add_argument("--dt", type=float, default=1.0, help="Time step for the Euler integration.")
    parser.add_argument('--max_shard_size', type=str, default='1GB', help='Maximum size of each model shard.')
    parser.add_argument('--use-pmb', action='store_true', help='Enable the Parameter Memory Bank in the model.')

    args = parser.parse_args()
    create_and_save_lnn_model(args)

if __name__ == '__main__':
    main()

