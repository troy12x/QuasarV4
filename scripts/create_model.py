"""
Script to create and save the QuasarV4 MoE model without training.
"""

import torch
import os
from transformers import AutoTokenizer
import argparse
from quasar.lnn import LNNModel, LNNConfig
from huggingface_hub import HfApi

def create_and_save_model(args):
    """Initializes and saves the LNN model."""

    # Load tokenizer to get vocab size
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Create LNN model configuration
    print("Creating LNN model configuration...")
    model_config = LNNConfig(
        vocab_size=vocab_size,
        num_hidden_layers=args.num_hidden_layers,
        hidden_size=args.embedding_dim, # Use embedding_dim for hidden_size
        activation='gelu'
    )
    print("\n--- LNN Model Configuration ---")
    print(f"  - Vocabulary Size:   {model_config.vocab_size}")
    print(f"  - Hidden Layers:     {model_config.num_hidden_layers}")
    print(f"  - Hidden Size:       {model_config.hidden_size}")
    print(f"  - Activation:        '{model_config.activation}'")
    print("---------------------------\n")

    # Initialize LNN model and convert to bfloat16
    print("Initializing LNN model and converting to bfloat16...")
    model = LNNModel(model_config).to(torch.bfloat16)

    # --- Storage Estimation ---
    print("\n--- Storage Estimation ---")
    total_params = sum(p.numel() for p in model.parameters())
    # torch.bfloat16 uses 2 bytes per parameter
    bytes_per_param = 2 
    total_size_bytes = total_params * bytes_per_param
    total_size_tb = total_size_bytes / (1000**4)

    # Parse the hardcoded max_shard_size from the save_pretrained call
    max_shard_size_str = "5GB"
    max_shard_bytes = int(max_shard_size_str[:-2]) * (10**9)

    import math
    estimated_shards = math.ceil(total_size_bytes / max_shard_bytes)

    print(f"Total Parameters:         {total_params/1e9:.2f}B")
    print(f"Model Data Type:          torch.bfloat16")
    print(f"Estimated Total Size:     {total_size_tb:.2f} TB")
    print(f"Max Shard Size:           {max_shard_size_str}")
    print(f"ESTIMATED SHARD COUNT:    {estimated_shards}")
    print("--------------------------\n")
    
    # Save the sharded model locally. This is the correct approach.
    # The model is saved once, creating all necessary shards in the output directory.
    print(f"\nSaving model shards to '{args.output_dir}'...")
    model.save_pretrained(args.output_dir, max_shard_size="5GB")
    print("Model shards saved successfully.")
    
    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained(args.output_dir, push_to_hub=True, token=args.hf_token)
    
    print("\nAll files processed successfully!")

    # Print final success message
    print(f"\nModel saved successfully to {args.output_dir}")
    
    # Print parameter counts
    print("\n--- Model Parameter-Count ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total Parameters:      {total_params/1e9:.2f}B")
    print(f"Trainable Parameters:  {trainable_params/1e9:.2f}B")
    print("-----------------------------")

def main():
    parser = argparse.ArgumentParser(description="Create and save QuasarV4 LNN Model")
    # Model Args (400B Configuration)
    parser.add_argument('--embedding_dim', type=int, default=8192, help='Dimension of the embedding and hidden layers.')
    parser.add_argument('--num_hidden_layers', type=int, default=96, help='Number of hidden layers')
    
    # Tokenizer and Output Args
    parser.add_argument('--tokenizer_path', type=str, default='deepseek-ai/DeepSeek-V3-0324', help='Path or Hub ID for tokenizer')
    parser.add_argument('--output_dir', type=str, default='./model', help='Directory to save the model')
    parser.add_argument('--hf_token', type=str, default='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', help='Hugging Face Hub token')
    
    args = parser.parse_args()
    create_and_save_model(args)

if __name__ == '__main__':
    main()
