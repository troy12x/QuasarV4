"""
Script to create and save the QuasarV4 MoE model without training.
"""

import torch
import os
import sys
from tqdm import tqdm

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer
import argparse
from quasar.lnn import LNNModel, LNNConfig
import math
# Set the default dtype to bfloat16 to build the model with memory-efficient parameters from the start
torch.set_default_dtype(torch.bfloat16)
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
        hidden_size=args.embedding_dim,
        num_hidden_layers=args.num_hidden_layers,
        activation='gelu',
        use_pmb=args.use_pmb,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        num_experts_per_tok=args.num_experts_per_tok,
        expert_dim=args.expert_dim,
        torch_dtype=torch.bfloat16
    )
    print("\n--- LNN Model Configuration ---")
    print(f"  - Vocabulary Size:   {model_config.vocab_size}")
    print(f"  - Hidden Layers:     {model_config.num_hidden_layers}")
    print(f"  - Hidden Size:       {model_config.hidden_size}")
    print(f"  - Activation:        '{model_config.activation}'")
    print(f"  - PMB Enabled:       {model_config.use_pmb}")
    if model_config.use_moe:
        print("  --- MoE Configuration ---")
        print(f"    - MoE Enabled:         {model_config.use_moe}")
        print(f"    - Num Experts:         {model_config.num_experts}")
        print(f"    - Experts per Token:   {model_config.num_experts_per_tok}")
        print(f"    - Expert Dimension:    {model_config.expert_dim}")

        # --- Pre-calculation of Parameters ---
        embedding_params = model_config.vocab_size * model_config.hidden_size
        lnn_cell_params = model_config.num_hidden_layers * (2 * model_config.hidden_size * model_config.hidden_size)
        moe_params = (
            model_config.num_hidden_layers *
            model_config.num_experts *
            (2 * model_config.hidden_size * model_config.expert_dim)
        )
        total_params_estimate = embedding_params + lnn_cell_params + moe_params

        active_expert_params = (
            model_config.num_hidden_layers *
            model_config.num_experts_per_tok *
            (2 * model_config.hidden_size * model_config.expert_dim)
        )
        active_params_estimate = embedding_params + lnn_cell_params + active_expert_params
        
        print(f"    ---------------------------------")
        print(f"    - Est. Total Params:   {total_params_estimate/1e9:.2f}B")
        print(f"    - Est. Active Params:  {active_params_estimate/1e9:.2f}B")
    print("---------------------------\n")

    # Initialize model and convert to bfloat16, with progress bar
    print("\nInitializing model and converting to bfloat16...")
    model = LNNModel(model_config).to(torch.bfloat16)

    # --- Parameter and Storage Calculation ---
    print("\n--- Model Size and Storage Estimation ---")

    # Calculate parameters from the model itself for accuracy
    total_params = sum(p.numel() for p in model.parameters())
    embedding_params = model.embedding.weight.numel()
    lnn_cell_params = sum(p.numel() for name, p in model.named_parameters() if 'cell' in name)

    if model_config.use_moe:
        moe_params = sum(p.numel() for name, p in model.named_parameters() if 'moe_layer' in name)
        active_expert_params = (
            model_config.num_hidden_layers *
            model_config.num_experts_per_tok *
            (2 * model_config.hidden_size * model_config.expert_dim) # From one expert
        )
        # Active params = Non-MoE params + Active-MoE params
        non_moe_params = total_params - moe_params
        active_params = non_moe_params + active_expert_params
    else:
        moe_params = 0
        active_params = total_params

    # Storage estimation
    bytes_per_param = 2  # torch.bfloat16
    total_size_bytes = total_params * bytes_per_param
    total_size_gb = total_size_bytes / (1024**3)
    max_shard_size_gb = 5  # Matching the save_pretrained call
    num_shards = math.ceil(total_size_gb / max_shard_size_gb)

    print(f"--- Parameter Breakdown ---")
    print(f"Embedding Parameters:     {embedding_params/1e9:.2f}B")
    print(f"LNN Cell Parameters:      {lnn_cell_params/1e9:.2f}B")
    if model_config.use_moe:
        print(f"Total MoE Parameters:     {moe_params/1e9:.2f}B")
    print(f"---------------------------------")
    print(f"Total Parameters:         {total_params/1e9:.2f}B")
    if model_config.use_moe:
        print(f"Active Parameters:        {active_params/1e9:.2f}B")
    print(f"---------------------------------")
    print(f"Model Data Type:          torch.bfloat16")
    print(f"Estimated Total Size:     {total_size_gb:.2f} GB")
    print(f"Max Shard Size:           {max_shard_size_gb} GB")
    print(f"ESTIMATED SHARD COUNT:    {num_shards}")
    print("-----------------------------------------\n")
    
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
    
    # LNN Specific Args
    parser.add_argument('--num_hidden_layers', type=int, default=96, help='Number of hidden layers in the LNN')
    parser.add_argument('--embedding_dim', type=int, default=8192, help='Embedding and hidden dimension size')
    parser.add_argument('--no-pmb', dest='use_pmb', action='store_false', help='Disable the Parameter Memory Bank (enabled by default)')
    
    # MoE Specific Args
    parser.add_argument('--use-moe', action='store_true', default=True, help='Enable the Mixture of Experts layers (enabled by default for 440B model)')
    parser.add_argument('--num-experts', type=int, default=128, help='Number of experts in each MoE layer for the 440B model')
    parser.add_argument('--num-experts-per-tok', type=int, default=2, help='Number of experts to route each token to')
    parser.add_argument('--expert-dim', type=int, default=2048, help='Dimension of the expert feed-forward networks (16K recommended for faster initialization)')
    
    # Tokenizer and Output Args
    parser.add_argument('--tokenizer_path', type=str, default='deepseek-ai/DeepSeek-V3-0324', help='Path or Hub ID for tokenizer')
    parser.add_argument('--output_dir', type=str, default='./model', help='Directory to save the model')
    parser.add_argument('--hf_token', type=str, default='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', help='Hugging Face Hub token')
    
    args = parser.parse_args()
    create_and_save_model(args)

if __name__ == '__main__':
    main()
