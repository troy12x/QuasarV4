"""
Script to create and save the QuasarV4 MoE model without training.
"""

import torch
import os
from transformers import AutoTokenizer
import argparse
from quasar.model import Quasar, QuasarConfig
from huggingface_hub import HfApi

def create_and_save_model(args):
    # Load tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, token=args.hf_token, trust_remote_code=True)
    vocab_size = len(tokenizer)

    # Create model configuration
    print("Creating model configuration...")
    model_config = QuasarConfig(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        top_k=args.top_k
    )

    # Initialize model and convert to bfloat16 to reduce size by 50%
    print("Initializing model and converting to bfloat16...")
    # Using .to() is a safe way to convert the model's data type.
    model = Quasar(model_config).to(torch.bfloat16)

    # Save the sharded model locally. This is the correct approach.
    # The model is saved once, creating all necessary shards in the output directory.
    print(f"\nSaving model shards to '{args.output_dir}'...")
    model.save_pretrained(args.output_dir, max_shard_size="1GB")
    print("Model shards saved successfully.")
    
    # Save tokenizer
    print("\nSaving tokenizer...")
    tokenizer.save_pretrained(HF_REPO, push_to_hub=True, token=HF_TOKEN)
    
    print("\nAll files processed successfully!")

    # Print final success message
    print(f"\nModel saved successfully to {HF_REPO}")
    
    # Print parameter counts
    print("\n--- Model Parameter-Count ---")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expert_params = 0
    if hasattr(model, 'moe_layer'):
        expert_params = sum(p.numel() for p in model.moe_layer.parameters())
    
    print(f"Total Parameters:      {total_params/1e9:.2f}B")
    print(f"Trainable Parameters:  {trainable_params/1e9:.2f}B")
    if expert_params > 0:
        print(f"Expert Parameters:     {expert_params/1e9:.2f}B")
        print(f"Shared Parameters:     {(total_params - expert_params)/1e9:.2f}B")
    print("-----------------------------")

def main():
    parser = argparse.ArgumentParser(description="Create and save QuasarV4 MoE Model")
    # Model Args (400B Configuration)
    parser.add_argument('--embedding_dim', type=int, default=8192, help='Embedding dimension')
    parser.add_argument('--num_hidden_layers', type=int, default=96, help='Number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=64, help='Number of attention heads')
    parser.add_argument('--num_experts', type=int, default=128, help='Number of experts')
    parser.add_argument('--expert_dim', type=int, default=2048, help='Expert dimension')
    parser.add_argument('--top_k', type=int, default=4, help='Top-k routing for MoE')
    
    # Tokenizer and Output Args
    parser.add_argument('--tokenizer_path', type=str, default='deepseek-ai/DeepSeek-V3-0324', help='Path or Hub ID for tokenizer')
    parser.add_argument('--output_dir', type=str, default='./model', help='Directory to save the model')
    parser.add_argument('--hf_token', type=str, default='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', help='Hugging Face Hub token')
    
    args = parser.parse_args()
    create_and_save_model(args)

if __name__ == '__main__':
    main()
