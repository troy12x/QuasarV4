# c:\quasarv4\scripts\pretrain_moe.py

import torch
import os
import argparse
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from accelerate import Accelerator
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.model import Quasar

def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expert_params = 0
    if hasattr(model, 'moe_layer'):
        expert_params = sum(p.numel() for p in model.moe_layer.parameters())
    
    print("\n--- Model Parameter-Count ---")
    print(f"Total Parameters:      {total_params/1e9:.2f}B")
    print(f"Trainable Parameters:  {trainable_params/1e9:.2f}B")
    if expert_params > 0:
        print(f"Expert Parameters:     {expert_params/1e9:.2f}B")
        print(f"Shared Parameters:     {(total_params - expert_params)/1e9:.2f}B")
    print("-----------------------------\n")

def main(args):
    # --- 1. Initialize Accelerator for distributed training ---
    accelerator = Accelerator()
    device = accelerator.device

    # --- 2. Set up Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    vocab_size = len(tokenizer)

    # --- 3. Load and Prepare Dataset ---
    if accelerator.is_main_process:
        print(f"Loading dataset '{args.dataset_name}' with subset '{args.dataset_subset}'...")
    dataset = load_dataset(args.dataset_name, args.dataset_subset, split='train')

    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=args.seq_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    train_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, shuffle=True)

    # --- 4. Initialize Quasar MoE Model ---
    if accelerator.is_main_process:
        print("Initializing QuasarV4 with MoE architecture...")
    model = Quasar(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        top_k=args.top_k
    )
    model.embedding.weight.data.uniform_(-0.1, 0.1)

    if accelerator.is_main_process:
        print_model_parameters(model)

    # --- 5. Set up Optimizer ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # --- 6. Prepare for Distributed Training with Accelerator ---
    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    # --- 7. Training Loop ---
    criterion = torch.nn.CrossEntropyLoss()
    total_steps = len(train_dataloader) * args.num_epochs
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):
            inputs = batch['input_ids'][:, :-1]
            targets = batch['input_ids'][:, 1:]

            optimizer.zero_grad()
            
            logits, load_balancing_loss = model(inputs)
            
            # Main language modeling loss
            main_loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            
            # Total loss with MoE load balancing
            total_loss = main_loss + args.lb_lambda * load_balancing_loss
            
            accelerator.backward(total_loss)
            optimizer.step()
            progress_bar.update(1)

            if step % args.log_interval == 0 and accelerator.is_main_process:
                print(f"Epoch {epoch+1}/{args.num_epochs} | Step {step}/{len(train_dataloader)} | Total Loss: {total_loss.item():.4f} | Main Loss: {main_loss.item():.4f} | LB Loss: {load_balancing_loss.item():.4f}")

    # --- 8. Save and Upload Model to Hugging Face Hub ---
    if accelerator.is_main_process:
        print("Training complete. Saving model...")
        unwrapped_model = accelerator.unwrap_model(model)
        
        # Create repo if it doesn't exist
        try:
            create_repo(args.hf_repo, token=args.hf_token, repo_type='model', exist_ok=True)
            print(f"Repository '{args.hf_repo}' created or already exists.")
        except Exception as e:
            print(f"Could not create repository: {e}")

        # Save model in safetensors format and tokenizer
        unwrapped_model.save_pretrained(
            args.hf_repo, 
            safe_serialization=True, 
            token=args.hf_token,
            push_to_hub=True
        )
        tokenizer.save_pretrained(args.hf_repo, push_to_hub=True, token=args.hf_token)
        print(f"Model and tokenizer successfully uploaded to {args.hf_repo}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pre-train QuasarV4 MoE Model")
    # Model Args
    parser.add_argument('--embedding_dim', type=int, default=1024, help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=4096, help='Hidden dimension of LNN')
    parser.add_argument('--num_experts', type=int, default=64, help='Number of experts in MoE')
    parser.add_argument('--expert_dim', type=int, default=2048, help='Dimension of each expert')
    parser.add_argument('--top_k', type=int, default=2, help='Number of activated experts per token')
    # Training Args
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--lb_lambda', type=float, default=0.01, help='Lambda for load balancing loss')
    parser.add_argument('--log_interval', type=int, default=100, help='Logging interval')
    # Data and Hub Args
    parser.add_argument('--tokenizer_path', type=str, default='c:/quasarv4', help='Path to local tokenizer files')
    parser.add_argument('--dataset_name', type=str, default='HuggingFaceTB/smoltalk', help='Dataset name')
    parser.add_argument('--dataset_subset', type=str, default='everyday-conversations', help='Dataset subset')
    parser.add_argument('--hf_repo', type=str, default='silx-ai/QuasarV4-400B-1M', help='Hugging Face Hub repo')
    parser.add_argument('--hf_token', type=str, default='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', help='Hugging Face Hub token')

    args = parser.parse_args()
    main(args)
