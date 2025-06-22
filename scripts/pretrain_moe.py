# c:\quasarv4\scripts\pretrain_moe.py

import torch
import os
import argparse
from datetime import datetime
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset
from torch.utils.data import DataLoader
import wandb
from accelerate import Accelerator, init_empty_weights, notebook_launcher
from accelerate.utils import FullyShardedDataParallelPlugin, ProjectConfiguration
from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from huggingface_hub import HfApi, create_repo
from tqdm import tqdm
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.model import Quasar, QuasarConfig

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
    # --- 1. Initialize Accelerator ---
    # --- Setup Accelerator with FSDP for Large Model Training ---
    # FSDP shards the model and optimizer across GPUs, and CPU Offload uses system RAM
    # to further reduce GPU memory usage, preventing out-of-memory errors.
    fsdp_plugin = FullyShardedDataParallelPlugin(cpu_offload=CPUOffload(offload_params=True))
    project_config = ProjectConfiguration(project_dir="./outputs", logging_dir="./outputs/logs")
    
    accelerator = Accelerator(
        log_with="wandb",
        mixed_precision="bf16",  # Use bfloat16 for faster training and less memory
        fsdp_plugin=fsdp_plugin,
        project_config=project_config
    )
    device = accelerator.device

    # --- 1. Load Tokenizer ---
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, token=args.hf_token, trust_remote_code=True)
    
    # Add a pad token if one doesn't exist. This is required for batched training.
    if tokenizer.pad_token is None:
        # If no pad token exists, use the EOS token for padding.
        # This is a common practice for autoregressive models.
        tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # --- 3. Load and Prepare Dataset ---
    if accelerator.is_main_process:
        print(f"Loading dataset '{args.dataset_name}'...")
    dataset = load_dataset(args.dataset_name, split='train')

    def tokenize_function(examples):
        # Pad all sequences to the maximum length to ensure uniform tensor shapes.
        return tokenizer(examples["text"], truncation=True, padding='max_length', max_length=args.seq_length)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_dataset, batch_size=args.batch_size, collate_fn=data_collator, shuffle=True)

    # --- 4. Initialize Quasar MoE Model using QuasarConfig ---
    if accelerator.is_main_process:
        print("Initializing QuasarV4 with MoE architecture...")

    # Create a config object from arguments
    model_config = QuasarConfig(
        vocab_size=vocab_size,
        embedding_dim=args.embedding_dim,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_experts=args.num_experts,
        expert_dim=args.expert_dim,
        top_k=args.top_k
    )

    with init_empty_weights():
        # Create a meta model for fast parameter counting
        meta_model = Quasar(model_config)

    # Print parameter counts based on meta model (fast, no memory overhead)
    if accelerator.is_main_process:
        print_model_parameters(meta_model)

    # Instantiate the real model on CPU (or default device) for training
    model = Quasar(model_config)

    # Manually enable gradient checkpointing. This is a workaround for a library issue
    # where our model's declared support for this feature is not being recognized.
    model.gradient_checkpointing = True

    # --- 4. Initialize W&B Tracker ---
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=args.wandb_project,
            config=vars(args),
            init_kwargs={"wandb": {"name": args.wandb_run_name}}
        )

    # --- 5. Prepare Model, then Set up Optimizer ---
    # When using `init_empty_weights`, the model must be prepared *before* the optimizer is created.
    # --- 5. Prepare for Distributed Training ---

    if accelerator.is_main_process:
        print("--- Checking Model Parameter Devices Before Preparation ---")
        for name, param in model.named_parameters():
            print(f"Parameter: {name:<60} Device: {param.device}")
        for name, buf in model.named_buffers():
            print(f"Buffer:    {name:<60} Device: {buf.device}")
        print("-----------------------------------------------------")

    # First, prepare the model. This moves it from the meta device to the target device(s).
    model = accelerator.prepare(model)

    # Now that the model is on the correct device, we can create the optimizer.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Then, prepare the optimizer and dataloader.
    optimizer, train_dataloader = accelerator.prepare(optimizer, train_dataloader)

    if accelerator.is_main_process:
        # Print device placement directly to the console
        print(f"--- Log entry at {datetime.now()} ---")
        print("--- Checking Model Parameter Devices after preparation ---")
        for name, param in model.named_parameters():
            print(f"Parameter: {name:<60} Device: {param.device}")
        for name, buf in model.named_buffers():
            print(f"Buffer:    {name:<60} Device: {buf.device}")
        print("--------------------------------------------------------\n")
        
    if accelerator.is_main_process:
        print_model_parameters(model)

    # Use tokenizer's vocab size (consistent before/after DDP wrapping)
    vocab_size = tokenizer.vocab_size

    # --- 7. Training Loop ---
    total_steps = len(train_dataloader) * args.num_epochs
    progress_bar = tqdm(range(total_steps), disable=not accelerator.is_main_process)

    for epoch in range(args.num_epochs):
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
            # The DataCollatorForLanguageModeling automatically creates the 'labels' field.
            # The model's forward pass accepts all items in the batch dictionary.
            outputs = model(**batch)

            # Extract losses from the model output
            total_loss = outputs['loss']
            load_balancing_loss = outputs['lb_loss']
            
            # For logging, we can derive the main cross-entropy loss
            main_loss = total_loss - load_balancing_loss

            accelerator.backward(total_loss)
            
            if args.max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)

            # Save checkpoint at step 1 and push to Hub
            if step == 1 and accelerator.is_main_process:
                print("\nSaving checkpoint at step 1...")
                checkpoint_dir = os.path.join(args.output_dir, "step_1_checkpoint")
                accelerator.save_state(checkpoint_dir)

                print("Pushing model to Hub...")
                try:
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        checkpoint_dir,
                        state_dict=accelerator.get_state_dict(model),
                        safe_serialization=True
                    )
                    tokenizer.save_pretrained(checkpoint_dir)
                    
                    api = HfApi()
                    api.upload_folder(
                        folder_path=checkpoint_dir,
                        repo_id=args.hf_repo,
                        repo_type="model",
                        token=args.hf_token,
                        commit_message="WIP: Add 400B model checkpoint at step 1"
                    )
                    print(f"Successfully pushed model to {args.hf_repo}")
                except Exception as e:
                    print(f"Error pushing to hub: {e}")

            # Log metrics
            if step % args.log_interval == 0:
                accelerator.log({
                    "train_loss": total_loss.item(),
                    "main_loss": main_loss.item(),
                    "load_balancing_loss": load_balancing_loss.item(),
                    "learning_rate": optimizer.param_groups[0]['lr']
                }, step=step)

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
    # Model Args (400B Configuration)
    parser.add_argument('--embedding_dim', type=int, default=8192, help='Embedding dimension')
    parser.add_argument('--num_hidden_layers', type=int, default=96, help='Number of hidden layers')
    parser.add_argument('--num_attention_heads', type=int, default=64, help='Number of attention heads')
    parser.add_argument('--num_experts', type=int, default=128, help='Number of experts')
    parser.add_argument('--expert_dim', type=int, default=2048, help='Expert dimension')
    parser.add_argument('--top_k', type=int, default=4, help='Top-k routing for MoE')
    # Training Args
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per GPU')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate')
    parser.add_argument('--seq_length', type=int, default=1024, help='Sequence length')
    parser.add_argument('--lb_lambda', type=float, default=0.01, help='Lambda for load balancing loss')
    parser.add_argument('--log_interval', type=int, default=1, help='Logging interval') # Log every step
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Max gradient norm for clipping.')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Directory to save checkpoints.')
    # Data and Hub Args
    parser.add_argument('--tokenizer_path', type=str, default='deepseek-ai/DeepSeek-V3-0324', help='Path or Hub ID for tokenizer')
    parser.add_argument("--dataset_name", type=str, default="SharedBailii/bailii-pretraining-order", help="Name of the dataset to use.")
    parser.add_argument('--hf_repo', type=str, default='silx-ai/QuasarV4-400B-1M', help='Hugging Face Hub repo')
    parser.add_argument('--hf_token', type=str, default='', help='Hugging Face Hub token')
    # W&B Args
    parser.add_argument('--wandb_project', type=str, default='quasar-400b-pretraining', help='W&B project name')
    parser.add_argument('--wandb_run_name', type=str, default=f'run-{datetime.now().strftime("%Y%m%d-%H%M%S")}', help='W&B run name')

    args = parser.parse_args()
    main(args)
