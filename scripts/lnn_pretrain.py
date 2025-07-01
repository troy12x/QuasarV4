import os
import sys
import argparse
import shutil
import math
import time
from huggingface_hub import login
import logging
import torch



# Add project root to sys.path to allow for local package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import wandb
from tqdm.auto import tqdm
from transformers import AutoTokenizer, get_scheduler, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from itertools import chain

# Local imports
from quasar.lnn import LNNModel, LNNConfig
from huggingface_hub import create_repo

# Note: We will use a standard AdamW optimizer and the default data collator,
# so a custom utils file is no longer needed.
from torch.optim import AdamW


# --- Distributed Training Setup ---
def setup_distributed():
    """Initializes the distributed process group."""
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def is_main_process(single_gpu_mode=False):
    """Checks if the current process is the main process."""
    if single_gpu_mode:
        return True
    if not dist.is_initialized():
        return True
    return dist.get_rank() == 0

# --- Logging Setup ---
logger = logging.getLogger(__name__)


# --- Checkpoint Saving ---
def save_checkpoint(model, tokenizer, args, global_step):
    """Saves model, tokenizer, and arguments to a checkpoint directory."""
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Saving model checkpoint to {checkpoint_dir}")

    # Unwrap the model if using DDP
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save model and tokenizer using Hugging Face's `save_pretrained`
    # safe_serialization=False is required for models with tied weights like this one.
    model_to_save.save_pretrained(checkpoint_dir, safe_serialization=False)
    tokenizer.save_pretrained(checkpoint_dir)

    # Save training arguments for easy resuming
    torch.save(args, os.path.join(checkpoint_dir, "training_args.bin"))
    logger.info(f"Checkpoint saved successfully to {checkpoint_dir}")

def setup_logging(single_gpu_mode=False):
    """Sets up logging, restricting verbose logs to the main process."""
    log_level = logging.INFO if is_main_process(single_gpu_mode) else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale pre-training script for LNN")
    
    # --- Model & Tokenizer Arguments ---
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Load a pretrained LNN model. If None, create a new model from scratch.")
    parser.add_argument("--tokenizer_name", type=str, default="deepseek-ai/DeepSeek-V3", help="Tokenizer name or path.")
    parser.add_argument("--use_slow_tokenizer", action="store_true", help="Use slow tokenizer.")

    # --- Arguments for Creating a New Model ---
    parser.add_argument("--hidden_size", type=int, default=1536, help="Hidden size for new models.")
    parser.add_argument("--num_hidden_layers", type=int, default=10, help="Number of hidden layers for new models.")
    parser.add_argument("--dt", type=float, default=0.1, help="Integration step size (dt) for the LNN cells.")
    parser.add_argument("--use_moe", action="store_true", help="Enable Mixture of Experts layers when creating a new model.")

    # --- Data Arguments ---
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to tokenize from the Hub.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use.")
    parser.add_argument("--train_split_name", type=str, default="train", help="The name of the training data split to use.")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the column in the dataset containing the text.")
    parser.add_argument("--sequence_length", type=int, default=2048, help="The sequence length for packing the dataset.")

    # --- Training & Output Arguments ---
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4, help="Batch size per GPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    # Note: A cosine scheduler is often more effective for large model pre-training.
    # It helps in achieving a lower loss by annealing the learning rate smoothly.
    # The learning rate has been adjusted to a more standard value for pre-training.
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.95], help="Beta values for AdamW optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform. For large datasets, --max_train_steps is recommended.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps. If provided, this overrides num_train_epochs.")
    # Using a cosine scheduler with a longer warmup is a common best practice for pre-training.
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type (e.g., 'linear', 'cosine').")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Ratio of total training steps for linear warmup (e.g., 0.02 for 2% warmup).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended for H100).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    
    # Checkpointing & Logging
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Save checkpoint every N steps.")
    parser.add_argument("--logging_steps", type=int, default=5, help="Log every N steps.")
    parser.add_argument("--wandb_project", type=str, default="lnn-pretraining", help="Weights & Biases project name.")
    parser.add_argument("--single_gpu", action="store_true", help="Run on a single GPU without distributed training for testing.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached tokenized dataset if it exists.")

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push checkpoints to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The model ID (repository name) on the Hugging Face Hub.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Create a private repository on the Hub.")
    parser.add_argument("--debug_nan", action="store_true", help="Enable anomaly detection for debugging NaN loss.")

    args = parser.parse_args()
    return args

# --- Main Training Logic ---
def main():
    args = parse_args()
    
 

    # --- Setup based on mode (single GPU vs. distributed) ---
    if args.single_gpu:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
        setup_logging(single_gpu_mode=True)
    else:
        local_rank = setup_distributed()
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        setup_logging()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)

    if is_main_process(args.single_gpu):
        os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub and is_main_process(args.single_gpu):
            if args.hub_model_id is None:
                raise ValueError("Must specify --hub_model_id when pushing to the Hub.")
            print(f"Pushing checkpoints to repository: {args.hub_model_id}")
            # Use HUGGING_FACE_HUB_TOKEN environment variable or `huggingface-cli login`
            create_repo(args.hub_model_id, private=args.hub_private_repo, exist_ok=True)

    if is_main_process(args.single_gpu):
        wandb.init(project=args.wandb_project, config=args)

    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset Loading and On-the-Fly Tokenization ---
    # This section handles streaming data from the Hub, tokenizing, and packing it.
    if not args.dataset_name or not args.train_split_name:
        raise ValueError("Both --dataset_name and --train_split_name must be provided for on-the-fly tokenization.")

    # Use the standard `load_dataset` with streaming mode to handle large datasets
    # efficiently without downloading the entire dataset at once.
    logger.info(f"Loading dataset '{args.dataset_name}' with streaming.")
    raw_dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        split=args.train_split_name,
        streaming=True
    )
    # Rename the text column to 'text' for consistency if it's different
    if args.text_column != 'text':
        logger.info(f"Renaming data column '{args.text_column}' to 'text'.")
        raw_dataset = raw_dataset.rename_column(args.text_column, 'text')

    def tokenize_function(examples):
        # We don't truncate or pad here; we handle fixed-size chunks in the packing step.
        return tokenizer(examples['text'], truncation=False, padding=False)

    # Use a larger number of processes to speed up tokenization.
    logger.info("Tokenizing dataset...")
    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=['text']
    )

    def pack_iterator(dataset, sequence_length):
        buffer = []
        for example in dataset:
            ids = example['input_ids']
            buffer.extend(ids)
            while len(buffer) >= sequence_length:
                chunk = buffer[:sequence_length]
                buffer = buffer[sequence_length:]
                yield {"input_ids": chunk, "labels": chunk.copy()}

    packed_dataset = Dataset.from_generator(
        lambda: pack_iterator(tokenized_dataset, args.sequence_length)
    )
    train_dataset = packed_dataset
    train_dataset.set_format("torch")
    logger.info("Dataset processed and packed successfully.")

    eval_dataset = None # Evaluation not implemented

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    train_sampler = RandomSampler(train_dataset) if args.single_gpu else DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
        pin_memory=True,
        drop_last=True,
    )

    eval_dataloader = None
    # Evaluation logic is removed for simplicity to focus on the training loop.
    if eval_dataset:
        logger.warning("Evaluation is not implemented in this script.")

    logger.info("Initializing model...")
    if args.model_name_or_path:
        logger.info(f"Loading pretrained LNN model from: {args.model_name_or_path}")
        model = LNNModel.from_pretrained(args.model_name_or_path)
    else:
        logger.info(f"Creating a new LNN model from scratch with hidden_size={args.hidden_size} and num_layers={args.num_hidden_layers}")
        config = LNNConfig(
            vocab_size=len(tokenizer),
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            dt=args.dt,
            use_moe=args.use_moe,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = LNNModel(config)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = model.to(device)
    if not args.single_gpu:
        # Set find_unused_parameters=False as our model architecture is static.
        # This resolves the DDP warning and provides a minor performance boost.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Enable anomaly detection if the debug flag is set
    if args.debug_nan:
        logger.warning("Enabling anomaly detection for debugging. This will slow down training.")
        torch.autograd.set_detect_anomaly(True)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=args.adam_betas,
        weight_decay=args.weight_decay
    )
    
    # Calculate total training steps. If max_train_steps is provided, it overrides num_train_epochs.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        # If max_train_steps is set, calculate the number of epochs for logging purposes.
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Calculate warmup steps from ratio
    args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
    logger.info(f"Calculated warmup steps: {args.num_warmup_steps}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    global_step = 0
    start_epoch = 0

    if args.resume_from_checkpoint:
        checkpoint_dir = args.resume_from_checkpoint
        logger.info(f"Attempting to resume from checkpoint: {checkpoint_dir}")

        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            logger.info(f"Found full training state at {training_state_path}. Loading model, optimizer, and scheduler.")
            # weights_only=False is required here to load the optimizer and scheduler state.
            checkpoint = torch.load(training_state_path, map_location=device, weights_only=False)
            
            model_to_load = model.module if not args.single_gpu else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'])
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # The following line is commented out to prevent loading a bad scheduler state from a checkpoint.
            # This forces the training to use the new, correctly configured scheduler.
            # lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.warning("Skipping scheduler state loading from checkpoint to reset the learning rate.")
            
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            logger.info(f"Successfully resumed from epoch {start_epoch}, global step {global_step}. LR is {lr_scheduler.get_last_lr()[0]:.2e}")
        else:
            logger.warning(f"'training_state.pt' not found. Attempting to load model weights from 'pytorch_model.bin'.")
            weights_path = os.path.join(checkpoint_dir, "pytorch_model.bin")
            if os.path.exists(weights_path):
                model_to_load = model.module if not args.single_gpu else model
                # weights_only=True is safer as we only expect model weights here.
                state_dict = torch.load(weights_path, map_location=device, weights_only=True)
                model_to_load.load_state_dict(state_dict)
                logger.info(f"Successfully loaded model weights from {weights_path}. Optimizer and scheduler are not restored.")
            else:
                logger.error(f"FATAL: Could not find 'training_state.pt' or 'pytorch_model.bin' in {checkpoint_dir}. Cannot resume training.")
                sys.exit(1) # Exit because we cannot fulfill the resume request.

    logger.info("***** Starting Training *****")
    logger.info(f"  Total train batch size = {args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=args.bf16)
    
    completed_steps = global_step
    model.train()

    if is_main_process(args.single_gpu):
        progress_bar = tqdm(initial=completed_steps, total=args.max_train_steps, desc="Training Progress")

    # The outer loop is for epochs, but we break once max_train_steps is reached.
    for epoch in range(start_epoch, args.num_train_epochs):
        if not args.single_gpu:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            # If we've reached the max steps, break out of the inner loop
            if completed_steps >= args.max_train_steps:
                break

            # Move batch to device and separate labels
            labels = batch.pop("labels", None).to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss
            
            # Scale the loss for mixed precision and perform the backward pass
            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Scaler-aware optimizer step
                scaler.step(optimizer)
                scaler.update()

                optimizer.zero_grad()
                lr_scheduler.step()
                completed_steps += 1

                if is_main_process(args.single_gpu):
                    current_loss = loss.item()
                    progress_bar.update(1)
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{lr_scheduler.get_last_lr()[0]:.2e}")

                if completed_steps % args.logging_steps == 0 and is_main_process(args.single_gpu):
                    wandb.log({
                        "train/loss": current_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "trainer/global_step": completed_steps,
                        "epoch": epoch
                    })

                if completed_steps > 0 and completed_steps % args.checkpointing_steps == 0 and is_main_process(args.single_gpu):
                    save_checkpoint(model, tokenizer, args, completed_steps)

        # If we've reached the max steps, break out of the outer loop
        if completed_steps >= args.max_train_steps:
            break

    if is_main_process(args.single_gpu):
        progress_bar.close()

    # --- Final Model Saving ---
    if is_main_process(args.single_gpu):
        logger.info("Training complete. Saving final model.")
        final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
        
        model_to_save = model.module if hasattr(model, 'module') else model

        # Save locally
        model_to_save.save_pretrained(final_checkpoint_dir, safe_serialization=False)
        tokenizer.save_pretrained(final_checkpoint_dir)
        logger.info(f"Final model saved locally to {final_checkpoint_dir}")

        # Push to hub if requested
        if args.push_to_hub and args.hub_model_id:
            logger.info(f"Pushing final model to Hub repository: {args.hub_model_id}")
            try:
                model_to_save.push_to_hub(args.hub_model_id, private=args.hub_private_repo, safe_serialization=False)
                tokenizer.push_to_hub(args.hub_model_id, private=args.hub_private_repo)
                logger.info(f"Successfully pushed to {args.hub_model_id}")
            except Exception as e:
                logger.error(f"Failed to push final model to Hub: {e}")

    if not args.single_gpu:
        cleanup_distributed()

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
