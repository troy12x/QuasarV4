import os
import sys
import argparse
import shutil
import math
import time
from huggingface_hub import login
from huggingface_hub.utils import HfHubHTTPError
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
from contextlib import contextmanager

@contextmanager
def main_process_first(local_rank: int):
    """
    A context manager for torch.distributed. It ensures that the main process
    (rank 0) executes the code in the `with` block first. Other processes will
    wait at a barrier until the main process has finished.
    """
    if local_rank != -1 and dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()

    yield

    if local_rank != -1 and dist.is_initialized() and dist.get_rank() == 0:
        dist.barrier()
from datasets import load_dataset, Dataset, get_dataset_split_names
from itertools import chain

# Local imports
from quasar.lnn import LNNModel, LNNConfig
from huggingface_hub import create_repo

# Note: We will use a standard AdamW optimizer and the default data collator,
# so a custom utils file is no longer needed.
from torch.optim import AdamW


# --- Distributed Training Setup ---
def setup_distributed(timeout_seconds):
    """Initializes the distributed process group with a configurable timeout."""
    import datetime
    dist.init_process_group(
        backend="nccl", 
        timeout=datetime.timedelta(seconds=timeout_seconds)
    )
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def cleanup_distributed():
    """Cleans up the distributed process group."""
    dist.destroy_process_group()

def is_main_process(single_gpu_mode=False):
    """Checks if the current process is the main one (rank 0)."""
    if single_gpu_mode:
        return True
    if not dist.is_initialized():
        # Before dist.init_process_group is called, we can consider it the main process.
        # This is useful for initial setup steps like creating directories.
        return os.environ.get("LOCAL_RANK", "0") == "0"
    return dist.get_rank() == 0

# --- Logging Setup ---
logger = logging.getLogger(__name__)


# --- Checkpoint Saving ---
def save_checkpoint(model, optimizer, scheduler, tokenizer, args, global_step, epoch, tokens_processed):
    """Saves a full training state checkpoint for resumability."""
    if not is_main_process(args.single_gpu):
        return

    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Saving full training checkpoint to {checkpoint_dir}")

    # Unwrap the model if using DDP
    model_to_save = model.module if hasattr(model, 'module') else model

    # Save training state in a single file for atomicity
    training_state = {
        'global_step': global_step,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': args,
        'epoch': epoch,
        'tokens_processed': tokens_processed,
    }
    torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"))

    # Also save the model and tokenizer in the standard Hugging Face format
    # for easy interoperability.
    model_to_save.save_pretrained(checkpoint_dir, safe_serialization=False)
    tokenizer.save_pretrained(checkpoint_dir)

    logger.info(f"Full checkpoint saved successfully to {checkpoint_dir}")

    # Clean up old checkpoints to save space, keeping the last few
    if args.max_checkpoints_to_keep > 0:
        all_checkpoints = sorted(
            [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))],
            key=lambda x: int(x.split('-')[1])
        )
        if len(all_checkpoints) > args.max_checkpoints_to_keep:
            for ckpt_to_delete in all_checkpoints[:-args.max_checkpoints_to_keep]:
                shutil.rmtree(os.path.join(args.output_dir, ckpt_to_delete))
                logger.info(f"Deleted old checkpoint: {ckpt_to_delete}")

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
    parser = argparse.ArgumentParser(description="Fine-tuning script for LNN models.")
    
    # --- Model & Tokenizer Arguments ---
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pretrained LNN model to fine-tune (e.g., 'silx-ai/QuasarV4-LNN-Tiny').")
    parser.add_argument("--tokenizer_name", type=str, default=None, help="Tokenizer name or path. If None, it will be loaded from the model path.")

    # --- Data Arguments ---
    parser.add_argument("--dataset_name", type=str, default="HuggingFaceTB/smol-smoltalk", help="Dataset to use for fine-tuning.")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use.")
    parser.add_argument("--train_split_name", type=str, default="train", help="The name of the training data split to use.")
    parser.add_argument("--validation_split_name", type=str, default="validation", help="The name of the validation data split to use.")
    parser.add_argument("--validation_split_percentage", type=float, default=5.0, help="Percentage of training data to use for validation if validation split doesn't exist (e.g., 5.0 for 5%%).")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the column in the dataset containing the text.")
    parser.add_argument("--sequence_length", type=int, default=512, help="The sequence length for packing the dataset. Adjust based on your VRAM.")

    # --- Training & Output Arguments ---
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate for fine-tuning.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999], help="Beta values for AdamW optimizer.")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="Total number of training epochs to perform.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps. If provided, this overrides num_train_epochs.")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type (e.g., 'linear', 'cosine').")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="Ratio of total training steps for linear warmup. Is ignored if --num_warmup_steps is set.")
    parser.add_argument("--num_warmup_steps", type=int, default=None, help="Number of steps for linear warmup. Overrides warmup_ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended for H100).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    
    # Checkpointing & Logging
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--checkpointing_steps", type=int, default=500, help="Save checkpoint every N steps. More frequent for fine-tuning.")
    parser.add_argument("--max_checkpoints_to_keep", type=int, default=3, help="The maximum number of recent checkpoints to keep.")
    parser.add_argument("--eval_steps", type=int, default=500, help="Run evaluation every N steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--wandb_project", type=str, default="lnn-finetuning", help="Weights & Biases project name.")
    parser.add_argument("--single_gpu", action="store_true", help="Run on a single GPU without distributed training for testing.")
    parser.add_argument("--ddp_timeout", type=int, default=300, help="Timeout in seconds for DDP initialization.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached tokenized dataset if it exists.")
    parser.add_argument("--data_cache_dir", type=str, default="./cached_data", help="Directory to cache the processed dataset.")
    parser.add_argument("--num_proc", type=int, default=None, help="Number of processes for dataset processing. Defaults to number of CPUs.")

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push final model to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The model ID (repository name) on the Hugging Face Hub for the fine-tuned model.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Create a private repository on the Hub.")
    parser.add_argument("--debug_nan", action="store_true", help="Enable anomaly detection for debugging NaN loss.")

    args, _ = parser.parse_known_args()
    if args.num_proc is None:
        args.num_proc = os.cpu_count()
        
    return args

# --- Evaluation Function ---
def evaluate(model, dataloader, device, autocast_dtype, args):
    """
    Runs evaluation on the validation set and returns the average loss.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    num_batches = 0

    # Only show progress bar on the main process
    iterable = dataloader
    if is_main_process(args.single_gpu):
        iterable = tqdm(dataloader, desc="Evaluating", leave=False)

    with torch.no_grad():
        for batch in iterable:
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            
            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss
            
            # Gather loss across all GPUs if in distributed mode
            if dist.is_initialized():
                # Reduce loss from all processes
                dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                loss /= dist.get_world_size()

            total_loss += loss.item()
            num_batches += 1
            
    model.train()  # Set the model back to training mode
    if num_batches == 0:
        return 0.0
    return total_loss / num_batches


# --- Main Training Logic ---
def main():
    args = parse_args()

    # --- Setup ---
    if args.single_gpu:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
        setup_logging(single_gpu_mode=True)
    else:
        local_rank = setup_distributed(args.ddp_timeout)
        device = torch.device(f"cuda:{local_rank}")
        world_size = dist.get_world_size()
        setup_logging()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    if is_main_process(args.single_gpu):
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, config=args)
    
    tokenizer_load_path = args.tokenizer_name if args.tokenizer_name else args.model_name_or_path
    logger.info(f"Loading tokenizer from: {tokenizer_load_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_load_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset Loading & Caching ---
    # Use main_process_first to ensure only one process downloads/processes the dataset.
    with main_process_first(local_rank=local_rank if not args.single_gpu else -1):
        # Determine if we need to create a validation split by checking the dataset's available splits.
        try:
            split_names = get_dataset_split_names(args.dataset_name, config_name=args.dataset_config_name)
            needs_split = args.validation_split_name not in split_names
        except Exception:
            logger.warning(f"Could not determine dataset splits from Hub. Will attempt to load and then split if needed.")
            needs_split = True  # Assume split is needed if we can't check, will verify after loading.

        # Create a unique path for the cached dataset.
        dataset_identifier = args.dataset_name.replace("/", "_")
        if args.dataset_config_name:
            dataset_identifier += f"_{args.dataset_config_name}"
        tokenizer_identifier = tokenizer_load_path.replace("/", "_")
        
        cache_name_base = f"{dataset_identifier}_{tokenizer_identifier}_seqlen{args.sequence_length}"
        if needs_split:
            cache_name_base += f"_val_split_{args.validation_split_percentage}"
        
        cached_dataset_path = os.path.join(args.data_cache_dir, cache_name_base)

        logger.info(f"Attempting to load cached dataset from: {cached_dataset_path}")

        # Check for a specific file to ensure the cache is valid, not just if the directory exists.
        cache_is_valid = not args.overwrite_cache and os.path.exists(os.path.join(cached_dataset_path, "dataset_info.json"))

        if cache_is_valid:
            logger.info(f"Loading processed dataset from cache: {cached_dataset_path}")
            train_dataset = Dataset.load_from_disk(cached_dataset_path)
        else:
            if os.path.exists(cached_dataset_path) and is_main_process(args.single_gpu):
                logger.warning(f"Found an incomplete or outdated cache at {cached_dataset_path}. Deleting and reprocessing.")
                shutil.rmtree(cached_dataset_path)
                eval_cache_path = cached_dataset_path + "_eval"
                if os.path.exists(eval_cache_path):
                    shutil.rmtree(eval_cache_path)

            logger.info("Cache not found or invalid. Processing dataset from scratch.")
            raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name, streaming=False)

            # --- ShareGPT Data Processing ---
            # This section is modified to handle conversational formats like ShareGPT.
            # It formats each conversation using the tokenizer's chat template,
            # then tokenizes it and masks the user's prompts in the labels.
            def process_conversations(examples):
                # The 'conversations' column in smol-smoltalk holds the list of turns.
                # We will format this into a single sequence for the model.
                outputs = {'input_ids': [], 'labels': [], 'attention_mask': []}

                for conversation in examples["conversations"]:
                    # Apply the chat template to the entire conversation.
                    # This adds the special tokens (e.g., for user, assistant) needed by the model.
                    full_tokenized = tokenizer.apply_chat_template(
                        conversation, 
                        truncation=True, 
                        max_length=args.sequence_length, 
                        padding=False, # Data collator will handle padding
                        return_tensors=None # Return list of ints
                    )
                    
                    if not full_tokenized:
                        continue # Skip empty or invalid conversations

                    # Create labels by cloning input_ids. We will then mask out the prompt sections.
                    labels = list(full_tokenized)
                    
                    # Find all instances of the assistant's turn to mask the prompts correctly.
                    # We find the start of each assistant message and mask everything before it.
                    # This ensures we only calculate loss on the assistant's responses.
                    assistant_turn_starts = []
                    for i in range(len(conversation) - 1):
                        if conversation[i]['role'] == 'user' and conversation[i+1]['role'] == 'assistant':
                            # Get the template up to the point *before* the assistant speaks
                            prompt_template = tokenizer.apply_chat_template(
                                conversation[:i+1], 
                                add_generation_prompt=True, # This adds the prompt for the assistant to start
                                tokenize=False
                            )
                            # Find where this prompt ends in the tokenized output
                            prompt_end_index = len(tokenizer.encode(prompt_template, add_special_tokens=False))
                            assistant_turn_starts.append(prompt_end_index)
                    
                    # Mask all tokens that are part of the prompt (i.e., not the assistant's response)
                    # We iterate backwards to handle multiple turns in a single conversation correctly.
                    # Start with a fully unmasked sequence of labels
                    is_response_part = [True] * len(labels)
                    
                    # For each turn, we mask everything *up to* the assistant's part
                    current_mask_end = len(labels)
                    for start_index in reversed(assistant_turn_starts):
                        if start_index < current_mask_end:
                            for i in range(start_index):
                                is_response_part[i] = False # Mask prompt tokens
                        current_mask_end = start_index

                    # Apply the mask to the labels. Unmasked tokens are model's responses.
                    for i in range(len(labels)):
                        if not is_response_part[i]:
                            labels[i] = -100

                    # A final check: if no part of the conversation is a response, skip it.
                    if all(label == -100 for label in labels):
                        continue
                            
                    outputs['input_ids'].append(full_tokenized)
                    outputs['labels'].append(labels)
                    outputs['attention_mask'].append([1] * len(full_tokenized))

                return outputs

            logger.info("Processing conversations with ShareGPT format...")
            
            # Ensure the correct column name is used
            column_names = list(raw_datasets.values())[0].column_names
            dataset_column = "message" if "messages" in column_names else args.text_column
            
            train_raw_dataset = raw_datasets.get(args.train_split_name)
            eval_raw_dataset = raw_datasets.get(args.validation_split_name)

            if train_raw_dataset is None:
                raise ValueError(f"Train split '{args.train_split_name}' not found. Available: {list(raw_datasets.keys())}")

            # If validation set doesn't exist, create it from the training set
            if eval_raw_dataset is None:
                logger.warning(f"Validation split '{args.validation_split_name}' not found. Creating a {args.validation_split_percentage}% split from training data.")
                
                split_percentage = args.validation_split_percentage / 100.0
                if not (0 < split_percentage < 1):
                    raise ValueError("validation_split_percentage must be between 0 and 100.")
                
                split_dataset = train_raw_dataset.train_test_split(test_size=split_percentage, shuffle=True, seed=args.seed)
                train_raw_dataset = split_dataset['train']
                eval_raw_dataset = split_dataset['test']
                logger.info(f"Created validation split with {len(eval_raw_dataset)} samples. New training set size: {len(train_raw_dataset)}.")

            logger.info("Applying chat template to training data...")
            train_dataset = train_raw_dataset.map(
                process_conversations,
                batched=True,
                num_proc=args.num_proc,
                remove_columns=train_raw_dataset.column_names,
                desc="Formatting training data"
            )
            
            if is_main_process(args.single_gpu):
                logger.info(f"Saving processed train dataset to cache: {cached_dataset_path}")
                train_dataset.save_to_disk(cached_dataset_path)

            # Process and cache validation dataset
            if eval_raw_dataset:
                cached_eval_dataset_path = cached_dataset_path + "_eval"
                logger.info("Applying chat template to validation data...")
                eval_dataset = eval_raw_dataset.map(
                    process_conversations,
                    batched=True,
                    num_proc=args.num_proc,
                    remove_columns=eval_raw_dataset.column_names,
                    desc="Formatting validation data"
                )
                if is_main_process(args.single_gpu):
                    logger.info(f"Saving processed validation dataset to cache: {cached_eval_dataset_path}")
                    eval_dataset.save_to_disk(cached_eval_dataset_path)

    train_dataset.set_format("torch")
    
    # Load validation dataset if it was processed
    eval_dataset = None
    cached_eval_dataset_path = cached_dataset_path + "_eval"
    if os.path.exists(cached_eval_dataset_path):
         logger.info(f"Loading processed validation dataset from cache: {cached_eval_dataset_path}")
         eval_dataset = Dataset.load_from_disk(cached_eval_dataset_path)
         eval_dataset.set_format("torch")
    
    train_sampler = RandomSampler(train_dataset) if args.single_gpu else DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.per_device_train_batch_size, sampler=train_sampler, collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), pin_memory=True, drop_last=True
    )

    eval_dataloader = None
    if eval_dataset:
        eval_sampler = SequentialSampler(eval_dataset) if args.single_gpu else DistributedSampler(eval_dataset, num_replicas=world_size, rank=local_rank, shuffle=False)
        eval_dataloader = DataLoader(
            eval_dataset, batch_size=args.per_device_eval_batch_size, sampler=eval_sampler, collate_fn=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False), pin_memory=True, drop_last=True
        )

    logger.info(f"Initializing model from: {args.model_name_or_path}")
    model = LNNModel.from_pretrained(args.model_name_or_path)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    model = model.to(device)
    if not args.single_gpu:
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
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Calculate warmup steps.
    if args.num_warmup_steps is None:
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
        logger.info(f"Calculated warmup steps from ratio: {args.num_warmup_steps}")
    else:
        logger.info(f"Using specified number of warmup steps: {args.num_warmup_steps}")

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps
    )

    # --- Resume from Checkpoint ---
    global_step = 0
    start_epoch = 0
    tokens_processed = 0

    if args.resume_from_checkpoint:
        checkpoint_dir = args.resume_from_checkpoint
        logger.info(f"Attempting to resume from checkpoint: {checkpoint_dir}")

        training_state_path = os.path.join(checkpoint_dir, "training_state.pt")
        if os.path.exists(training_state_path):
            logger.info(f"Found full training state at {training_state_path}. Loading...")
            checkpoint = torch.load(training_state_path, map_location=device, weights_only=False)
            
            model_to_load = model.module if not args.single_gpu else model
            model_to_load.load_state_dict(checkpoint['model_state_dict'], strict=True)
            
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint.get('epoch', 0)
            global_step = checkpoint.get('global_step', 0)
            tokens_processed = checkpoint.get('tokens_processed', 0)
            logger.info(f"Successfully resumed from epoch {start_epoch}, global step {global_step}. LR is {lr_scheduler.get_last_lr()[0]:.2e}")
        else:
            logger.error(f"FATAL: Could not find 'training_state.pt' in {checkpoint_dir}. Cannot resume training.")
            sys.exit(1) # Exit because we cannot fulfill the resume request.

    logger.info("***** Starting Fine-Tuning *****")
    logger.info(f"  Total train batch size = {args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    autocast_dtype = torch.bfloat16 if args.bf16 else torch.float32
    scaler = torch.cuda.amp.GradScaler(enabled=args.bf16)
    
    completed_steps = global_step
    model.train()

    if is_main_process(args.single_gpu):
        progress_bar = tqdm(initial=completed_steps, total=args.max_train_steps, desc="Fine-Tuning Progress")

    for epoch in range(start_epoch, args.num_train_epochs):
        if not args.single_gpu:
            train_sampler.set_epoch(epoch)

        for step, batch in enumerate(train_dataloader):
            if completed_steps >= args.max_train_steps:
                break

            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            batch_tokens = batch['input_ids'].numel() * world_size
            tokens_processed += batch_tokens

            with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                outputs = model(**batch, labels=labels)
                loss = outputs.loss

            scaler.scale(loss / args.gradient_accumulation_steps).backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                scaler.unscale_(optimizer)
                
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

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
                    log_stats = {
                        "train/loss": current_loss,
                        "train/learning_rate": lr_scheduler.get_last_lr()[0],
                        "tokens_processed": tokens_processed,
                        "epoch": epoch,
                    }
                    wandb.log(log_stats, step=completed_steps)

                if completed_steps > 0 and completed_steps % args.checkpointing_steps == 0 and is_main_process(args.single_gpu):
                    save_checkpoint(model, optimizer, lr_scheduler, tokenizer, args, completed_steps, epoch, tokens_processed)

                if eval_dataloader and completed_steps > 0 and completed_steps % args.eval_steps == 0:
                    if is_main_process(args.single_gpu):
                        logger.info(f"--- Starting evaluation at step {completed_steps} ---")
                    
                    eval_loss = evaluate(model, eval_dataloader, device, autocast_dtype, args)
                    
                    if is_main_process(args.single_gpu):
                        logger.info(f"--- Evaluation finished. Eval loss: {eval_loss:.4f} ---")
                        wandb.log({"eval/loss": eval_loss}, step=completed_steps)

        if completed_steps >= args.max_train_steps:
            break

    if is_main_process(args.single_gpu):
        progress_bar.close()

    # --- Final Model Saving ---
    if is_main_process(args.single_gpu):
        logger.info("Fine-tuning complete. Saving final model.")
        final_checkpoint_dir = os.path.join(args.output_dir, "final_checkpoint")
        
        model_to_save = model.module if hasattr(model, 'module') else model

        model_to_save.save_pretrained(final_checkpoint_dir, safe_serialization=False)
        tokenizer.save_pretrained(final_checkpoint_dir)
        logger.info(f"Final model saved locally to {final_checkpoint_dir}")

        if args.push_to_hub and args.hub_model_id:
            logger.info(f"Pushing final model to Hub repository: {args.hub_model_id}")
            try:
                # Create the repo if it doesn't exist
                create_repo(args.hub_model_id, private=args.hub_private_repo, exist_ok=True)
                # Push the model and tokenizer
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