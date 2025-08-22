import os
import sys
import argparse
import shutil
import math
import time
import warnings

from huggingface_hub import login, snapshot_download
from huggingface_hub.utils import HfHubHTTPError
import logging
import time
import threading
import torch

# Add project root and scripts directory to sys.path to allow for local package imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
# The cycling_utils library is in the scripts directory, so we add it to the path
sys.path.insert(0, os.path.join(project_root, 'scripts','cycling_utils'))

# Suppress the torch pytree deprecation warning from transformers
warnings.filterwarnings("ignore", message="`torch.utils._pytree._register_pytree_node` is deprecated. Please use `torch.utils._pytree.register_pytree_node` instead.", category=FutureWarning)

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
from datasets import load_dataset, Dataset, get_dataset_split_names, load_from_disk
from itertools import chain

import deepspeed

# Local imports
from quasar.lnn import LNNModel, LNNConfig
from huggingface_hub import create_repo

# Note: We will use a standard AdamW optimizer and the default data collator,
# so a custom utils file is no longer needed.
from torch.optim import AdamW
import torch.nn.functional as F
from deepspeed.ops.adam import DeepSpeedCPUAdam


# --- Distributed Training Setup ---
def setup_distributed(timeout_seconds):
    """Initializes the distributed process group and sets the device for the current process."""
    import datetime

    # Get local rank from environment variable set by torchrun.
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set the device for the current process. This MUST be done before initializing the process group.
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # It's recommended to have MASTER_ADDR and MASTER_PORT set in the environment
    if not all(k in os.environ for k in ("MASTER_ADDR", "MASTER_PORT")):
        logger.warning("MASTER_ADDR or MASTER_PORT not set, using default localhost:12355")
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "12355")

    # Initialize the process group. The device is implicitly set by torch.cuda.set_device().
    dist.init_process_group(
        backend="nccl",
        timeout=datetime.timedelta(seconds=timeout_seconds)
    )

    # Barrier to synchronize all processes.
    dist.barrier()
    logger.info(f"Rank {dist.get_rank()} initialized on device {device} successfully.")
    return local_rank, device

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



def setup_logging(single_gpu_mode=False):
    """Sets up logging, restricting verbose logs to the main process."""
    log_level = logging.INFO if is_main_process(single_gpu_mode) else logging.WARNING
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
    )


# --- Checkpoint Saving ---
def save_checkpoint(model, optimizer, scheduler, tokenizer, args, global_step, epoch, tokens_processed):
    """Saves a full training state checkpoint, handling DeepSpeed and DDP cases correctly."""
    checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")

    if args.deepspeed:
        # This is a collective call that all ranks must execute.
        # DeepSpeed internally handles gathering the sharded states.
        logger.info(f"Saving DeepSpeed checkpoint to {checkpoint_dir}")
        model.save_checkpoint(checkpoint_dir, client_state={'epoch': epoch, 'global_step': global_step, 'tokens_processed': tokens_processed})
    
    # The following operations should only be done by the main process.
    if is_main_process(args.single_gpu):
        logger.info(f"Performing main process save operations for checkpoint {checkpoint_dir}")
        os.makedirs(checkpoint_dir, exist_ok=True)

        if not args.deepspeed:
            # For standard DDP, we save the model, tokenizer, and state dicts manually.
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(checkpoint_dir)
            training_state = {
                'epoch': epoch,
                'global_step': global_step,
                'tokens_processed': tokens_processed,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }
            torch.save(training_state, os.path.join(checkpoint_dir, "training_state.pt"), _use_new_zipfile_serialization=True)

        # Save tokenizer and args in both DDP and DeepSpeed cases.
        tokenizer.save_pretrained(checkpoint_dir)
        torch.save(args, os.path.join(checkpoint_dir, "training_args.bin"))
        logger.info(f"Full checkpoint saved successfully to {checkpoint_dir}")

        # Clean up old checkpoints to save space.
        if args.max_checkpoints_to_keep > 0:
            all_checkpoints = sorted(
                [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.output_dir, d))],
                key=lambda x: int(x.split('-')[1])
            )
            if len(all_checkpoints) > args.max_checkpoints_to_keep:
                for ckpt_to_delete in all_checkpoints[:-args.max_checkpoints_to_keep]:
                    shutil.rmtree(os.path.join(args.output_dir, ckpt_to_delete))
                    logger.info(f"Deleted old checkpoint: {ckpt_to_delete}")

    # Crucial barrier to prevent race conditions.
    # Ensures no process moves on until the checkpoint is fully saved and cleaned up.
    if dist.is_initialized():
        dist.barrier()

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Large-scale pre-training script for LNN")
    
    # --- Model & Tokenizer Arguments ---
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Load a pretrained LNN model. If None, create a new model from scratch.")
    parser.add_argument("--tokenizer_name", type=str, default="deepseek-ai/DeepSeek-V3", help="Tokenizer name or path.")

    # --- Arguments for Creating a New Model ---
    parser.add_argument("--hidden_size", type=int, default=3048, help="Hidden size for new models.")
    parser.add_argument("--num_hidden_layers", type=int, default=6, help="Number of hidden layers for new models.")
    parser.add_argument("--dt", type=float, default=0.1, help="Integration step size (dt) for the LNN cells.")
    parser.add_argument("--use_moe", action="store_true", help="Enable Mixture of Experts layers when creating a new model.")
    parser.add_argument("--num_experts", type=int, default=14, help="Number of experts for MoE layers.")
    parser.add_argument("--num_experts_per_tok", type=int, default=2, help="Number of experts to use per token for MoE layers.")

    # --- Data Arguments ---
    parser.add_argument("--dataset_name", type=str, default=None, required=False, help="The name of the dataset to use from the Hugging Face Hub (not required if mixing datasets).")
    parser.add_argument("--dataset_config_name", type=str, default=None, help="The configuration name of the dataset to use.")
    parser.add_argument("--train_split_name", type=str, default="train", help="The name of the training data split to use.")
    parser.add_argument("--validation_split_name", type=str, default="validation", help="The name of the validation data split to use.")
    parser.add_argument("--validation_split_percentage", type=int, default=5, help="The percentage of the train set used as validation set in case there's no validation split")
    parser.add_argument("--mix_reasoning_and_language_data", action="store_true", help="Mix reasoning and language datasets.")
    parser.add_argument("--language_to_reasoning_ratio", type=int, nargs=2, default=None, help="The ratio of language to reasoning data (e.g., 70 30 for 70%% language, 30%% reasoning).")
    parser.add_argument("--save_tokenized_data", action="store_true", help="Save the tokenized dataset to disk.")
    parser.add_argument("--tokenized_data_path", type=str, default=None, help="Path to save the tokenized dataset.")
    parser.add_argument("--text_column", type=str, default="text", help="The name of the column in the dataset containing the text.")
    parser.add_argument("--sequence_length", type=int, default=2048, help="The sequence length for packing the dataset.")

    # --- Training & Output Arguments ---
    parser.add_argument("--output_dir", type=str, default="./output", help="Directory to save checkpoints and final model.")
    parser.add_argument("--resume-from-checkpoint", type=str, default=None, help="Path to a specific checkpoint directory to resume training from.")
    parser.add_argument("--deepspeed_checkpoint", type=str, default=None, help="Path to a DeepSpeed checkpoint to resume training from.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8, help="Batch size per GPU for training.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Eval batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps.")
    # Note: A cosine scheduler is often more effective for large model pre-training.
    # It helps in achieving a lower loss by annealing the learning rate smoothly.
    # The learning rate and Adam betas have been adjusted for more stable training to prevent NaN loss.
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate. Lowered for stability.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=[0.9, 0.999], help="Beta values for AdamW optimizer. Beta2 increased for stability.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs to perform. For large datasets, --max_train_steps is recommended.")
    parser.add_argument("--max_train_steps", type=int, default=None, help="Total number of training steps. If provided, this overrides num_train_epochs.")
    # Using a cosine scheduler with a longer warmup is a common best practice for pre-training.
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine", help="LR scheduler type (e.g., 'linear', 'cosine').")
    parser.add_argument("--warmup_ratio", type=float, default=0.02, help="Ratio of total training steps for linear warmup. Is ignored if --num_warmup_steps is set.")
    parser.add_argument("--num_warmup_steps", type=int, default=None, help="Number of steps for linear warmup. Overrides warmup_ratio.")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Gradient clipping max norm.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--bf16", action="store_true", help="Use BF16 mixed precision (recommended for H100).")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 mixed precision (recommended for older GPUs).")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing to save memory.")
    
    # Checkpointing & Logging
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--checkpointing_steps", type=int, default=1000, help="Save checkpoint every N steps.")
    parser.add_argument("--max_checkpoints_to_keep", type=int, default=3, help="The maximum number of recent checkpoints to keep.")
    parser.add_argument("--eval_steps", type=int, default=10000, help="Run evaluation every N steps.")
    parser.add_argument("--max_eval_steps", type=int, default=None, help="Maximum number of evaluation steps to run. If None, runs on the entire validation set.")
    parser.add_argument("--logging_steps", type=int, default=1, help="Log every N steps.")
    parser.add_argument("--wandb_project", type=str, default="lnn-pretraining", help="Weights & Biases project name.")
    parser.add_argument("--single_gpu", action="store_true", help="Run on a single GPU without distributed training for testing.")
    parser.add_argument("--ddp_timeout", type=int, default=21600, help="Timeout for DDP initialization (in seconds). Increased default to 6 hours for slow checkpointing.")

    # --- DeepSpeed Arguments ---
    parser.add_argument("--deepspeed", action="store_true", help="Enable DeepSpeed.")
    parser.add_argument("--deepspeed_config", type=str, default="ds_config.json", help="Path to DeepSpeed config file.")
    parser.add_argument("--overwrite_cache", action="store_true", help="Overwrite the cached tokenized dataset if it exists.")
    parser.add_argument("--data_cache_dir", type=str, default="./cached_data", help="Directory to cache the processed dataset.")
    parser.add_argument("--num_proc", type=int, default=32, help="Number of processes for dataset processing. Reduce if you encounter out-of-memory errors.")

    # Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", help="Push checkpoints to the Hugging Face Hub.")
    parser.add_argument("--hub_model_id", type=str, default=None, help="The model ID (repository name) on the Hugging Face Hub.")
    parser.add_argument("--hub_private_repo", action="store_true", help="Create a private repository on the Hub.")
    parser.add_argument("--debug_nan", action="store_true", help="Enable anomaly detection for debugging NaN loss.")

    # New argument for checkpoint conversion
    parser.add_argument("--convert_checkpoint", action="store_true", help="Convert a DeepSpeed checkpoint to a standard FP32 model and exit.")

    # New argument for skipping tokens
    parser.add_argument(
        "--skip-tokens",
        type=int,
        default=0,
        help="Skip the first N tokens of the dataset before starting training. This is useful for resuming a run from a specific point without a checkpoint."
    )

    args, _ = parser.parse_known_args()
    
    # Validate precision arguments
    if args.bf16 and args.fp16:
        raise ValueError("Cannot use both --bf16 and --fp16. Please choose one.")
    
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
        for i, batch in enumerate(iterable):
            # If a max number of steps is set, break the loop
            if args.max_eval_steps is not None and i >= args.max_eval_steps:
                break
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            
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
    logger.info(f"Parsed Arguments: {args}")

    # --- Argument Validation ---
    if not args.dataset_name and not args.mix_reasoning_and_language_data:
        print("ERROR: You must specify either --dataset_name or use --mix_reasoning_and_language_data", file=sys.stderr)
        sys.exit(1)

    if args.dataset_name and args.mix_reasoning_and_language_data:
        print("Warning: --dataset_name is ignored when --mix_reasoning_and_language_data is used.", file=sys.stderr)
        args.dataset_name = None  # Clear it to prevent confusion

    # --- Checkpoint Conversion Mode ---
    if args.convert_checkpoint:
        if not args.resume_from_checkpoint or not args.output_dir:
            print("ERROR: --convert_checkpoint requires --resume_from_checkpoint (path to DS checkpoint) and --output_dir (where to save FP32 model).")
            sys.exit(1)

        print(f"--- Starting Checkpoint Conversion ---")
        # We don't need a full distributed setup, just a device to work on.
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 1. Load the model CONFIGURATION ONLY from the base model id.
        print(f"Loading configuration from: {args.model_name_or_path}")
        config = LNNConfig.from_pretrained(args.model_name_or_path)

        # 2. Build the model architecture (with random weights)
        print("Initializing model from configuration...")
        model = LNNModel(config)

        # 3. Use DeepSpeed to load the checkpoint into the model structure.
        # This is the most reliable way as it uses the same engine that saved it.
        # We need a dummy optimizer and a minimal config for initialization.
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        ds_config = {"train_batch_size": 1}

        model, _, _, _ = deepspeed.initialize(model=model, optimizer=optimizer, config=ds_config)

        print(f"Loading DeepSpeed checkpoint from: {args.resume_from_checkpoint}")
        # load_checkpoint will consolidate the weights from ZeRO shards.
        load_path, _ = model.load_checkpoint(args.resume_from_checkpoint, load_optimizer_states=False, load_lr_scheduler_states=False)
        if load_path is None:
            print(f"FATAL: Failed to load DeepSpeed checkpoint from {args.resume_from_checkpoint}")
            sys.exit(1)

        # 4. Save the consolidated FP32 model.
        fp32_output_dir = os.path.join(args.output_dir, "converted_fp32_model")
        print(f"Saving converted FP32 model to: {fp32_output_dir}")

        # The model object now holds the full, consolidated weights.
        # We may need to access the underlying module.
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(fp32_output_dir)

        # Also save the tokenizer for completeness.
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        tokenizer.save_pretrained(fp32_output_dir)

        print("--- Conversion Complete ---")
        sys.exit(0)


    # Disable wandb in all processes except the main one to prevent NFS errors.
    # This check is done before initializing torch.distributed. We assume the main process
    # has either RANK or LOCAL_RANK set to 0. This is a safe assumption for most launchers.
    is_main = os.environ.get("RANK", "-1") == "0" or os.environ.get("LOCAL_RANK", "-1") == "0"
    if not is_main:
        os.environ["WANDB_DISABLED"] = "true"
        # We can't use the logger yet as it's not configured, so we print to stderr.
        print(f"Process with RANK={os.environ.get('RANK', 'N/A')}, LOCAL_RANK={os.environ.get('LOCAL_RANK', 'N/A')} is not the main process. Disabling wandb.", file=sys.stderr)

    # --- Setup ---
    if args.single_gpu:
        local_rank = 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        world_size = 1
        setup_logging(single_gpu_mode=True)
    else:
        # setup_distributed now handles all device setup and returns the configured rank and device.
        local_rank, device = setup_distributed(args.ddp_timeout)
        world_size = dist.get_world_size()
        setup_logging()

    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    
    # Initialize the AtomicDirectory saver. This must be done on all ranks.
    # We will use it to manage checkpoint directories atomically.
    # The keep_last argument is set from the script's arguments.


    if is_main_process(args.single_gpu):
        os.makedirs(args.output_dir, exist_ok=True)
        wandb.init(project=args.wandb_project, config=args)

    logger.info(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Dataset Loading & Caching ---
    # Use main_process_first to ensure only one process downloads/processes the dataset.
    with main_process_first(local_rank=local_rank if not args.single_gpu else -1):
        if args.mix_reasoning_and_language_data:
            logger.info("Mixing reasoning and language datasets.")
            try:
                # For mixed data, we ignore streaming for the initial load to allow concatenation.
                from datasets import concatenate_datasets, load_dataset

                logger.info("Loading datasets for merging (this might take a while and consume memory)...")

                # Load language data (Ultra-FineWeb)
                lang_dataset = load_dataset("sumuks/Ultra-FineWeb-1B", split='train', cache_dir=None)
                lang_dataset = lang_dataset.rename_column("content", "text")
                lang_dataset = lang_dataset.remove_columns([col for col in lang_dataset.column_names if col != 'text'])

                # Load reasoning data (MegaMath)
                reasoning_dataset = load_dataset("semran1/megamath-web-pro", split='train', cache_dir=None)
                reasoning_dataset = reasoning_dataset.remove_columns([col for col in reasoning_dataset.column_names if col != 'text'])

                # Shuffle datasets for random sampling
                lang_dataset = lang_dataset.shuffle(seed=args.seed)
                reasoning_dataset = reasoning_dataset.shuffle(seed=args.seed)

                if args.language_to_reasoning_ratio:
                    lang_ratio, reas_ratio = args.language_to_reasoning_ratio
                    logger.info(f"Applying language-to-reasoning ratio: {lang_ratio}:{reas_ratio}")

                    n_lang = len(lang_dataset)
                    n_reas = len(reasoning_dataset)

                    # Determine the number of samples to take from each dataset to match the ratio
                    # We want to find k such that k*lang_ratio <= n_lang and k*reas_ratio <= n_reas
                    # k = min(n_lang/lang_ratio, n_reas/reas_ratio)
                    k = min(n_lang / lang_ratio, n_reas / reas_ratio)
                    
                    n_lang_samples = int(k * lang_ratio)
                    n_reas_samples = int(k * reas_ratio)

                    logger.info(f"Sampling {n_lang_samples} from language dataset and {n_reas_samples} from reasoning dataset.")

                    lang_subset = lang_dataset.select(range(n_lang_samples))
                    reas_subset = reasoning_dataset.select(range(n_reas_samples))
                    
                    merged_dataset = concatenate_datasets([lang_subset, reas_subset])
                else:
                    logger.info("No ratio specified. Merging all loaded datasets.")
                    merged_dataset = concatenate_datasets([lang_dataset, reasoning_dataset])

                # Create a train/validation split from the merged dataset
                if args.validation_split_percentage > 0:
                    split_dataset = merged_dataset.train_test_split(test_size=(args.validation_split_percentage / 100.0), seed=args.seed)
                    raw_train_dataset = split_dataset["train"]
                    eval_raw_dataset = split_dataset["test"]
                    logger.info(f"Created a validation split of {len(eval_raw_dataset)} samples.")
                else:
                    raw_train_dataset = merged_dataset
                    eval_raw_dataset = None
                    logger.warning("No validation split created for the mixed dataset.")

            except Exception as e:
                import traceback
                logger.error(f"Failed to load or merge datasets: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                if not args.single_gpu:
                    cleanup_distributed()
                sys.exit(1)
        else:
            # Original dataset loading logic
            try:
                split_names = get_dataset_split_names(args.dataset_name, config_name=args.dataset_config_name)
                needs_split = args.validation_split_name not in split_names
            except Exception:
                logger.warning(f"Could not determine dataset splits from Hub. Will attempt to load and then split if needed.")
                needs_split = True

        # Determine the path for the tokenized dataset.
        # Priority is given to the user-provided path.
        if args.tokenized_data_path:
            cache_path = args.tokenized_data_path
            logger.info(f"Using user-provided path for tokenized data: {cache_path}")
        else:
            # Create a unique path for the cached dataset if no path is provided.
            if args.mix_reasoning_and_language_data:
                cache_desc = "_mixed_reasoning_language"
            else:
                cache_desc = f"_{args.dataset_name.replace('/', '_')}"
                try:
                    split_names = get_dataset_split_names(args.dataset_name, config_name=args.dataset_config_name)
                    if args.validation_split_name not in split_names:
                        cache_desc += f"_split_{args.validation_split_percentage}"
                except Exception:
                    cache_desc += f"_split_{args.validation_split_percentage}"
            cache_path = os.path.join(args.data_cache_dir, f"tokenized_{cache_desc}_{args.sequence_length}")

        # Load from cache if it exists and overwrite is not requested
        if not args.overwrite_cache and os.path.exists(cache_path):
            logger.info(f"Loading tokenized dataset from cache: {cache_path}")
            train_dataset = load_from_disk(cache_path)
        else:
            logger.info("Generating tokenized dataset...")
            # If the directory exists but is invalid, clean it up before reprocessing.
            if not args.overwrite_cache and os.path.exists(cache_path):
                logger.warning(f"Found an incomplete cache at {cache_path}. Deleting and reprocessing.")
                if is_main_process(args.single_gpu):
                    shutil.rmtree(cache_path)
                    # Also remove the corresponding eval cache if it exists
                    eval_cache_path = cache_path + "_eval"
                    if os.path.exists(eval_cache_path):
                        shutil.rmtree(eval_cache_path)

            logger.info("Cache not found or invalid. Processing dataset from scratch.")
            if args.mix_reasoning_and_language_data:
                # Use the datasets directly from the mixing block
                pass  # raw_train_dataset and raw_eval_dataset are already set
            else:
                raw_datasets_hf = load_dataset(args.dataset_name, args.dataset_config_name, streaming=False)
                raw_train_dataset = raw_datasets_hf.get(args.train_split_name)
                raw_eval_dataset = raw_datasets_hf.get(args.validation_split_name)

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

            if args.text_column != 'text':
                train_raw_dataset = train_raw_dataset.rename_column(args.text_column, 'text')
                if eval_raw_dataset:
                    eval_raw_dataset = eval_raw_dataset.rename_column(args.text_column, 'text')

            def tokenize_function(examples):
                return tokenizer(examples['text'], truncation=False, padding=False)

            logger.info(f"Tokenizing training data ({len(train_raw_dataset)} samples) with {args.num_proc} processes...")
            tokenized_dataset = train_raw_dataset.map(
                tokenize_function, batched=True, num_proc=args.num_proc, remove_columns=train_raw_dataset.column_names
            )
            logger.info("Tokenization of training data complete.")

            def group_texts(examples):
                concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
                total_length = len(concatenated_examples[list(examples.keys())[0]])
                if total_length >= args.sequence_length:
                    total_length = (total_length // args.sequence_length) * args.sequence_length
                result = {
                    k: [t[i : i + args.sequence_length] for i in range(0, total_length, args.sequence_length)]
                    for k, t in concatenated_examples.items()
                }
                result["labels"] = result["input_ids"].copy()
                return result

            logger.info(f"Grouping texts for training set ({len(tokenized_dataset)} samples) with {args.num_proc} processes...")
            train_dataset = tokenized_dataset.map(
                group_texts, batched=True, num_proc=args.num_proc
            )
            logger.info("Grouping of training texts complete.")
            
            if is_main_process(args.single_gpu):
                logger.info(f"Saving tokenized dataset to: {cache_path}")
                train_dataset.save_to_disk(cache_path)

            # Process and cache validation dataset
            if eval_raw_dataset:
                cached_eval_dataset_path = cache_path + "_eval"
                logger.info(f"Tokenizing validation data ({len(eval_raw_dataset)} samples) with {args.num_proc} processes...")
                eval_tokenized_dataset = eval_raw_dataset.map(
                    tokenize_function, batched=True, num_proc=args.num_proc, remove_columns=eval_raw_dataset.column_names
                )
                logger.info("Tokenization of validation data complete.")

                logger.info(f"Grouping texts for validation set ({len(eval_tokenized_dataset)} samples) with {args.num_proc} processes...")
                eval_dataset = eval_tokenized_dataset.map(
                    group_texts, batched=True, num_proc=args.num_proc
                )
                logger.info("Grouping of validation texts complete.")
                if is_main_process(args.single_gpu):
                    logger.info(f"Saving processed validation dataset to cache: {cached_eval_dataset_path}")
                    eval_dataset.save_to_disk(cached_eval_dataset_path)

    train_dataset.set_format("torch")

    # --- Optionally skip tokens ---
    # This logic is applied only when not resuming from a checkpoint, as the checkpoint
    # already contains the training progress.
    if args.skip_tokens > 0 and not args.resume_from_checkpoint:
        if is_main_process(args.single_gpu):
            logger.info(f"--skip-tokens specified. Attempting to skip {args.skip_tokens} tokens.")
        
        # Calculate the number of samples to skip.
        # This assumes a uniform sequence length, which is the case after the `group_texts` function.
        if len(train_dataset) > 0:
            sequence_length = len(train_dataset[0]['input_ids'])
            if sequence_length > 0:
                samples_to_skip = args.skip_tokens // sequence_length
                
                if samples_to_skip >= len(train_dataset):
                    logger.error(f"Cannot skip {args.skip_tokens} tokens ({samples_to_skip} samples) as it's more than or equal to the dataset size of {len(train_dataset)} samples. Exiting.")
                    if not args.single_gpu: cleanup_distributed()
                    return

                if is_main_process(args.single_gpu):
                    logger.info(f"Sequence length is {sequence_length}. Skipping {samples_to_skip} samples.")
                
                # Use .select() to create a new dataset view from the desired offset
                train_dataset = train_dataset.select(range(samples_to_skip, len(train_dataset)))
                # Update the number of tokens processed to reflect the skip
                tokens_processed = samples_to_skip * sequence_length
                
                if is_main_process(args.single_gpu):
                    logger.info(f"Dataset adjusted. New size: {len(train_dataset)} samples. Starting with 'tokens_processed' = {tokens_processed}")
            else:
                 if is_main_process(args.single_gpu):
                    logger.warning("Could not determine sequence length from the first sample (it's zero). Cannot skip tokens.")       
        else:
            if is_main_process(args.single_gpu):
                logger.warning("Training dataset is empty. Cannot skip tokens.")
    elif args.skip_tokens > 0 and args.resume_from_checkpoint:
        if is_main_process(args.single_gpu):
            logger.warning("--skip-tokens is ignored when --resume-from-checkpoint is used.")
    
    # Load validation dataset if it was processed
    eval_dataset = None
    cached_eval_dataset_path = cache_path + "_eval"
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

    logger.info("Initializing model...")
    
    # --- Model Initialization & Weight Loading ---
    # This logic handles creating a new model or loading weights from a checkpoint.
    # It's crucial to load weights *before* initializing DeepSpeed.
    model_config_args = {
        'vocab_size': len(tokenizer),
        'hidden_size': args.hidden_size,
        'num_hidden_layers': args.num_hidden_layers,
        'dt': args.dt,
        'use_moe': args.use_moe,
        'num_experts': args.num_experts,
        'num_experts_per_tok': args.num_experts_per_tok,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }

    # --- Model Initialization & Weight Loading ---
    # This logic handles creating a new model or loading weights from a checkpoint.
    # It's crucial to load weights *before* initializing DeepSpeed.
    if args.deepspeed_checkpoint:
        # When resuming, the checkpoint is the single source of truth for config and weights.
        logger.info(f"Preparing to resume from DeepSpeed checkpoint: {args.deepspeed_checkpoint}")
        # We load the CONFIG from the checkpoint path first to build the model structure.
        config = LNNConfig.from_pretrained(args.deepspeed_checkpoint)
        model = LNNModel(config)
        logger.info(f"Model structure loaded from config at {args.deepspeed_checkpoint}")
    elif args.model_name_or_path:
        # For starting a new run from a pretrained model (not a DS checkpoint).
        logger.info(f"Loading model weights from Hugging Face model: {args.model_name_or_path}")
        model = LNNModel.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
        logger.info("Model weights successfully loaded.")
    else:
        # If no checkpoint or model is provided, create a new model from scratch.
        logger.info("No checkpoint or model specified. Creating a new model from scratch.")
        config = LNNConfig(**model_config_args)
        model = LNNModel(config)

    # Log the precision strategy
    if args.bf16:
        logger.info("Model parameters in FP32, using BF16 mixed precision for forward/backward passes")
    elif args.fp16:
        logger.info("Model parameters in FP32, using FP16 mixed precision for forward/backward passes")
    else:
        logger.info("Using FP32 precision training")

    if args.gradient_checkpointing:
        logger.info("Gradient checkpointing enabled.")

    # Move model to the correct device. For DDP, this is handled by the DDP wrapper.
    if args.single_gpu:
        model.to(device)
    # Set up optimizer parameters, separating weight decay groups
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    # --- Optimizer --- 
    if args.deepspeed:
        optimizer = DeepSpeedCPUAdam(optimizer_grouped_parameters, lr=args.learning_rate, betas=args.adam_betas)
    else:
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            betas=args.adam_betas,
            weight_decay=args.weight_decay
        )

    # --- Learning Rate Scheduler ---
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    if args.num_warmup_steps is None:
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # --- DeepSpeed Initialization or DDP Wrapping ---
    if args.deepspeed:
        model, optimizer, _, lr_scheduler = deepspeed.initialize(
            args=args,
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler, # Pass our scheduler to DeepSpeed
            dist_init_required=False
        )
        
        # --- Load Checkpoint AFTER DeepSpeed Initialization ---
        if args.deepspeed_checkpoint:
            if model.local_rank == 0:
                logger.info(f"Loading DeepSpeed checkpoint into initialized engine from: {args.deepspeed_checkpoint}")
            
            load_path, client_state = model.load_checkpoint(args.deepspeed_checkpoint, load_module_strict=False)
            
            if load_path is None:
                raise RuntimeError(f"Failed to load DeepSpeed checkpoint from {args.deepspeed_checkpoint}")
            
            completed_steps = model.global_steps
            epoch = client_state.get('epoch', 0)
            tokens_processed = client_state.get('tokens_processed', 0)
            
            if model.local_rank == 0:
                logger.info(f"Successfully resumed from checkpoint. Step: {completed_steps}, Epoch: {epoch}")

            # CRITICAL: After resuming, we must reset the data sampler's epoch to ensure the data
            # loader continues from the exact same state. This prevents loss spikes caused by
            # seeing unexpected data batches.
            if not args.single_gpu and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
    else:
        # For standard DDP, we wrap the model after creating the optimizer and scheduler
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)

    # Enable anomaly detection if the debug flag is set
    if args.debug_nan:
        logger.warning("Enabling anomaly detection for debugging. This will slow down training.")
        torch.autograd.set_detect_anomaly(True)

    # Calculate total training steps. If max_train_steps is provided, it overrides num_train_epochs.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    else:
        # If max_train_steps is set, calculate the number of epochs for logging purposes.
        args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Calculate warmup steps. Priority is given to the explicit num_warmup_steps argument.
    if args.num_warmup_steps is None:
        args.num_warmup_steps = int(args.max_train_steps * args.warmup_ratio)
        logger.info(f"Calculated warmup steps from ratio: {args.num_warmup_steps}")
    else:
        logger.info(f"Using specified number of warmup steps: {args.num_warmup_steps}")

    if not args.deepspeed:
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps
        )

    # --- State Initialization for Training Loop ---
    # If not resuming from a DeepSpeed checkpoint, initialize state variables to zero.
    # Otherwise, they will have been set correctly during the checkpoint loading process.
    if not args.deepspeed_checkpoint:
        completed_steps = 0
        epoch = 0
        tokens_processed = 0

    # The main training loop uses `global_step` and `start_epoch`.
    # We initialize them from `completed_steps` and `epoch`, which are the
    # authoritative values either from the checkpoint or initialized to 0.
    global_step = completed_steps
    start_epoch = epoch




    logger.info("***** Running training *****")
    logger.info(f"  Total train batch size = {args.per_device_train_batch_size * world_size * args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    # Determine precision and autocast dtype
    if args.bf16:
        autocast_dtype = torch.bfloat16
        use_amp = True
        logger.info("Using BF16 mixed precision training")
    elif args.fp16:
        autocast_dtype = torch.float16
        use_amp = True
        logger.info("Using FP16 mixed precision training")
    else:
        autocast_dtype = torch.float32
        use_amp = False
        logger.info("Using FP32 precision training")
    
    if not args.deepspeed:
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
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
            labels = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}

            # --- Token Counting ---
            # We do this for every batch, regardless of gradient accumulation
            # batch['input_ids'].numel() gives batch_size * sequence_length
            batch_tokens = batch['input_ids'].numel() * world_size
            tokens_processed += batch_tokens

            # DeepSpeed handles autocasting internally based on its config.
            # We only need to manually manage autocast for standard DDP.
            if not args.deepspeed:
                with torch.amp.autocast(device_type='cuda', dtype=autocast_dtype):
                    outputs = model(**batch, labels=labels)
                    loss = outputs.loss
            else:
                # When using DeepSpeed, the forward pass is clean.
                outputs = model(**batch, labels=labels)
                loss = outputs.loss

            if args.deepspeed:
                model.backward(loss)
                model.step()
            else:
                scaler.scale(loss / args.gradient_accumulation_steps).backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    lr_scheduler.step()
                    optimizer.zero_grad()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                completed_steps += 1

                # --- Evaluation, Checkpointing, and Logging ---
                eval_loss = None
                # --- Evaluation (Collective Operation) ---
                if completed_steps > 0 and completed_steps % args.eval_steps == 0:
                    eval_loss = evaluate(model, eval_dataloader, device, autocast_dtype, args)

                # --- Checkpointing (Collective Operation) ---
                if completed_steps > 0 and completed_steps % args.checkpointing_steps == 0:
                    save_checkpoint(model, optimizer, lr_scheduler, tokenizer, args, completed_steps, epoch, tokens_processed)

                # --- Main Process Tasks (Logging, Progress Bar) ---
                if is_main_process(args.single_gpu):
                    current_loss = loss.item()
                    progress_bar.update(1)
                    if args.deepspeed:
                        current_lr = model.get_lr()[0]
                    else:
                        current_lr = lr_scheduler.get_last_lr()[0]
                    progress_bar.set_postfix(loss=f"{current_loss:.4f}", lr=f"{current_lr:.2e}")

                    if completed_steps > 0 and completed_steps % args.logging_steps == 0:
                        # Calculate token-level accuracy for the current batch
                        with torch.no_grad():
                            # For causal language models, logits are shifted relative to labels.
                            # We need to align them before calculating accuracy.
                            # The prediction for token i is logits[:, i-1, :]
                            # So, we compare logits[:, :-1, :] with labels[:, 1:, :]
                            logits = outputs.logits[:, :-1, :].contiguous()
                            target_labels = labels[:, 1:].contiguous()

                            predictions = torch.argmax(logits, dim=-1)
                            
                            # Mask out padding tokens (-100) for accuracy calculation
                            mask = target_labels != -100
                            
                            # Ensure predictions and mask have the same shape
                            correct_predictions = (predictions == target_labels) & mask
                            
                            # Calculate accuracy
                            if mask.sum() > 0:
                                token_accuracy = correct_predictions.sum().float() / mask.sum().float()
                                
                                # --- Calculate Top-k Accuracy ---
                                # Get top 10 predictions. Shape: (batch, seq_len, 10)
                                _, top_k_preds = torch.topk(logits, k=10, dim=-1)
                                # Unsqueeze labels to shape (batch, seq_len, 1) for broadcasting
                                unsqueezed_labels = target_labels.unsqueeze(-1)

                                # Compare labels against top 5 and top 10 predictions
                                top_5_correct = (top_k_preds[:, :, :5] == unsqueezed_labels).any(dim=-1) & mask
                                top_10_correct = (top_k_preds == unsqueezed_labels).any(dim=-1) & mask

                                top_5_accuracy = top_5_correct.sum().float() / mask.sum().float()
                                top_10_accuracy = top_10_correct.sum().float() / mask.sum().float()

                                # --- Calculate Logit Entropy (Confidence) ---
                                # Use log_softmax for numerical stability
                                log_probs = F.log_softmax(logits, dim=-1)
                                probs = torch.exp(log_probs)
                                entropy = -(probs * log_probs).sum(dim=-1)
                                # Average entropy only over non-masked tokens
                                avg_entropy = (entropy * mask).sum() / mask.sum().float()

                            else:
                                token_accuracy = torch.tensor(0.0, device=logits.device)
                                top_5_accuracy = torch.tensor(0.0, device=logits.device)
                                top_10_accuracy = torch.tensor(0.0, device=logits.device)
                                avg_entropy = torch.tensor(0.0, device=logits.device)

                        # --- Perplexity --- 
                        perplexity = torch.exp(torch.tensor(current_loss))

                        log_stats = {
                            "train/loss": current_loss,
                            "train/perplexity": perplexity.item(),
                            "train/learning_rate": current_lr,
                            "tokens_processed": tokens_processed,
                            "epoch": epoch,
                            "train/token_accuracy": token_accuracy.item(),
                            "train/top_5_accuracy": top_5_accuracy.item(),
                            "train/top_10_accuracy": top_10_accuracy.item(),
                            "train/logit_entropy": avg_entropy.item()
                        }
                        if eval_loss is not None:
                            log_stats["eval/loss"] = eval_loss
                        
                        wandb.log(log_stats, step=completed_steps)

                # --- Synchronization Barrier ---
                # All processes must wait here before starting the next step.
                if dist.is_initialized():
                    dist.barrier()

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

    # Finish the W&B run to ensure all data is synced and processes are cleaned up
    if is_main_process(args.single_gpu):
        wandb.finish()

    if not args.single_gpu:
        cleanup_distributed()

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
