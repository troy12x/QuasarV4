# scripts/pretrain_framework.py
# An FSDP-based framework for pre-training LNN models on large datasets.

import argparse
import os
import sys

import torch
from datasets import load_dataset, IterableDataset
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

# Add project root to the Python path to allow importing the 'quasar' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quasar.lnn import LNNModel, LNNConfig, LNNBlock

def tokenize_and_pack_generator(dataset, tokenizer, sequence_length):
    """
    Generator function to tokenize and pack text data into fixed-length sequences.
    This is memory-efficient for large datasets.
    """
    buffer = []
    for examples in iter(dataset):
        text = examples.get("text")
        if not isinstance(text, str):
            continue
        tokenized = tokenizer(text, truncation=False, padding=False)["input_ids"]
        buffer.extend(tokenized)
        while len(buffer) >= sequence_length:
            chunk = buffer[:sequence_length]
            buffer = buffer[sequence_length:]
            yield {"input_ids": chunk, "labels": chunk}

def main():
    parser = argparse.ArgumentParser(description="FSDP Pre-training Framework for LNN Models")

    # --- Core Arguments ---
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")

    # --- Model Loading/Creation ---
    parser.add_argument("--model_name_or_path", type=str, default="silx-ai/QuasarV4-8B-A3B-LNN", help="Load a pretrained model from this path. If None, create a new model.")
    parser.add_argument("--use_moe", action="store_true", help="Enable Mixture of Experts layers when creating a new model.")
    # Add other model config arguments if needed for 'create from scratch' mode

    # --- Tokenizer and Dataset ---
    parser.add_argument("--tokenizer_name", type=str, default="deepseek-ai/DeepSeek-V3-0324", help="Tokenizer to use.")
    parser.add_argument("--dataset_name", type=str, default="openbmb/Ultra-FineWeb", help="Dataset to use for pre-training.")
    parser.add_argument("--dataset_config_name", type=str, default="en", help="Dataset configuration (e.g., 'en' for English split).")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length for packing.")

    # --- Training Arguments (passed to Trainer) ---
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Peak learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of steps to accumulate gradients.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=250, help="Save a checkpoint every N steps.")

    args = parser.parse_args()

    # --- 1. Register and Load Tokenizer/Model ---
    print("--- Initializing Model and Tokenizer ---")
    AutoConfig.register("quasar", LNNConfig)
    AutoModelForCausalLM.register(LNNConfig, LNNModel)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        print("Tokenizer does not have a pad token, setting it to eos_token.")
        tokenizer.pad_token = tokenizer.eos_token

    if args.model_name_or_path:
        print(f"Loading model from Hugging Face Hub: {args.model_name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16, # Load in bf16 to save memory
        )
    else:
        # This block is for creating a new model from scratch.
        # You can expand this with more arguments (hidden_size, num_layers, etc.)
        print("Creating a new LNN model from scratch...")
        config = LNNConfig(
            vocab_size=tokenizer.vocab_size,
            use_moe=args.use_moe,
            # ... add other config parameters here ...
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = LNNModel(config)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model initialized with {total_params:.2f}M parameters.")

    # --- 2. Load and Process Dataset ---
    print(f"Loading streaming dataset: {args.dataset_name} (Split: {args.dataset_config_name})")
    raw_dataset = load_dataset(args.dataset_name, args.dataset_config_name, split="train", streaming=True)
    # The dataset is huge, so we'll take a subset for demonstration purposes if needed.
    # raw_dataset = raw_dataset.take(10000) # Uncomment to test with a smaller sample
    train_dataset = IterableDataset.from_generator(
        tokenize_and_pack_generator,
        gen_kwargs={"dataset": raw_dataset, "tokenizer": tokenizer, "sequence_length": args.sequence_length},
    )

    # --- 3. Configure and Start Training with FSDP ---
    # FSDP config is passed directly to TrainingArguments.
    # We use 'full_shard' for maximum memory savings and 'auto_wrap' to automatically
    # wrap the LNNBlock layers.
    fsdp_config = {
        "fsdp_transformer_layer_cls_to_wrap": [LNNBlock.__name__],
        "fsdp_offload_params": False, # Crucially, we keep parameters on the GPU.
        "fsdp_auto_wrap_policy": "transformer_auto_wrap_policy",
        "fsdp_sharding_strategy": "FULL_SHARD",
        "fsdp_backward_prefetch": "backward_pre",
        "fsdp_forward_prefetch": True,
        "fsdp_use_orig_params": True, # Required for some new FSDP features and parameter tying
    }

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="wandb",
        # --- FSDP & Performance ---
        fsdp="full_shard",
        fsdp_config=fsdp_config,
        bf16=True, # Use bfloat16 for H100 performance
        bf16_full_eval=True,
        # --- Logging and Saving ---
        logging_first_step=True,
        save_strategy="steps",
        save_total_limit=3, # Keep only the last 3 checkpoints
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("--- Starting Pre-training with FSDP --- ")
    trainer.train()
    print("--- Pre-training Complete ---")

if __name__ == "__main__":
    main()
