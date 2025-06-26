"""
High-performance pre-training script for QuasarV4 LNN models.

This script is designed for large-scale pre-training on multi-GPU systems
using Hugging Face Accelerate and DeepSpeed.

Key Features:
- Loads a base model and adapts it to a new tokenizer.
- Streams a large-scale dataset from the Hugging Face Hub.
- Tokenizes and packs the dataset into fixed-length sequences on-the-fly.
- Uses the Hugging Face Trainer for a robust and feature-rich training loop.

Example command to run (requires accelerate config):

accelerate launch scripts/pretrain_lnn.py \
    --model_name_or_path silx-ai/QuasarV4-8B-A3B-LNN \
    --tokenizer_name_or_path deepseek-ai/deepseek-coder-v2-lite-instruct \
    --dataset_name togethercomputer/RedPajama-Data-1T-Sample \
    --output_dir ./pretrain_output \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --bf16 True \
    --logging_steps 10 \
    --save_steps 500 \
    --num_train_epochs 1
"""

import argparse
import os
import sys

import torch
from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)

# Add project root to the Python path to allow importing 'quasar'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quasar.lnn import LNNModel, LNNConfig

def main(args):
    # --- 1. Load Tokenizer ---
    # The tokenizer is loaded first to determine the vocabulary size.
    print(f"Loading tokenizer from: {args.tokenizer_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path, trust_remote_code=True)
    # Set a padding token if it doesn't exist. This is required for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- 2. Load Model ---
    print(f"Loading model from: {args.model_name_or_path}")
    # We register our custom LNNModel class so AutoModelForCausalLM can find it.
    AutoConfig.register("lnn", LNNConfig)
    LNNModel.register_for_auto_class("AutoModelForCausalLM")

    model = LNNModel.from_pretrained(args.model_name_or_path)

    # --- 3. Resize Model Embeddings ---
    # The base model may have a different vocab size than our new tokenizer.
    # We resize the token embeddings to match, which is a critical step.
    current_vocab_size = model.config.vocab_size
    new_vocab_size = len(tokenizer)

    if current_vocab_size != new_vocab_size:
        print(f"Resizing model token embeddings from {current_vocab_size} to {new_vocab_size}")
        model.resize_token_embeddings(new_vocab_size)
        # The model's config needs to be updated with the new vocab size.
        model.config.vocab_size = new_vocab_size
    else:
        print(f"Model vocabulary size already matches tokenizer ({new_vocab_size}). No resize needed.")

    # --- 4. Load and Process Dataset ---
    print(f"Loading and processing dataset: {args.dataset_name}")
    # Use streaming to handle massive datasets without downloading them entirely.
    # The dataset is expected to have a 'text' column.
    raw_dataset = load_dataset(args.dataset_name, split="train", streaming=True)

    # This function tokenizes and packs the text into fixed-length blocks.
    def tokenize_and_pack(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= args.sequence_length:
            total_length = (total_length // args.sequence_length) * args.sequence_length
        # Split by chunks of sequence_length.
        result = {
            k: [t[i : i + args.sequence_length] for i in range(0, total_length, args.sequence_length)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Tokenize the dataset on-the-fly.
    tokenized_dataset = raw_dataset.map(
        lambda examples: tokenizer(examples["text"]),
        batched=True,
    )

    # Pack the tokenized dataset into fixed-length sequences.
    train_dataset = tokenized_dataset.map(
        tokenize_and_pack,
        batched=True,
    )

    # --- 5. Configure Training ---
    print("Configuring training arguments...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="tensorboard",
        deepspeed=args.deepspeed_config, # Path to DeepSpeed config file
        gradient_checkpointing=True, # Saves memory
    )

    # Data collator for language modeling. It handles creating the 'labels' automatically.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # --- 6. Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- 7. Start Training ---
    print("Starting pre-training...")
    trainer.train()

    # --- 8. Save Final Model ---
    print("Training complete. Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train an LNN model.")

    # Model and Tokenizer arguments
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Path to the tokenizer.")

    # Dataset arguments
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset on Hugging Face Hub.")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length for packing.")

    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=2, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8, help="Steps for gradient accumulation.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs.")
    parser.add_argument("--bf16", type=bool, default=True, help="Use bfloat16 mixed-precision training.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log training status every N steps.")
    parser.add_argument("--save_steps", type=int, default=500, help="Save a checkpoint every N steps.")
    parser.add_argument("--deepspeed_config", type=str, default=None, help="Path to the DeepSpeed config file.")

    args = parser.parse_args()
    main(args)
