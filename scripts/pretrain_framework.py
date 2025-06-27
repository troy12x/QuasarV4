# scripts/pretrain_framework.py
# An FSDP-based framework for pre-training LNN models on large datasets.

import argparse
import os
import sys

import torch
from datasets import Dataset, Features, Value, Sequence, load_from_disk
from transformers import (
    Trainer,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq
from itertools import chain

# Add project root to the Python path to allow importing the 'quasar' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quasar.lnn import LNNModel, LNNConfig, LNNBlock



def main():
    parser = argparse.ArgumentParser(description="FSDP Pre-training Framework for LNN Models")

    # --- Core Arguments ---
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints and logs.")

    # --- Model Loading/Creation ---
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Load a pretrained model. If None, creates a new model from scratch.")
    parser.add_argument("--use_moe", action="store_true", help="Enable Mixture of Experts layers when creating a new model.")
    
    # --- New Model Configuration (for 'from scratch') ---
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for a new model.")
    parser.add_argument("--num_hidden_layers", type=int, default=12, help="Number of layers for a new model.")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step 'dt' for the LNN cells.")

    # --- Tokenizer and Dataset ---
    parser.add_argument("--tokenizer_name", type=str, default="deepseek-ai/deepseek-v2", help="Tokenizer to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to tokenize from the Hub.")
    parser.add_argument("--dataset_split", type=str, required=True, help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default="content", help="The column in the dataset containing the text.")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length for packing.")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of CPU processes for tokenization.")

    # --- Training Arguments (passed to Trainer) ---
    parser.add_argument("--batch_size", type=int, default=1, help="Per-device training batch size.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Peak learning rate.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16, help="Number of steps to accumulate gradients.")
    parser.add_argument("--max_steps", type=int, default=10000, help="Total number of training steps.")
    parser.add_argument("--logging_steps", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_steps", type=int, default=250, help="Save a checkpoint every N steps.")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 training.")
    parser.add_argument("--wandb_project", type=str, default=None, help="Weights & Biases project name.")

    args = parser.parse_args()

    # --- 1. Register and Load Tokenizer/Model ---
    print("--- Initializing Model and Tokenizer ---")
    AutoConfig.register("quasar", LNNConfig)
    AutoModelForCausalLM.register(LNNConfig, LNNModel)

    print(f"Loading tokenizer: {args.tokenizer_name}")
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
        print("Creating a new LNN model from scratch...")
        config = LNNConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            num_hidden_layers=args.num_hidden_layers,
            dt=args.dt,
            use_moe=args.use_moe,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        model = LNNModel(config)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model initialized with {total_params:.2f}M parameters.")

    # --- 2. Load, Tokenize, and Pack Dataset On-the-Fly ---
    print("--- Starting On-the-Fly Data Processing ---")

    def stream_parquet_split(repo_id, split, text_column):
        api = HfApi()
        repo_info = api.repo_info(repo_id, repo_type="dataset")
        prefix = "data/"
        files = sorted([f.rfilename for f in repo_info.siblings if f.rfilename.endswith(".parquet") and f.rfilename.startswith(prefix) and split in f.rfilename])
        if not files: raise ValueError(f"No Parquet files found for split '{split}' in repo '{repo_id}'.")
        print(f"Found {len(files)} Parquet files for split '{split}'.")
        for filename in files:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            table = pq.read_table(local_path, columns=[text_column])
            for batch in table.to_batches():
                for text in batch.to_pydict()[text_column]:
                    if text: yield {text_column: text}

    raw_features = Features({args.text_column: Value('string')})
    raw_dataset = Dataset.from_generator(
        lambda: stream_parquet_split(args.dataset_name, args.dataset_split, args.text_column),
        features=raw_features
    )

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], truncation=False, padding=False)

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=[args.text_column]
    )

    def pack_iterator(dataset, sequence_length):
        buffer = []
        for example in dataset:
            buffer.extend(example['input_ids'])
            while len(buffer) >= sequence_length:
                chunk = buffer[:sequence_length]
                buffer = buffer[sequence_length:]
                yield {"input_ids": chunk, "labels": chunk.copy()}

    packed_features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

    train_dataset = Dataset.from_generator(
        lambda: pack_iterator(tokenized_dataset, args.sequence_length),
        features=packed_features
    )
    train_dataset.set_format("torch")
    print("Dataset processed and packed successfully.")

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

    # Set W&B project if provided
    if args.wandb_project:
        os.environ["WANDB_PROJECT"] = args.wandb_project

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        report_to="wandb" if args.wandb_project else None,
        # --- FSDP & Performance ---
        fsdp="full_shard",
        fsdp_config=fsdp_config,
        bf16=args.bf16,
        bf16_full_eval=args.bf16,
        # --- Logging and Saving ---
        logging_first_step=True,
        save_strategy="steps",
        save_total_limit=3, # Keep only the last 3 checkpoints
        dataloader_num_workers=0, # CRITICAL: Set to 0 to avoid dataloader deadlocks in FSDP
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        # Although the dataset is pre-packed, using the official data collator
        # is more robust and ensures the data is structured exactly as the Trainer
        # expects, especially in a distributed FSDP environment.
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("--- Starting Pre-training with FSDP --- ")
    trainer.train()
    print("--- Pre-training Complete ---")

if __name__ == "__main__":
    main()



