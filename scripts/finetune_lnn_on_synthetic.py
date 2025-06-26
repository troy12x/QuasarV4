import os
import sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, Trainer

# Add the project root to the Python path to allow importing from the 'quasar' package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# The LNNModel and LNNConfig must be registered for AutoModel to find them.
from quasar.lnn import LNNModel, LNNConfig

# --- Configuration ---
MODEL_ID = "silx-ai/QuasarV4-8B-A3B-LNN"
DATASET_ID = "Gaoj124/pretraining_synthetic_long_100_5_real_random_examples"
OUTPUT_DIR = "./quasar_v4_8b_finetuned"

# --- 1. Load Tokenizer and Model ---
print(f"Loading tokenizer and model for {MODEL_ID}...")

# Note: You might need to log in to Hugging Face if the model is gated.
# from huggingface_hub import login
# login("YOUR_HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = LNNModel.from_pretrained(MODEL_ID, trust_remote_code=True, torch_dtype=torch.bfloat16)
# The Trainer will handle moving the model to the correct device.

print("Model and tokenizer loaded successfully.")

# --- 2. Load and Prepare Dataset ---
print(f"Loading and preparing dataset: {DATASET_ID}")
dataset = load_dataset(DATASET_ID, split="train")

# Define the tokenization function
def tokenize_function(examples):
    # The 'text' column will be tokenized.
    return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

# Tokenize the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
print("Dataset tokenized.")

# --- 3. Set up Training ---

# Define training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,              # Fine-tuning usually requires fewer epochs
    per_device_train_batch_size=1,   # Adjust based on your GPU memory
    gradient_accumulation_steps=4,   # Effective batch size = 1 * 4 = 4
    learning_rate=2e-5,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_steps=500,
    fp16=False, # Set to True if your GPU supports it and you have apex installed
    bf16=True,  # Recommended for Ampere GPUs and newer
)

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# --- 4. Start Fine-Tuning ---
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# --- 5. Save the Final Model ---
final_model_path = f"{OUTPUT_DIR}/final_model"
print(f"Saving the fine-tuned model to {final_model_path}")
trainer.save_model(final_model_path)
print("Model saved successfully.")
