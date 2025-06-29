# Copyright 2024 Quasar AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path to allow importing quasar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quasar.lnn import LNNModel, LNNConfig
from quasar.transformer import TransformerModel, TransformerConfig

# --- 1. Synthetic Memory-Retrieval Data Generation ---
def generate_retrieval_batch(batch_size, seq_len, vocab_size, key_value_map):
    """
    Generates a batch of sequences for the key-value retrieval task from a fixed map.
    """
    batch_inputs = []
    batch_labels = []
    
    # Get a list of all possible keys to sample from
    possible_keys = list(key_value_map.keys())

    for _ in range(batch_size):
        # Sample a key-value pair for this sequence
        key_token = np.random.choice(possible_keys)
        value_token = key_value_map[key_token]

        # Generate random filler text
        sequence = np.random.randint(1, vocab_size, size=seq_len, dtype=np.int64)
        
        # Place key-value pair at a fixed position for simplicity in training
        sequence[1] = key_token
        sequence[2] = value_token
        sequence[-2] = key_token # Query for the key
        
        batch_inputs.append(sequence)
        batch_labels.append(value_token)

    return torch.from_numpy(np.array(batch_inputs, dtype=np.int64)), torch.LongTensor(batch_labels)

# --- 2. Evaluation Function ---
def train_retrieval_model(model, optimizer, criterion, device, vocab_size, key_value_map, training_steps=1400, seq_len=128):
    """
    Trains the model on the key-value retrieval task.
    """
    model.train()
    for step in range(training_steps):
        inputs, labels = generate_retrieval_batch(batch_size=32, seq_len=seq_len, vocab_size=vocab_size, key_value_map=key_value_map)
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        # Get model output
        if isinstance(model, TransformerModel):
            output = model(input_ids=inputs)
        else:
            output = model(inputs)

        # Unpack logits
        if isinstance(model, LNNModel):
            logits = output[0]
        else:
            logits = output

        # Calculate loss only on the last token prediction
        last_token_logits = logits[:, -1, :]
        loss = criterion(last_token_logits, labels)

        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            print(f"  Training Step {step+1}/{training_steps}, Loss: {loss.item():.4f}")

def evaluate_retrieval_accuracy(model, seq_len, vocab_size, device, key_value_map):
    """
    Evaluates the model's accuracy on the retrieval task for a given sequence length,
    processing one item at a time to conserve memory.
    """
    model.eval()
    correct = 0
    total = 50  # Number of random trials to average over

    # Generate the full batch of test data on the CPU first
    inputs_cpu, labels_cpu = generate_retrieval_batch(batch_size=total, seq_len=seq_len, vocab_size=vocab_size, key_value_map=key_value_map)

    with torch.no_grad():
        for i in range(total):
            # Process one sequence at a time
            input_seq = inputs_cpu[i:i+1].to(device)
            label = labels_cpu[i].item()

            if isinstance(model, TransformerModel):
                # Transformer needs input_ids kwarg
                output = model(input_ids=input_seq)
            else:
                # LNN takes positional arg and returns a tuple
                output, _ = model(input_seq)

            # We only care about the prediction for the very last token
            last_logit = output[:, -1, :]
            prediction = torch.argmax(last_logit, dim=-1).item()
            
            if prediction == label:
                correct += 1

    return (correct / total) * 100

# --- 3. Main Execution ---
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Setup ---
    VOCAB_SIZE = 500 # Base vocabulary
    NUM_KEY_VALUE_PAIRS = 100

    # Create a fixed map of key-value pairs for the entire run
    # Keys and values are high-ID tokens outside the main vocabulary
    keys = np.arange(VOCAB_SIZE, VOCAB_SIZE + NUM_KEY_VALUE_PAIRS * 2, 2)
    values = keys + 1
    key_value_map = {k: v for k, v in zip(keys, values)}
    
    # LNN Model
    lnn_config = LNNConfig(vocab_size=VOCAB_SIZE*2, hidden_size=128, num_hidden_layers=2)
    lnn_model = LNNModel(lnn_config).to(device)
    
    # Transformer Model (matched params)
    transformer_config = TransformerConfig(vocab_size=VOCAB_SIZE*2, hidden_size=128, num_layers=2, num_heads=2, dim_feedforward=256)
    transformer_model = TransformerModel(transformer_config).to(device)

    # --- Model Training ---
    criterion = nn.CrossEntropyLoss()

    print("\n--- Training LNN Model on Retrieval Task ---")
    lnn_optimizer = torch.optim.AdamW(lnn_model.parameters(), lr=1e-3)
    train_retrieval_model(lnn_model, lnn_optimizer, criterion, device, VOCAB_SIZE, key_value_map)

    print("\n--- Training Transformer Model on Retrieval Task ---")
    transformer_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=1e-3)
    train_retrieval_model(transformer_model, transformer_optimizer, criterion, device, VOCAB_SIZE, key_value_map)

    # --- Benchmarking Memory Retrieval --- 
    print("\n--- Benchmarking In-Context Memory Retrieval ---")
    sequence_lengths = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    lnn_accuracies = []
    transformer_accuracies = []

    transformer_failed = False
    for length in sequence_lengths:
        print(f"Testing sequence length: {length}...")
        # LNN has constant memory, so it should not fail.
        lnn_acc = evaluate_retrieval_accuracy(lnn_model, length, VOCAB_SIZE, device, key_value_map)
        lnn_accuracies.append(lnn_acc)

        # Transformer has quadratic memory and is expected to fail.
        if not transformer_failed:
            try:
                transformer_acc = evaluate_retrieval_accuracy(transformer_model, length, VOCAB_SIZE, device, key_value_map)
                transformer_accuracies.append(transformer_acc)
            except torch.OutOfMemoryError:
                print(f"  --> Transformer ran out of memory at sequence length {length}. This is expected.")
                transformer_failed = True
                transformer_accuracies.append(float('nan'))
        else:
            # If it failed once, it will fail for all longer sequences.
            transformer_accuracies.append(float('nan'))

    # --- Final Results ---
    print("\n--- Memory Retrieval Accuracy vs. Sequence Length ---")
    print(f"\n--- Memory Retrieval Accuracy vs. Sequence Length ---")
    print(f"Sequence Length | LNN Accuracy (%) | Transformer Accuracy (%)")
    print(f"----------------|------------------|--------------------------")
    for i, length in enumerate(sequence_lengths):
        lnn_result = f"{lnn_accuracies[i]:.2f}"
        trans_acc = transformer_accuracies[i]
        trans_result = "Out of Memory" if np.isnan(trans_acc) else f"{trans_acc:.2f}"
        print(f"{length:<15} | {lnn_result:<16} | {trans_result:<24}")

if __name__ == "__main__":
    main()
