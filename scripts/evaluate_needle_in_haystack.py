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
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.patches as mpatches

# Add project root to path to allow importing quasar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quasar.lnn import LNNModel, LNNConfig

# --- 1. Data Generation ---

def generate_needle_training_batch(batch_size, seq_len, vocab_size, needle_token):
    """Generates a batch for training. The needle is always the last token to predict."""
    batch_inputs = []
    for _ in range(batch_size):
        haystack = np.random.randint(1, vocab_size, size=seq_len - 1, dtype=np.int64)
        sequence = np.append(haystack, needle_token)
        batch_inputs.append(sequence)
    
    inputs_np = np.array(batch_inputs, dtype=np.int64)
    inputs = torch.from_numpy(inputs_np[:, :-1])
    labels = torch.from_numpy(inputs_np[:, -1])
    return inputs, labels

# --- 2. Training and Evaluation ---

def train_on_needle_task(model, device, vocab_size, needle_token, training_steps=1000, seq_len=512):
    """Trains the LNN model on the needle-in-a-haystack task."""
    print("\n--- Training LNN on Needle Task ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) # Reduced for stability
    criterion = nn.CrossEntropyLoss()

    pbar = tqdm(range(training_steps), desc="Training on Needle Task")
    chunk_size = 512
    print(f"--- Using Truncated Backpropagation Through Time with chunk size: {chunk_size} ---")

    for step in pbar:
        inputs, labels = generate_needle_training_batch(
            batch_size=8, # Reduced to fit longer sequences in memory
            seq_len=seq_len,
            vocab_size=vocab_size,
            needle_token=needle_token
        )
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # --- TBPTT Implementation ---
        hidden_states = None
        total_loss = 0

        # Process sequence in chunks
        chunks = torch.split(inputs, chunk_size, dim=1)
        for i, chunk in enumerate(chunks):
            is_last_chunk = (i == len(chunks) - 1)
            logits, hidden_states = model(chunk, hidden_states)

            # Detach hidden states to truncate the gradient path
            # This is the core of TBPTT
            hidden_states = [h.detach() for h in hidden_states]

            if is_last_chunk:
                last_logits = logits[:, -1, :]
                loss = criterion(last_logits, labels)
                loss.backward()
                total_loss = loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
        optimizer.step()

        pbar.set_postfix(loss=f"{total_loss:.4f}")
    print("--- Training Complete ---")


def evaluate_needle_retrieval(model, context_length, needle_depth_percent, device, vocab_size, needle_token):
    """ 
    Tests the model's ability to retrieve a needle from a haystack.
    Returns True if successful, False otherwise.
    """
    model.eval()
    print(f"\n-- Testing Context: {context_length}, Needle Depth: {needle_depth_percent}% --")

    needle_pos = int(context_length * (needle_depth_percent / 100))

    haystack = np.random.randint(1, vocab_size, size=context_length, dtype=np.int64)
    haystack[needle_pos] = needle_token
    
    input_tensor = torch.from_numpy(haystack).long().to(device).unsqueeze(0)

    with torch.no_grad():
        hidden_states = None
        chunk_size = 16384

        input_sequence = input_tensor[:, :needle_pos]

        # Process context in chunks
        for i in range(0, input_sequence.size(1), chunk_size):
            end = min(i + chunk_size, input_sequence.size(1))
            chunk = input_sequence[:, i:end]
            if chunk.size(1) > 0:
                logits, hidden_states = model(chunk, hidden_states)
            else: # Handle case where needle is at the beginning
                logits = torch.zeros(1, 1, model.config.vocab_size).to(device)

        prediction = torch.argmax(logits[:, -1, :], dim=-1).item()

    if prediction == needle_token:
        print(f"  -> Success: Needle Found! (Predicted: {prediction})")
        return True
    else:
        print(f"  -> Failure: Needle Not Found. (Predicted: {prediction}, Expected: {needle_token})")
        return False

# --- 3. Plotting ---

def format_token_count(n):
    """Formats a number into a human-readable string like 10K, 1.4M."""
    if n >= 1_000_000:
        return f'{n/1_000_000:.1f}M'.replace('.0', '')
    if n >= 1_000:
        return f'{n/1_000:.0f}K'
    return str(n)

def plot_results(results, context_lengths, needle_depths, save_path="needle_in_haystack_results.png"):
    """Plots the needle-in-a-haystack results as a heatmap and saves it."""
    print(f"\n--- Generating and saving results plot to {save_path} ---")

    # Convert results dict to a DataFrame for plotting
    df = pd.DataFrame(results).reindex(index=needle_depths, columns=context_lengths)
    df = df.apply(pd.to_numeric) # Convert bools to 0/1

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 8), dpi=200)
    sns.set_theme(style="white")

    # Custom colormap: 0=Failure (light grey), 1=Success (blue)
    cmap = ['#EAEAF2', '#007BFF']

    sns.heatmap(
        df,
        ax=ax,
        cmap=cmap,
        linewidths=2,
        linecolor='white',
        cbar=False,
        square=True,
        vmin=0, # Explicitly set the color map range
        vmax=1  # To handle cases with all-success or all-failure
    )

    # --- Style the plot to match the reference ---
    # Set title and subtitle
    fig.suptitle('Needle-in-a-haystack (NiH)', fontsize=20, fontweight='bold', ha='center', y=0.99)
    max_context_str = format_token_count(max(context_lengths))
    ax.set_title(f'Quasar LNN Model\nBelow, text NiH up to {max_context_str} tokens', fontsize=14, loc='left', y=1.0, pad=30)

    # Set axis labels and ticks
    ax.set_xlabel('Context Length (Tokens)', fontsize=14, labelpad=15)
    ax.set_ylabel('Depth (%)', fontsize=14, labelpad=15)

    # Format x-axis ticks
    ax.set_xticklabels([format_token_count(l) for l in context_lengths], rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    # Remove tick marks for a cleaner look
    ax.tick_params(axis='both', which='both', length=0)

    # --- Create custom legend ---
    success_patch = mpatches.Patch(color='#007BFF', label='Successful retrieval')
    failure_patch = mpatches.Patch(facecolor='white', edgecolor='darkgrey', label='Failure to retrieve')
    ax.legend(
        handles=[success_patch, failure_patch],
        loc='upper left',
        bbox_to_anchor=(-0.02, 1.25),
        frameon=False,
        fontsize=12,
        handletextpad=0.8,
        labelspacing=0.8,
        ncol=2
    )

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.9]) # Adjust for suptitle
    fig.savefig(save_path, bbox_inches='tight')
    print(f"--- Plot saved successfully to {save_path}. ---")


# --- 4. Main Execution ---

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model Setup ---
    VOCAB_SIZE = 500
    needle_token = VOCAB_SIZE + 1
    lnn_config = LNNConfig(vocab_size=VOCAB_SIZE + 10, hidden_size=256, num_hidden_layers=4)
    lnn_model = LNNModel(lnn_config).to(device)

    # --- Pre-train the Model ---
    train_on_needle_task(lnn_model, device, VOCAB_SIZE, needle_token, training_steps=100, seq_len=2048)

    # --- Benchmarking Parameters ---
    context_lengths = [2_000_000]
    needle_depths = [50] # Test at 50% depth
    results = {}

    for length in context_lengths:
        results[length] = {}
        for depth in needle_depths:
            success = evaluate_needle_retrieval(lnn_model, length, depth, device, VOCAB_SIZE, needle_token)
            results[length][depth] = success
    
    # --- Print Final Report ---
    print("\n--- Needle in a Haystack Final Report ---")
    header = f"{'Context Length':<16} | " + " | ".join([f"{d}% Depth" for d in needle_depths])
    print(header)
    print("-" * len(header))

    for length in context_lengths:
        row = f"{length:<16} | "
        row += " | ".join([" Success " if results[length][d] else " FAILURE " for d in needle_depths])
        print(row)

    # --- Generate and Save Plot ---
    plot_results(results, context_lengths, needle_depths)

if __name__ == "__main__":
    main()
