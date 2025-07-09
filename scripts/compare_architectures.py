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

import os
import sys
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import pandas as pd
from dataclasses import dataclass

# --- Setup Paths ---
# Add project root to sys.path to allow for local package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath("./LFM"))

# --- Model Imports ---
# Quasar LNN Model
from quasar.lnn import LNNModel, LNNConfig
# Liquid Transformer Model
from lfm_torch.liquid_t_moe import LiquidTransformer

# --- Experiment Configuration ---
# Use a simple text for the dummy dataset
DUMMY_TEXT = "abcdefghijklmnopqrstuvwxyz " * 100
SEQ_LENGTH = 32
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2 # Train for a few epochs to see the learning trend
VALIDATION_SPLIT = 0.2
SKIP_LNN_TRAINING = True # Set to True to skip LNN training for faster debugging

# --- Model Configurations (simplified and matched for comparison) ---
# We will create small models to run quickly on the dummy data.
# The key is to have a similar number of parameters.

# Shared config
VOCAB_SIZE = len(set(DUMMY_TEXT))
HIDDEN_SIZE = 128
NUM_LAYERS = 4

LNN_MODEL_CONFIG = LNNConfig(
    vocab_size=VOCAB_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_hidden_layers=NUM_LAYERS,
)

@dataclass
class LFMConfig:
    vocab_size: int = VOCAB_SIZE
    embed_size: int = HIDDEN_SIZE
    num_heads: int = 4
    num_experts: int = 2
    expert_size: int = HIDDEN_SIZE
    num_layers: int = NUM_LAYERS

# --- LFM Wrapper for Causal LM ---
class LFMForCausalLM(nn.Module):
    """Wraps the LiquidTransformer to add embedding and a language model head."""
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.embed_size)
        self.transformer = LiquidTransformer(
            embed_size=config.embed_size,
            num_heads=config.num_heads,
            num_experts=config.num_experts,
            expert_size=config.expert_size,
            num_layers=config.num_layers
        )
        self.lm_head = nn.Linear(config.embed_size, config.vocab_size)
        self.transformer.hidden_state = None

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        if self.transformer.hidden_state is None or self.transformer.hidden_state.size(0) != batch_size:
            self.transformer.hidden_state = torch.zeros(batch_size, self.config.embed_size, device=input_ids.device)

        embedded_input = self.embedding(input_ids)
        transformer_output = self.transformer(embedded_input.unsqueeze(0))
        logits = self.lm_head(transformer_output)

        # Create a simple output object for compatibility
        @dataclass
        class LFMOutput:
            loss: torch.Tensor
            logits: torch.Tensor

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        return LFMOutput(loss=loss, logits=logits)

# --- Dummy Dataset ---
class TextDataset(Dataset):
    """A simple dataset to serve chunks of text."""
    def __init__(self, data, seq_length, char_to_int):
        self.data = data
        self.seq_length = seq_length
        self.char_to_int = char_to_int

    def __len__(self):
        return len(self.data) - self.seq_length - 1

    def __getitem__(self, idx):
        inputs = torch.tensor([self.char_to_int[c] for c in self.data[idx : idx + self.seq_length]], dtype=torch.long)
        # Target is the next character in the sequence
        targets = torch.tensor([self.char_to_int[c] for c in self.data[idx + 1 : idx + self.seq_length + 1]], dtype=torch.long)
        return inputs, targets

# --- Helper Functions ---
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, val_loader, criterion, device):
    """A simple evaluation loop."""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            total_loss += loss.item()
    return total_loss / len(val_loader)

def train_and_evaluate(model, model_name, train_loader, val_loader, device):
    """A simplified training and evaluation function."""
    print(f"\n--- Training {model_name} ---")
    print(f"Model params: {count_parameters(model)/1e6:.3f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss() # Although loss is calculated in model, useful for reference
    best_val_loss = float('inf')
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=False)
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids=inputs, labels=targets)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    total_time = time.time() - start_time
    print(f"--- Finished training {model_name} in {total_time:.2f}s ---")
    return best_val_loss, total_time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Architecture Comparison on Dummy Data---")
    print(f"Using device: {device}\n")

    # --- Data Prep ---
    chars = sorted(list(set(DUMMY_TEXT)))
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    
    split_idx = int(len(DUMMY_TEXT) * (1 - VALIDATION_SPLIT))
    train_data, val_data = DUMMY_TEXT[:split_idx], DUMMY_TEXT[split_idx:]

    train_dataset = TextDataset(train_data, SEQ_LENGTH, char_to_int)
    val_dataset = TextDataset(val_data, SEQ_LENGTH, char_to_int)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    print(f"Vocab size: {VOCAB_SIZE}, Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    # --- Initialize Models ---
    print("\nInitializing models...")
    lnn_model = LNNModel(LNN_MODEL_CONFIG).to(device)
    lfm_model = LFMForCausalLM(LFMConfig()).to(device)

    # --- Run Experiments ---
    results = {}
    if not SKIP_LNN_TRAINING:
        lnn_loss, lnn_time = train_and_evaluate(lnn_model, "Quasar LNN", train_loader, val_loader, device)
        results["Quasar LNN"] = {"Validation Loss": lnn_loss, "Training Time (s)": lnn_time, "Params (M)": count_parameters(lnn_model)/1e6}
    else:
        print("\n--- Skipping LNN training as requested ---")
    
    lfm_loss, lfm_time = train_and_evaluate(lfm_model, "Liquid Transformer", train_loader, val_loader, device)
    results["Liquid Transformer"] = {"Validation Loss": lfm_loss, "Training Time (s)": lfm_time, "Params (M)": count_parameters(lfm_model)/1e6}

    # --- Final Report ---
    print("\n\n--- Comparison Complete: Final Report ---")
    report_df = pd.DataFrame.from_dict(results, orient="index")
    print(report_df.to_string(formatters={'Params (M)': '{:,.3f}'.format, 'Validation Loss': '{:,.4f}'.format, 'Training Time (s)': '{:,.2f}'.format}))
    
    winner = report_df["Validation Loss"].idxmin()
    print(f"\n--- Winner: {winner} (based on lowest validation loss) ---")

if __name__ == "__main__":
    main()
