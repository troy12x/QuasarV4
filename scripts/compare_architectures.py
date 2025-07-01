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
from torch.utils.data import Dataset, DataLoader
import sys
import os
import math
from tqdm import tqdm

# Add project root to path to allow importing quasar
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quasar.lnn import LNNModel, LNNConfig

# --- Experiment Configuration ---
FILE_PATH = 'c:\\quasarv4\\input.txt'
DATA_SUBSET_SIZE = 90000  # Use a consistent subset for a fair, quick comparison
SEQ_LENGTH = 128
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2 # Train for a few epochs to see learning trends
VALIDATION_SPLIT = 0.1
SKIP_LNN_TRAINING = False # Set to True to skip LNN training for faster debugging

# --- Model Definitions ---
# We will aim for a similar parameter count for a fair comparison.
# We will aim for a much smaller parameter count (~1-2M) that is suitable for a consumer GPU.
LNN_MODEL_CONFIG = LNNConfig(
    vocab_size=50257, # This will be updated by the tokenizer vocab size later
    hidden_size=256,
    num_hidden_layers=2,
    chunk_size=64,
    use_moe=True,
    use_pmb=True
)

TRANSFORMER_MODEL_CONFIG = {
    "name": "Transformer Model",
    "embed_size": 256,        # Reduced
    "num_layers": 2,          # Reduced
    "num_heads": 4,           # Reduced
    "forward_expansion": 2    # Reduced
}

# --- A Standard Transformer Model for Comparison ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_layers, num_heads, forward_expansion, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, num_heads, embed_size * forward_expansion, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.config = lambda: None # for compatibility with evaluate function
        self.config.vocab_size = vocab_size

    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.fc_out(output)
        return output, None # Return None for hidden state to match LNN's output signature

# --- Generic Training & Evaluation Code ---
class TextDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        return torch.tensor(chunk[:-1], dtype=torch.long), torch.tensor(chunk[1:], dtype=torch.long)

def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            # Use autocast for evaluation as well for consistency
            with torch.cuda.amp.autocast():
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, model.config.vocab_size), targets.view(-1))
            total_loss += loss.item()
    return total_loss / len(dataloader)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_and_evaluate(model, train_loader, val_loader, device, model_name):
    print(f"\n--- Training {model_name} ---")
    print(f"Model params: {count_parameters(model)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    # GradScaler for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        model.train()
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            # Forward pass with autocasting
            with torch.cuda.amp.autocast():
                logits, _ = model(inputs)
                loss = criterion(logits.view(-1, model.config.vocab_size), targets.view(-1))

            # Backward pass with scaling
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1} | Validation Loss: {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss

    return best_val_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Architecture Comparison: LNN vs. Transformer ---")
    print(f"Using device: {device}\n")

    with open(FILE_PATH, 'r', encoding='utf-8') as f:
        text = f.read()[:DATA_SUBSET_SIZE]
    print(f"Using a subset of the data: first {len(text)} characters.")

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    data = [char_to_int[ch] for ch in text]

    split_idx = int(len(data) * (1 - VALIDATION_SPLIT))
    train_data, val_data = data[:split_idx], data[split_idx:]

    train_dataset = TextDataset(train_data, SEQ_LENGTH)
    val_dataset = TextDataset(val_data, SEQ_LENGTH)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    print(f"Vocab size: {vocab_size}, Train sequences: {len(train_dataset)}, Val sequences: {len(val_dataset)}")

    # --- Initialize and Run Models ---
    print("Initializing models...")

    # LNN Model Run
    lnn_loss = float('inf')
    if not SKIP_LNN_TRAINING:
        LNN_MODEL_CONFIG.vocab_size = vocab_size
        lnn_model = LNNModel(LNN_MODEL_CONFIG).to(device)
        lnn_loss = train_and_evaluate(lnn_model, train_loader, val_loader, device, "LNN Model")
    else:
        print("\n--- Skipping LNN training as requested ---")

    # Transformer Model Run
    transformer_params = {k: v for k, v in TRANSFORMER_MODEL_CONFIG.items() if k != 'name'}
    transformer_model = TransformerModel(vocab_size=vocab_size, **transformer_params).to(device)
    transformer_loss = train_and_evaluate(transformer_model, train_loader, val_loader, device, TRANSFORMER_MODEL_CONFIG['name'])

    print("\n--- Experiment Complete: Final Report ---")
    if not SKIP_LNN_TRAINING:
        print(f"LNN Model Final Validation Loss: {lnn_loss:.4f}")
    print(f"Transformer Model Final Validation Loss: {transformer_loss:.4f}")

    print("\n--- Conclusion ---")
    if not SKIP_LNN_TRAINING:
        if lnn_loss < transformer_loss:
            print("The LNN model achieved a lower validation loss, supporting the hypothesis that it is more parameter-efficient.")
        elif transformer_loss < lnn_loss:
            print("The Transformer model achieved a lower validation loss in this experiment.")
        else:
            print("The models performed almost identically.")

if __name__ == "__main__":
    main()
