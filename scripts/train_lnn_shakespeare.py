import os
import sys
import mmap
import torch
import numpy as np
import torch.nn.functional as F

# Add the project root to the Python path to allow importing from the 'quasar' package.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from quasar.lnn import LNNModel, LNNConfig

# --- Configuration ---
# Dataloader config
batch_size = 8
block_size = 256 # sequence length

# Model config (using a smaller configuration for this example)
# The default LNNConfig is for a very large model (440B+)
n_layer = 4
n_embd = 256
use_moe = False # Keep it simple for now
use_pmb = False # Keep it simple for now

# Training config
learning_rate = 3e-4
max_iters = 2000
eval_interval = 200
eval_iters = 100
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"Using device: {device}")
torch.manual_seed(1337)

# --- Data Loading ---
def get_data(split):
    """Memory-map the data file for efficient access."""
    data_path = os.path.join(project_root, f'{split}.bin')
    if not os.path.exists(data_path):
        return None
    # Data is stored as uint16 in train.bin/val.bin
    data = np.memmap(data_path, dtype=np.uint16, mode='r')
    return data

print("Loading data...")
train_data = get_data('train')
val_data = get_data('val')

if train_data is None:
    raise FileNotFoundError(f"Training data not found at 'c:\\quasarv4\\train.bin'. Please ensure this file exists.")
if val_data is None:
    print("Validation data 'val.bin' not found. Will only report training loss.")

# Determine vocab size from the data itself
vocab_size = int(np.max(train_data)) + 1
print(f"Determined vocab size from data: {vocab_size}")

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # Generate random starting points for batches
    ix = torch.randint(len(data) - block_size, (batch_size,))
    # Create input sequences (x)
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    # Create target sequences (y), which are shifted by one
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Model and Optimizer ---
print("Initializing model...")
model_config = LNNConfig(
    vocab_size=vocab_size,
    hidden_size=n_embd,
    num_hidden_layers=n_layer,
    use_pmb=use_pmb,
    use_moe=use_moe,
)

model = LNNModel(model_config)
model.to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# --- Loss Estimation ---
@torch.no_grad()
def estimate_loss():
    """Calculates the average loss over a number of iterations for train and val splits."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if (split == 'val' and val_data is None):
            continue # Skip validation if val.bin doesn't exist
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            output = model(X)
            logits = output.logits
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1))
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- Main Training Loop ---

print("Starting training...")
for iter in range(max_iters):
    # Evaluate loss periodically
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        val_loss_str = f", val loss {losses['val']:.4f}" if 'val' in losses else ""
        print(f"step {iter}: train loss {losses['train']:.4f}{val_loss_str}")

    # Get a batch of data
    xb, yb = get_batch('train')

    # Forward pass and loss calculation
    output = model(xb)
    logits = output.logits
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), yb.view(-1))
    
    # Backward pass and optimization
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print("Training finished.")

# --- Save Model ---
model_save_path = os.path.join(project_root, 'lnn_shakespeare.pth')
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
