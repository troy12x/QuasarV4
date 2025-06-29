import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time
import math
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.lnn import LNNModel, LNNConfig
from quasar.transformer_model import TransformerModel

# --- Training Function (from debug_lnn.py) ---
def train_model(model, dataloader, device, epochs=3): # Fewer epochs for this test
    """A simple training loop for a given model."""
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"--- Training {model.__class__.__name__} ---")
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            input_ids, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            
            # Handle different model forward signatures
            if isinstance(model, LNNModel):
                outputs = model(input_ids=input_ids)
            else:
                outputs = model(input_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/{epochs}, Avg Loss: {avg_loss:.4f}")
    return avg_loss

# --- Evaluation Function ---
def evaluate_model(model, dataloader, device):
    """Evaluates a model's perplexity and inference speed."""
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum') # Sum losses for perplexity calc
    total_loss = 0
    total_tokens = 0
    
    start_time = time.time()
    with torch.no_grad():
        for batch in dataloader:
            input_ids, labels = [t.to(device) for t in batch]
            
            # Handle different model forward signatures
            if isinstance(model, LNNModel):
                outputs = model(input_ids=input_ids)
            else:
                outputs = model(input_ids)

            if hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            total_loss += loss.item()
            total_tokens += labels.numel()

    end_time = time.time()
    
    # Avoid division by zero if dataloader is empty
    if total_tokens == 0:
        return float('inf'), 0

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    inference_time = end_time - start_time
    
    return perplexity, inference_time

# --- Main Comparison Logic ---
def main():
    """Sets up and runs the long-context scaling comparison."""
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Get a long text corpus
    long_text = """
    Alice was beginning to get very tired of sitting by her sister on the
    bank, and of having nothing to do: once or twice she had peeped into the
    book her sister was reading, but it had no pictures or conversations in
    it, 'and what is the use of a book,' thought Alice 'without pictures or
    conversations?'
    """ * 50 # Repeat to make it longer and support longer sequences

    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = tokenizer.encode(long_text)
    vocab_size = tokenizer.vocab_size

    # 4. Define training and evaluation parameters
    TRAIN_SEQ_LEN = 256
    EVAL_SEQ_LENS = [128, 256, 512, 1024]
    hidden_size = 64
    num_layers = 2

    # 5. Create a single training dataset
    print(f"--- Creating training dataset with sequence length {TRAIN_SEQ_LEN} ---")
    train_inputs, train_labels = [], []
    for i in range(0, len(tokens) - TRAIN_SEQ_LEN, TRAIN_SEQ_LEN // 2):
        train_inputs.append(tokens[i:i+TRAIN_SEQ_LEN])
        train_labels.append(tokens[i+1:i+TRAIN_SEQ_LEN+1])
    
    train_dataset = TensorDataset(torch.tensor(train_inputs), torch.tensor(train_labels))
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

    # 6. Instantiate and Train Models
    lnn_config = LNNConfig(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_layers)
    lnn_model = LNNModel(lnn_config)
    print(f"\nTraining LNN with {sum(p.numel() for p in lnn_model.parameters()):,} parameters...")
    train_model(lnn_model, train_loader, device)

    transformer_model = TransformerModel(
        vocab_size=vocab_size, embedding_dim=hidden_size, nhead=4,
        hidden_dim=hidden_size * 4, nlayers=num_layers
    )
    print(f"\nTraining Transformer with {sum(p.numel() for p in transformer_model.parameters()):,} parameters...")
    train_model(transformer_model, train_loader, device)

    # 7. Run scaling evaluation
    results = {"lnn": {}, "transformer": {}}
    print("\n--- Starting Inference Time Scaling Evaluation ---")

    for seq_len in EVAL_SEQ_LENS:
        print(f"\n--- Evaluating at Sequence Length: {seq_len} ---")
        
        # Create test dataloader for this specific sequence length
        test_inputs, test_labels = [], []
        for i in range(0, len(tokens) - seq_len, seq_len):
            test_inputs.append(tokens[i:i+seq_len])
            test_labels.append(tokens[i+1:i+seq_len+1])
        
        if not test_inputs:
            print(f"Corpus not long enough for sequence length {seq_len}. Skipping.")
            continue

        test_dataset = TensorDataset(torch.tensor(test_inputs), torch.tensor(test_labels))
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

        # Evaluate LNN
        print("Evaluating LNN...")
        _, lnn_time = evaluate_model(lnn_model, test_loader, device)
        results["lnn"][seq_len] = lnn_time

        # Evaluate Transformer
        print("Evaluating Transformer...")
        _, transformer_time = evaluate_model(transformer_model, test_loader, device)
        results["transformer"][seq_len] = transformer_time

    # 8. Print summary table
    print("\n--- Inference Time Scaling Results ---")
    print(f"{'Seq Length':<12} | {'LNN Time (s)':<15} | {'Transformer Time (s)':<20}")
    print(f"{'':-<13}+{'-':-<17}+{'':-<20}")
    for seq_len in EVAL_SEQ_LENS:
        lnn_t = results["lnn"].get(seq_len)
        trans_t = results["transformer"].get(seq_len)
        lnn_t_str = f"{lnn_t:.4f}" if lnn_t is not None else 'N/A'
        trans_t_str = f"{trans_t:.4f}" if trans_t is not None else 'N/A'
        print(f"{seq_len:<12} | {lnn_t_str:<15} | {trans_t_str:<20}")

if __name__ == "__main__":
    main()
