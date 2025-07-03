import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from datasets import load_dataset
from itertools import cycle
import math
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
from huggingface_hub import HfApi, login

from quasar.lnn import LNNModel, LNNConfig

# Helper to count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def evaluate(model, dataloader, loss_fn, device, vocab_size):
    """Evaluates the LNN model on a given dataset."""
    model.eval()
    total_loss = 0
    total_correct = 0
    total_tokens = 0
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x).logits
            
            loss = loss_fn(logits.view(-1, vocab_size), batch_y.view(-1))
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=-1)
            total_correct += (preds == batch_y).sum().item()
            total_tokens += batch_y.numel()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_tokens
    return avg_loss, accuracy

def main(args):
    print("--- Training Quasar LNN on WikiText-2 ---")

    # --- Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Hugging Face Dataset Loader ---
    print("Loading dataset 'wikitext', 'wikitext-2-raw-v1' from Hugging Face...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    
    # Concatenate all text samples into one large string
    train_text = " ".join([example['text'] for example in dataset['train'] if example['text'].strip()])
    val_text = " ".join([example['text'] for example in dataset['validation'] if example['text'].strip()])
    
    print(f"Building vocabulary from training data...")
    words = train_text.split()
    word_counts = Counter(words)
    vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    vocab = vocab[:499] # Limit vocab size to 499 + <unk> = 500
    vocab.insert(0, '<unk>') # Add an unknown token
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    
    word_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_word = {i: word for i, word in enumerate(vocab)}

    def text_to_ids(text, word_to_int_map):
        return [word_to_int_map.get(word, word_to_int_map['<unk>']) for word in text.split()]

    train_ids = text_to_ids(train_text, word_to_int)
    val_ids = text_to_ids(val_text, word_to_int)
    
    def create_sequences(ids, seq_length):
        num_sequences = (len(ids) - 1) // seq_length
        inputs = torch.zeros((num_sequences, seq_length), dtype=torch.long)
        targets = torch.zeros((num_sequences, seq_length), dtype=torch.long)
        for i in range(num_sequences):
            start = i * seq_length
            end = start + seq_length
            inputs[i] = torch.tensor(ids[start:end], dtype=torch.long)
            targets[i] = torch.tensor(ids[start+1:end+1], dtype=torch.long)
        return TensorDataset(inputs, targets)

    train_dataset = create_sequences(train_ids, args.sequence_length)
    val_dataset = create_sequences(val_ids, args.sequence_length)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size)

    # --- Model Initialization ---
    lnn_config = LNNConfig(
        vocab_size=vocab_size,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.lnn_num_layers,
    )
    quasar_lnn = LNNModel(lnn_config).to(device)

    print(f"Quasar LNN Initialized. Trainable Parameters: {count_parameters(quasar_lnn)}")

    print("\n--- Model Parameters ---")
    for name, param in quasar_lnn.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name} | Size: {param.size()} | Num Parameters: {param.numel()}")
    print("------------------------\n")

    # --- Training Setup ---
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f"\nStarting training on {device} for {args.num_epochs} epochs...")

    # --- Train Quasar LNN ---
    print("\n--- Training Quasar LNN ---")
    optimizer_lnn = torch.optim.Adam(quasar_lnn.parameters(), lr=args.lr_lnn)
    for epoch in range(args.num_epochs):
        quasar_lnn.train()
        pbar_lnn = tqdm(train_dataloader, desc=f"LNN Epoch {epoch+1}/{args.num_epochs}")
        for batch_x, batch_y in pbar_lnn:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer_lnn.zero_grad()
            logits = quasar_lnn(batch_x).logits
            loss = loss_fn(logits.view(-1, vocab_size), batch_y.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(quasar_lnn.parameters(), 1.0)
            optimizer_lnn.step()
            pbar_lnn.set_postfix({
                "LNN Loss": f"{loss.item():.4f}",
                "Lambda": f"{F.softplus(quasar_lnn.lambda_param).item():.4f}"
            })
        
        val_loss, val_acc = evaluate(quasar_lnn, val_dataloader, loss_fn, device, vocab_size)
        print(f"LNN Epoch {epoch+1} - Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

    # --- Final Evaluation & Upload ---
    print("\n--- Training Finished ---")
    if args.push_to_hub:
        print(f"Uploading model to Hugging Face Hub repository: {args.repo_id}")
        try:
            # Use the token to save and upload the model
            quasar_lnn.save_pretrained(
                "./trained_lnn",
                push_to_hub=True,
                repo_id=args.repo_id,
                token=args.hf_token,
                safe_serialization=True
            )
            print("Model uploaded successfully!")
        except Exception as e:
            print(f"An error occurred during upload: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Quasar LNN on WikiText-2")
    # Shared args
    parser.add_argument("--sequence_length", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=16)
    # LNN args
    parser.add_argument("--hidden_size", type=int, default=32) # Reduced for lower params
    parser.add_argument("--lnn_num_layers", type=int, default=2)   # Reduced for lower params
    parser.add_argument("--num_epochs", type=int, default=2)   # More epochs for real data
    parser.add_argument("--lr_lnn", type=float, default=1e-3)
    # Upload args
    parser.add_argument("--push_to_hub", action="store_true", help="Push the trained LNN model to the Hugging Face Hub.")
    parser.add_argument("--repo_id", type=str, default="eyad-silx/si-1", help="The repository ID on the Hugging Face Hub.")
    parser.add_argument("--hf_token", type=str, default="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", help="Your Hugging Face API token.")
    
    args = parser.parse_args()
    main(args)
