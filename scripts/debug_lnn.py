# scripts/debug_lnn.py
# A script to compare the language modeling capabilities of a small LNN vs. a standard Transformer.

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.lnn import LNNModel, LNNConfig
from quasar.transformer_model import TransformerModel

# --- Training Function ---
def train_model(model, dataloader, device, epochs=5, lr=1e-3):
    """A simple training loop for a given model."""
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    print(f"\n--- Training {model.__class__.__name__} ---")
    for epoch in range(epochs):
        total_loss = 0
        for i, batch in enumerate(dataloader):
            input_ids, labels = [t.to(device) for t in batch]

            optimizer.zero_grad()
            
            # Handle different model forward signatures
            outputs = model(input_ids=input_ids)

            # Unpack logits from model output
            if isinstance(outputs, tuple):
                logits = outputs[0]  # LNNModel returns (logits, hidden_states)
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits  # HuggingFace model output
            else:
                logits = outputs  # Simple tensor output
            
            loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 20 == 0: # Print progress every 20 batches
                print(f"  Epoch {epoch+1}/{epochs}, Batch {i+1}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} Final Average Loss: {avg_loss:.4f}")
    
    print(f"--- Final Loss for {model.__class__.__name__}: {avg_loss:.4f} ---")
    return avg_loss

# --- Utility Function ---
def count_parameters(model):
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Text Generation Function ---
def generate_text(model, tokenizer, device, prompt="The quick brown fox", max_length=20):
    """Generates text from a trained model using greedy decoding."""
    model.to(device)
    model.eval() # Set model to evaluation mode

    print(f"\n--- Generating text with {model.__class__.__name__} ---")
    print(f"Prompt: '{prompt}'")

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

    with torch.no_grad():
        for _ in range(max_length):
            # Get model outputs
            # Handle different model forward signatures
            if isinstance(model, LNNModel):
                outputs = model(input_ids=input_ids)
            else:
                outputs = model(input_ids)

            # Unpack logits from model output
            if isinstance(outputs, tuple):
                logits = outputs[0]  # LNNModel returns (logits, hidden_states)
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits  # HuggingFace model output
            else:
                logits = outputs  # Simple tensor output

            # Get the most likely token for the next position
            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

            # Append the predicted token to the input sequence
            input_ids = torch.cat([input_ids, next_token_id], dim=1)

            # Stop if the model generates an end-of-sequence token
            if next_token_id.item() == tokenizer.eos_token_id:
                break

    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    print(f"Generated Text: '{generated_text}'")
    return generated_text

# --- Main Comparison Logic ---
def main():
    """Sets up and runs the comparison between the LNN and Transformer models."""
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Create a simple, repetitive dummy dataset
    # The goal is for the models to learn a slightly more complex pattern.
    sentences = [
        "The quick brown fox jumps over the lazy dog. ",
        "A fast brown fox leaps over a sleeping dog. ",
        "The speedy fox vaults over the tired dog. ",
        "That quick fox jumped above the lazy dog. ",
        "The brown fox is quick and jumps high. ",
        "A lazy dog was underneath the jumping fox. "
    ]
    dummy_text = "".join(sentences * 20) # Repeat the varied sentences
    
    # 3. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. Prepare Data
    # We create overlapping sequences to give the model more training examples.
    seq_length = 32
    tokens = tokenizer.encode(dummy_text)
    
    inputs = []
    labels = []
    for i in range(len(tokens) - seq_length):
        inputs.append(tokens[i:i+seq_length])
        labels.append(tokens[i+1:i+seq_length+1])

    input_ids = torch.tensor(inputs)
    labels = torch.tensor(labels)

    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 5. Define Model Hyperparameters (using smaller values for faster debugging)
    vocab_size = tokenizer.vocab_size
    hidden_size = 32 # embedding_dim for Transformer
    num_layers = 1
    
    # 6. Instantiate Models
    # LNN Model
    lnn_config = LNNConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
    )
    lnn_model = LNNModel(lnn_config)

    # Standard Transformer Model
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=hidden_size,
        nhead=2, # Number of attention heads, must divide hidden_size
        hidden_dim=hidden_size * 4, # Feedforward dimension
        nlayers=num_layers
    )

    # 7. Print Parameter Counts
    lnn_params = count_parameters(lnn_model)
    transformer_params = count_parameters(transformer_model)
    print("\n--- Model Parameter Count ---")
    print(f"LNN Model Parameters:         {lnn_params:,}")
    print(f"Transformer Model Parameters:   {transformer_params:,}")

    # 8. Train Models
    lnn_final_loss = train_model(lnn_model, dataloader, device)
    transformer_final_loss = train_model(transformer_model, dataloader, device)

    # 9. Generate and Compare Text
    prompt = "The quick brown fox"
    generate_text(lnn_model, tokenizer, device, prompt=prompt, max_length=15)
    generate_text(transformer_model, tokenizer, device, prompt=prompt, max_length=15)

    print("\n--- Comparison Complete ---")
    print(f"LNN Final Loss:          {lnn_final_loss:.4f}")
    print(f"Transformer Final Loss:    {transformer_final_loss:.4f}")

    if lnn_final_loss < transformer_final_loss:
        print("\nResult: LNN model achieved a lower loss on this task, but check generated text for quality.")
    elif transformer_final_loss < lnn_final_loss:
        print("\nResult: Transformer model achieved a lower loss on this task.")
    else:
        print("\nResult: Both models achieved a similar loss.")

if __name__ == "__main__":
    main()
