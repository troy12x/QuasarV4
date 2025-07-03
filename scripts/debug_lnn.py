# scripts/debug_lnn.py
# A script to compare the language modeling capabilities of a small LNN vs. a standard Transformer.

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_dataset

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
def generate_text(model, char_to_int, int_to_char, device, prompt="The quick brown fox", max_length=50):
    """Generates text from a trained model using greedy decoding."""
    model.to(device)
    model.eval()

    print(f"\n--- Generating text with {model.__class__.__name__} ---")
    print(f"Prompt: '{prompt}'")

    # Filter out characters in the prompt that are not in the vocabulary
    prompt_chars = [c for c in prompt if c in char_to_int]
    if not prompt_chars:
        print("Prompt contains no known characters. Cannot generate text.")
        return prompt
        
    input_ids = torch.tensor([char_to_int[c] for c in prompt_chars], dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            if isinstance(model, LNNModel):
                outputs = model(input_ids=input_ids)
                logits = outputs.logits
            else: # TransformerModel
                logits = model(input_ids)

            next_token_logits = logits[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()
            
            if next_token_id not in int_to_char:
                break # Stop if we generate an unknown token

            next_token_tensor = torch.tensor([[next_token_id]], device=device)
            input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    generated_text = "".join([int_to_char.get(i, '') for i in input_ids.squeeze(0).tolist()])
    print(f"Generated Text: '{generated_text}'")
    return generated_text

# --- Main Comparison Logic ---
def main():
    """Sets up and runs the comparison between the LNN and Transformer models."""
    # 1. Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Real Dataset
    print("Loading dataset 'Gaoj124/llama3_4_of_4_ours_0.7_0.05_sentences' from Hugging Face...")
    dataset_hf = load_dataset("Gaoj124/llama3_4_of_4_ours_0.7_0.05_sentences", split='train', trust_remote_code=True)
    
    text_data = " ".join([example['text'] for example in dataset_hf])
    print(f"Dataset loaded. Total characters: {len(text_data)}")
    
    # 3. Create Character-Level Vocabulary
    chars = sorted(list(set(text_data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size}")
    char_to_int = {ch: i for i, ch in enumerate(chars)}
    int_to_char = {i: ch for i, ch in enumerate(chars)}

    # 4. Prepare Data
    seq_length = 248
    batch_size = 16
    data_ids = [char_to_int[c] for c in text_data]
    num_sequences = (len(data_ids) - 1) // seq_length

    if num_sequences == 0:
        print("Error: Not enough data to create a single sequence. Exiting.")
        return

    input_tensors = torch.zeros((num_sequences, seq_length), dtype=torch.long)
    target_tensors = torch.zeros((num_sequences, seq_length), dtype=torch.long)
    for i in range(num_sequences):
        start = i * seq_length
        end = start + seq_length
        input_tensors[i] = torch.tensor(data_ids[start:end], dtype=torch.long)
        target_tensors[i] = torch.tensor(data_ids[start+1:end+1], dtype=torch.long)

    dataset = TensorDataset(input_tensors, target_tensors)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 5. Define Model Hyperparameters
    hidden_size = 128
    num_layers = 2
    
    # 6. Instantiate Models
    # LNN Model
    lnn_config = LNNConfig(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_hidden_layers=num_layers,
    )
    lnn_model = LNNModel(lnn_config)

    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=hidden_size,
        nhead=4, # Number of attention heads, must divide hidden_size
        hidden_dim=hidden_size * 4, # Feedforward dimension
        nlayers=num_layers
    )

    # 7. Print Parameter Counts
    lnn_params = count_parameters(lnn_model)
    transformer_params = count_parameters(transformer_model)
    print("\n--- Model Parameter Count ---")
    print(f"LNN Model Parameters:         {lnn_params:,}")
    print(f"Transformer Model Parameters:   {transformer_params:,}")

    # 8. Train Models (with fewer epochs for debugging)
    lnn_final_loss = train_model(lnn_model, dataloader, device, epochs=20)
    transformer_final_loss = train_model(transformer_model, dataloader, device, epochs=20)

    # 9. Generate and Compare Text
    prompt = "Liam Thompson was born on"
    generate_text(lnn_model, char_to_int, int_to_char, device, prompt=prompt, max_length=150)
    generate_text(transformer_model, char_to_int, int_to_char, device, prompt=prompt, max_length=150)

    print("\n--- Comparison Complete ---")
    print(f"LNN Final Loss:          {lnn_final_loss:.4f}")
    print(f"Transformer Final Loss:    {transformer_final_loss:.4f}")

if __name__ == "__main__":
    main()
