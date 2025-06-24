# c:\quasarv4\scripts\compare_models.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import LNN model and Transformer for comparison
from quasar.lnn import LNNModel, LNNConfig
from quasar.transformer_model import TransformerModel
from quasar.utils import SimpleTokenizer, prepare_batch

# --- Configuration ---
EMBEDDING_DIM = 128
HIDDEN_DIM = 256  # For Transformer's feed-forward layer
LEARNING_RATE = 0.001
NUM_EPOCHS = 15
BATCH_SIZE = 4

# Model-specific configs
NHEAD = 4       # For Transformer
NLAYERS = 2     # For both models
ACTIVATION = 'tanh' # For LNN
DT = 1.0        # For LNN

# --- Dummy Data ---
dummy_corpus = [
    "The LNN model is a new architecture for language understanding.",
    "It uses a Liquid Neural Network instead of a Transformer.",
    "This allows for potentially infinite context length.",
    "The goal is to achieve high performance without attention mechanisms.",
    "Training is done using a standard next-token prediction loss.",
    "This script demonstrates a basic training loop for the LNN model.",
    "A standard transformer uses self-attention to process sequences.",
    "Positional encodings are needed to understand token order.",
    "Transformers can have very large parameter counts.",
    "This comparison will test performance on a small, controlled task."
]

def train_model(model, tokenizer, device, model_name):
    """A generic function to train a model and report metrics."""
    print(f"\n--- Training {model_name} ---")
    model.to(device)
    model.train()  # Set model to training mode
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<pad>'])

    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        num_batches = 0
        for i in range(0, len(dummy_corpus), BATCH_SIZE):
            batch_texts = dummy_corpus[i:i + BATCH_SIZE]
            if not batch_texts: continue

            batch_tensor = prepare_batch(batch_texts, tokenizer, device)
            inputs = batch_tensor[:, :-1]
            targets = batch_tensor[:, 1:]

            optimizer.zero_grad()

            # Get logits from the model
            output = model(inputs)
            logits = output.logits if hasattr(output, 'logits') else output

            # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size)
            loss = criterion(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        if (epoch + 1) % 5 == 0:
            avg_loss = total_loss / num_batches
            print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    end_time = time.time()
    training_time = end_time - start_time
    final_loss = total_loss / num_batches

    print(f"Training complete for {model_name}.")
    print(f"Final Loss: {final_loss:.4f}")
    print(f"Total Training Time: {training_time:.2f} seconds")
    return training_time, final_loss

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Tokenizer
    tokenizer = SimpleTokenizer(dummy_corpus)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    # Initialize LNN Model
    lnn_config = LNNConfig(
        vocab_size=vocab_size,
        hidden_size=EMBEDDING_DIM,
        num_hidden_layers=NLAYERS,
        activation=ACTIVATION,
        dt=DT
    )
    lnn_model = LNNModel(lnn_config)

    # Initialize Transformer Model
    transformer_model = TransformerModel(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        nhead=NHEAD,
        hidden_dim=HIDDEN_DIM,
        nlayers=NLAYERS
    )

    # Train and Compare
    lnn_time, lnn_loss = train_model(lnn_model, tokenizer, device, "LNN")
    transformer_time, transformer_loss = train_model(transformer_model, tokenizer, device, "Transformer")

    # --- Final Report ---
    print("\n--- Comparison Report ---")
    print(f"                        | LNN         | Transformer")
    print(f"------------------------|-------------|-------------")
    print(f"Training Time (s)     | {lnn_time:<11.2f} | {transformer_time:<11.2f}")
    print(f"Final Loss              | {lnn_loss:<11.4f} | {transformer_loss:<11.4f}")
    print(f"------------------------|-------------|-------------")

if __name__ == '__main__':
    main()

