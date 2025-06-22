# c:\quasarv4\scripts\train.py

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.model import Quasar
from quasar.utils import SimpleTokenizer, prepare_batch

# --- Configuration ---
VOCAB_SIZE = 1000 # Placeholder, will be updated by tokenizer
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
BATCH_SIZE = 4
MODEL_SAVE_PATH = "quasar_model.pth"

# --- Dummy Data ---
# In a real scenario, you would load a large text corpus from a file.
dummy_corpus = [
    "The Quasar model is a new architecture for language understanding.",
    "It uses a Liquid Neural Network instead of a Transformer.",
    "Memory is handled by a Parameter Memory Bank.",
    "This allows for potentially infinite context length.",
    "Text is segmented using a semantic chunker.",
    "The goal is to achieve high performance without attention mechanisms.",
    "Training is done using a standard next-token prediction loss.",
    "This script demonstrates a basic training loop for the Quasar model."
]

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Initialize Tokenizer and Model
    tokenizer = SimpleTokenizer(dummy_corpus)
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")

    model = Quasar(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.vocab['<pad>'])

    # 2. Training Loop
    print("\nStarting training...")
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        # Simple batching
        for i in range(0, len(dummy_corpus), BATCH_SIZE):
            batch_texts = dummy_corpus[i:i+BATCH_SIZE]
            if not batch_texts: continue

            # Prepare batch
            batch_tensor = prepare_batch(batch_texts, tokenizer, device)
            inputs = batch_tensor[:, :-1] # All but the last token
            targets = batch_tensor[:, 1:]  # All but the first token (shifted)

            # Forward pass
            optimizer.zero_grad()
            # We get one logit vector per sequence in the batch
            logits = model(inputs, memory_query=False) # (batch_size, vocab_size)
            
            # For training, we need to predict the next token at each step.
            # Our current model architecture only gives one output logit at the end.
            # This is a simplification. For a real causal LM, the model would output
            # logits for every token in the sequence.
            # We will adapt the loss calculation to this simplified structure
            # by only predicting the token that follows the input sequence.
            last_token_target = targets[:, -1]

            loss = criterion(logits, last_token_target)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_loss = total_loss / (len(dummy_corpus) // BATCH_SIZE)
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # 3. Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}")

if __name__ == '__main__':
    main()
