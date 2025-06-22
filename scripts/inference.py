# c:\quasarv4\scripts\inference.py

import torch
import os
import sys
import argparse

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.model import Quasar
from quasar.utils import SimpleTokenizer
from scripts.train import EMBEDDING_DIM, HIDDEN_DIM, MODEL_SAVE_PATH, dummy_corpus

def generate_next_token(model, tokenizer, prompt, device):
    """Generates the single next token for a given prompt."""
    model.eval()
    with torch.no_grad():
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long).to(device)
        logits = model(input_ids, memory_query=True)
        
        # Get the token with the highest probability
        next_token_id = torch.argmax(logits, dim=-1).item()
        return tokenizer.decode([next_token_id])

def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Quasar model.")
    parser.add_argument("--prompt", type=str, required=True, help="The prompt to generate text from.")
    parser.add_argument("--model_path", type=str, default=MODEL_SAVE_PATH, help="Path to the trained model file.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Tokenizer and Model
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at '{args.model_path}'.")
        print("Please run the training script first: python scripts/train.py")
        return

    tokenizer = SimpleTokenizer(dummy_corpus)
    vocab_size = len(tokenizer)

    model = Quasar(
        vocab_size=vocab_size,
        embedding_dim=EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print(f"Model loaded from {args.model_path}")

    # 2. Ingest a document into the model's memory (for demonstration)
    # In a real application, you would ingest your entire knowledge base here.
    document_to_ingest = "The Quasar model, developed in Silicon Valley, uses Liquid Neural Networks and a Parameter Memory Bank to achieve state-of-the-art results without attention."
    model.ingest(document_to_ingest)

    # 3. Generate a response
    print(f"\nPrompt: '{args.prompt}'")
    next_token = generate_next_token(model, tokenizer, args.prompt, device)
    print(f"Generated next token: '{next_token}'")
    print(f"Full response: '{args.prompt} {next_token}'")

if __name__ == '__main__':
    main()
