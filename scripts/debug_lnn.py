# scripts/debug_lnn.py
# A simple script to isolate and test the LNNModel on a single GPU.

import torch
import os
import sys

# Add project root to the Python path to allow importing the 'quasar' module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from quasar.lnn import LNNModel, LNNConfig

def main():
    """Initializes a tiny LNN model and runs a single forward pass to check the loss."""
    print("--- Starting LNN Model Debug Script ---")

    # 1. Define a minimal model configuration
    # Using a small vocab size to get a predictable initial loss.
    # Expected random loss should be approx. ln(vocab_size) = ln(1000) ~= 6.9
    pad_token_id = 0
    config = LNNConfig(
        vocab_size=1000,
        hidden_size=32,
        num_hidden_layers=2,
        pad_token_id=pad_token_id,
        use_moe=False # Keep it simple, no MoE for this test
    )
    print(f"Created a tiny LNNConfig: {config}")

    # 2. Instantiate the model
    model = LNNModel(config)
    print("LNNModel instantiated successfully.")

    # 3. Set up device (use GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    # 4. Create a batch of dummy data
    batch_size = 4
    seq_length = 64
    input_ids = torch.randint(1, config.vocab_size, (batch_size, seq_length), device=device)
    # Set some tokens to the pad_token_id to test the ignore_index logic
    input_ids[:, -10:] = pad_token_id

    # Labels are the same as input_ids for Causal LM
    labels = input_ids.clone()

    print(f"Created dummy data with shape: {input_ids.shape}")

    # 5. Run a single forward pass
    print("--- Performing a single forward pass ---")
    try:
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        print(f"\nSUCCESS: Forward pass completed.")
        print(f"--> Initial Loss: {loss.item()}")
        print(f"--> Expected Loss (approx.): {torch.log(torch.tensor(config.vocab_size, dtype=torch.float)).item()}")

        if loss.item() > 10:
            print("\nWARNING: Loss is still unexpectedly high. The issue is likely within the LNNModel's core logic.")
        else:
            print("\nSUCCESS: Loss is in a reasonable range. The issue may be related to the FSDP training framework or data loading.")

    except Exception as e:
        print(f"\nERROR: The forward pass failed with an exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
