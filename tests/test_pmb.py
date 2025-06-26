# c:\quasarv4\tests\test_pmb.py

import unittest
import torch
import sys
import os
import numpy as np
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from quasar.lnn import LNNModel, LNNConfig
from transformers import AutoTokenizer

# --- Data Loader ---
def get_batch(data, sequence_length, batch_size):
    """Generates a small batch of data of inputs x and targets y."""
    ix = torch.randint(len(data) - sequence_length, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+sequence_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+sequence_length]).astype(np.int64)) for i in ix])
    return x, y

class TestShakespeareTraining(unittest.TestCase):

    def test_model_learns_on_shakespeare(self):
        """
        Tests if the model can learn from the Tiny Shakespeare dataset.
        Checks if validation loss decreases after a short training run.
        """
        print("\n--- Testing Model Training on Tiny Shakespeare ---")

        # --- 1. Hyperparameters ---
        batch_size = 16
        sequence_length = 50
        learning_rate = 5e-5 # Reduced LR to prevent loss explosion
        max_iters = 200
        eval_interval = 50
        eval_iters = 20

        # --- 2. Setup Model and Tokenizer ---
        # The prepare.py script uses gpt2 encoding, so we MUST use the gpt2 tokenizer.
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        config = LNNConfig(
            vocab_size=tokenizer.vocab_size,
            hidden_size=128, # A bit larger for a real dataset
            num_hidden_layers=4,
            use_pmb=True,
            pmb_num_blocks=4,
            pmb_slots_per_block=16,
        )
        model = LNNModel(config)
        optimizer = AdamW(model.parameters(), lr=learning_rate)

        # --- 3. Load Data ---
        train_data = np.memmap(os.path.join(os.path.dirname(__file__), '..', 'train.bin'), dtype=np.uint16, mode='r')
        val_data = np.memmap(os.path.join(os.path.dirname(__file__), '..', 'val.bin'), dtype=np.uint16, mode='r')

        # --- 4. Training & Evaluation Loop ---
        print("\n--- Starting Training Loop ---")
        model.train()
        initial_val_loss = 0
        final_val_loss = 0
        current_train_loss = float('nan') # Use NaN for the first step

        for iter_num in range(max_iters):
            # Evaluate validation loss at the beginning and periodically
            if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
                model.eval()
                losses = torch.zeros(eval_iters)
                for k in range(eval_iters):
                    X, Y = get_batch(val_data, sequence_length, batch_size)
                    with torch.no_grad():
                        outputs = model(X)
                        logits = outputs.logits
                        loss_fct = CrossEntropyLoss()
                        loss = loss_fct(logits.view(-1, logits.size(-1)), Y.view(-1))
                        losses[k] = loss.item()
                val_loss = losses.mean()
                if iter_num == 0:
                    initial_val_loss = val_loss
                final_val_loss = val_loss
                print(f"Step {iter_num}: Train Loss = {current_train_loss:.4f}, Validation Loss = {val_loss:.4f}")
                model.train()

            # Get a batch of training data and update model
            xb, yb = get_batch(train_data, sequence_length, batch_size)
            optimizer.zero_grad()
            outputs = model(xb)
            logits = outputs.logits
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), yb.view(-1))
            current_train_loss = loss.item()
            loss.backward()
            optimizer.step()

        print("--- Training Complete ---")
        print(f"Initial Validation Loss: {initial_val_loss:.4f}")
        print(f"Final Validation Loss:   {final_val_loss:.4f}")

        # --- 5. Generate Text ---
        print("\n--- Generating Sample Text ---")
        model.eval()
        # Manually implement greedy decoding since model.generate() doesn't exist
        context = torch.zeros((1, 1), dtype=torch.long) # Start with a single <BOS> token (ID 0)
        generated_ids = context
        with torch.no_grad():
            for _ in range(100): # max_new_tokens
                outputs = model(generated_ids)
                logits = outputs.logits
                # Get the logits for the last token in the sequence
                next_token_logits = logits[:, -1, :]
                # Greedily select the token with the highest probability
                next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
                # Append the new token to the sequence
                generated_ids = torch.cat((generated_ids, next_token_id), dim=1)
        
        print(tokenizer.decode(generated_ids[0].tolist()))

        self.assertLess(final_val_loss, initial_val_loss, "Validation loss did not decrease, model failed to learn.")
        print("\n✅ Shakespeare Training Test Passed: Validation loss decreased.")

if __name__ == '__main__':
    unittest.main()
