import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import sys
import os

# Add project root to sys.path to allow for local package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the custom model architecture
from quasar.lnn import LNNModel, LNNConfig

# --- Explicitly register the custom model with transformers ---
# This is the robust way to ensure the library knows about our custom architecture.
AutoConfig.register("quasar", LNNConfig)
AutoModelForCausalLM.register(LNNConfig, LNNModel)

def generate_text(model_id, prompt):
    """
    Loads a model from the Hugging Face Hub and generates text based on a prompt.
    """
    print(f"Loading model: {model_id}")
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        # Load tokenizer and model, allowing custom code
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

        # Some models might not have a pad_token set, which can cause issues with generation.
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        # --- Manual Generation Loop ---
        num_tokens_to_generate = 10
        print(f"\n--- Manually generating next {num_tokens_to_generate} tokens ---")
        print(f"Prompt: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        generated_ids = input_ids.clone() # Use a separate tensor for the full sequence
        hidden_states = None # Start with no hidden state

        print("\nGenerated tokens:")
        with torch.no_grad():
            # First, process the prompt to get the initial hidden state
            outputs = model(input_ids=input_ids, hidden_states=hidden_states)
            logits = outputs.logits
            # The model returns a tuple, but expects a list for the next input
            hidden_states = list(outputs.hidden_states) 

            # --- Stabilized Top-k Sampling ---
            next_token_logits = logits[:, -1, :]

            # 1. Sanitize model output: Replace any NaN/inf values from the model itself.
            if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                print("  [Warning] Model produced NaN/inf logits. Sanitizing output.")
                next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

            # 2. Forbid specific tokens (e.g., BOS token)
            next_token_logits[:, 0] = -float('inf')

            # 3. Apply Top-k filtering
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)

            # 4. Sample from the filtered distribution
            probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
            sampled_indices = torch.multinomial(probabilities, num_samples=1)
            next_token_id = top_k_indices.gather(-1, sampled_indices)

            # Generate tokens one by one
            for i in range(num_tokens_to_generate):
                # Decode and print the newly generated token
                token_text = tokenizer.decode(next_token_id.item())
                print(f"  Token {i+1}: '{token_text}' (ID: {next_token_id.item()})")

                # Append the new token for the final output
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                # Check for EOS token
                if next_token_id.item() == tokenizer.eos_token_id:
                    print("  (End of sequence token generated, stopping.)")
                    break

                # The input for the next step is just the token we just generated
                # We pass the hidden state from the previous step
                outputs = model(input_ids=next_token_id, hidden_states=hidden_states)
                logits = outputs.logits
                hidden_states = list(outputs.hidden_states) # Update the hidden state

                # --- Stabilized Top-k Sampling ---
                next_token_logits = logits[:, -1, :]

                # 1. Sanitize model output
                if torch.isnan(next_token_logits).any() or torch.isinf(next_token_logits).any():
                    print("  [Warning] Model produced NaN/inf logits. Sanitizing output.")
                    next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e9, posinf=-1e9, neginf=-1e9)

                # 2. Forbid specific tokens
                next_token_logits[:, 0] = -float('inf')

                # 3. Apply Top-k filtering
                top_k = 50
                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)

                # 4. Sample from the filtered distribution
                probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
                sampled_indices = torch.multinomial(probabilities, num_samples=1)
                next_token_id = top_k_indices.gather(-1, sampled_indices)

        # Decode the final full sequence
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        
        print(f"Generated text:\n{generated_text}")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure the model ID is correct and you have an active internet connection.")


if __name__ == "__main__":
    # Set the model to test
    MODEL_ID = r"pretrain_output/checkpoint-1000"
    
    # Interactively ask the user for a prompt
    while True:
        PROMPT = input("Enter your prompt: ")
        if PROMPT:
            break
        print("Prompt cannot be empty. Please try again.")

    generate_text(MODEL_ID, PROMPT)
