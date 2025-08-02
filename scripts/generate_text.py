import torch
import sys
import os
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
 

# Import the custom model class
from quasar.lnn import LNNModel, LNNConfig

# Register your model config and class
AutoConfig.register("quasar", LNNConfig)
AutoModelForCausalLM.register(LNNConfig, LNNModel)

def generate_text(model_id, prompt):
    print(f"Loading model: {model_id}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id

        num_tokens_to_generate = 10
        print(f"\n--- Generating {num_tokens_to_generate} tokens ---")
        print(f"Prompt: '{prompt}'")

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = inputs.input_ids
        generated_ids = input_ids.clone()
        hidden_states = None

        print("\nGenerated tokens:")
        with torch.no_grad():
            outputs = model(input_ids=input_ids, hidden_states=hidden_states)
            logits = outputs.logits
            hidden_states = list(outputs.hidden_states)
            next_token_logits = logits[:, -1, :]

            # Sanitize
            next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
            next_token_logits[:, 0] = -float('inf')

            # Sample
            top_k = 50
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
            probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
            sampled_indices = torch.multinomial(probabilities, num_samples=1)
            next_token_id = top_k_indices.gather(-1, sampled_indices)

            for i in range(num_tokens_to_generate):
                token_text = tokenizer.decode(next_token_id.item())
                print(f"  Token {i+1}: '{token_text}' (ID: {next_token_id.item()})")

                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)

                if next_token_id.item() == tokenizer.eos_token_id:
                    print("  (End of sequence token generated, stopping.)")
                    break

                outputs = model(input_ids=next_token_id, hidden_states=hidden_states)
                logits = outputs.logits
                hidden_states = list(outputs.hidden_states)
                next_token_logits = logits[:, -1, :]
                next_token_logits = torch.nan_to_num(next_token_logits, nan=-1e9, posinf=-1e9, neginf=-1e9)
                next_token_logits[:, 0] = -float('inf')

                top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                probabilities = torch.nn.functional.softmax(top_k_logits, dim=-1)
                sampled_indices = torch.multinomial(probabilities, num_samples=1)
                next_token_id = top_k_indices.gather(-1, sampled_indices)

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"\nFinal Generated Text:\n{generated_text}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    MODEL_ID = "silx-ai/TARS-1B"
    while True:
        PROMPT = input("Enter your prompt: ")
        if PROMPT:
            break
        print("Prompt cannot be empty. Please try again.")
    generate_text(MODEL_ID, PROMPT)
