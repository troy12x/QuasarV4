import os
import argparse
import json

# This is a simplified representation. In a real scenario, you would import this from your model's code.
# from quasar.model import QuasarConfig

def generate_config_file(output_dir):
    """
    Generates and saves the config.json file for the QuasarV4 model.
    """
    print(f"--- Generating config.json for QuasarV4-400B ---")

    # These parameters MUST match the model you intend to save/upload.
    # This is the blueprint for your model's architecture.
    # Inspired by best practices from other major models, we create a comprehensive config.
    config_data = {
        # --- Model Architecture ---
        "model_type": "quasar",
        "architectures": ["Quasar"],
        "hidden_size": 8192,
        "num_hidden_layers": 96,
        "num_attention_heads": 64, # From QuasarConfig in model.py
        "num_key_value_heads": 64, # Standard MHA, not GQA

        # --- MoE Details ---
        "num_experts": 128,
        "num_experts_per_tok": 32, # This is our 'top_k' parameter
        "expert_dim": 2048, # The intermediate size within each expert
        "hidden_act": "gelu", # Found in the Expert class in moe.py

        # --- Tokenizer & Vocab ---
        "vocab_size": 151552,
        "bos_token_id": 100006, # Standard for deepseek tokenizer
        "eos_token_id": 100007, # Standard for deepseek tokenizer
        "tie_word_embeddings": False,

        # --- Technical Details ---
        "initializer_range": 0.02, # Standard practice
        "layer_norm_eps": 1e-5, # Default for torch.nn.LayerNorm
        "torch_dtype": "bfloat16",

        # --- HF Ecosystem Integration ---
        "auto_map": {
            "AutoConfig": "quasar.model.QuasarConfig",
            "AutoModelForCausalLM": "quasar.model.Quasar"
        }
    }

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, 'config.json')

    # Save the configuration file
    with open(file_path, 'w') as f:
        json.dump(config_data, f, indent=2)

    print(f"Successfully saved config.json to: {file_path}")
    print("--- Generation Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Generate config.json for the Quasar model.")
    parser.add_argument('--output_dir', type=str, default='./model', help='Directory to save the config.json file.')
    args = parser.parse_args()
    generate_config_file(args.output_dir)

if __name__ == "__main__":
    main()
