import os
import argparse
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def inspect_remote_shard(repo_id, filename, token):
    """
    Downloads a single model shard from a Hugging Face Hub repository
    and inspects its contents.
    """
    print(f"Attempting to download '{filename}' from repo '{repo_id}'...")
    
    try:
        # Download the specific file to a local cache
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=token,
            cache_dir="./hf_cache"
        )
        print(f"Successfully downloaded to: {local_path}")
        
        # Load the safetensors file
        print("\n--- Inspecting Tensors ---")
        tensors = load_file(local_path)
        
        if not tensors:
            print("The shard is empty or could not be read.")
            return

        max_layer_idx = -1
        max_expert_idx = -1

        # Find the highest layer and expert index from the tensor names
        for name in tensors.keys():
            parts = name.split('.')
            for i, part in enumerate(parts):
                if part == 'layers' and i + 1 < len(parts) and parts[i+1].isdigit():
                    layer_idx = int(parts[i+1])
                    if layer_idx > max_layer_idx:
                        max_layer_idx = layer_idx
                elif part == 'experts' and i + 1 < len(parts) and parts[i+1].isdigit():
                    expert_idx = int(parts[i+1])
                    if expert_idx > max_expert_idx:
                        max_expert_idx = expert_idx

        print("\n--- Configuration Analysis from Shard ---")
        if max_layer_idx > -1:
            # Add 1 because indices are 0-based (e.g., layer 95 means 96 layers total)
            num_layers = max_layer_idx + 1
            print(f"Detected Number of Hidden Layers: {num_layers}")
        else:
            print("Could not determine the number of hidden layers from this shard.")

        if max_expert_idx > -1:
            # Add 1 because indices are 0-based
            num_experts = max_expert_idx + 1
            print(f"Detected Number of Experts per Layer: {num_experts}")
        else:
            print("Could not determine the number of experts from this shard.")

        print("\nInspection complete.")
        
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check the repository ID, filename, and your Hugging Face token.")

def main():
    parser = argparse.ArgumentParser(description="Download and inspect a single shard from Hugging Face Hub.")
    parser.add_argument('--repo_id', type=str, default='silx-ai/QuasarV4-400B-1M', help='Hugging Face Hub repository ID.')
    parser.add_argument('--filename', type=str, default='model-00284-of-00355.safetensors', help='The exact filename of the shard to download.')
    parser.add_argument('--hf_token', type=str, required=True, help='Your Hugging Face Hub token.')
    
    args = parser.parse_args()
    
    inspect_remote_shard(args.repo_id, args.filename, args.hf_token)

if __name__ == "__main__":
    main()
