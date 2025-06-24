import os
import argparse
from huggingface_hub import HfApi
from transformers import AutoTokenizer

def upload_index_and_tokenizer(repo_id, model_dir, tokenizer_path, token):
    """
    Uploads the model index file and the tokenizer to a Hugging Face repository.
    """
    print(f"--- Starting Auxiliary File Upload to {repo_id} ---")
    api = HfApi(token=token)

    # 1. Upload the model index file
    index_file_path = os.path.join(model_dir, 'model.safetensors.index.json')
    print(f"\nLooking for index file at: {index_file_path}")

    if os.path.exists(index_file_path):
        print(f"Found index file. Uploading to {repo_id}...")
        try:
            api.upload_file(
                path_or_fileobj=index_file_path,
                path_in_repo='model.safetensors.index.json',
                repo_id=repo_id,
                repo_type='model'
            )
            print("Successfully uploaded model.safetensors.index.json.")
        except Exception as e:
            print(f"Error uploading index file: {e}")
    else:
        print("ERROR: model.safetensors.index.json not found in the model directory.")

    # 2. Download and re-upload the tokenizer
    print(f"\nDownloading tokenizer from {tokenizer_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, token=token)
        print("Tokenizer downloaded. Now pushing to your repository...")
        
        # This will upload all tokenizer files (tokenizer.json, special_tokens_map.json, etc.)
        tokenizer.push_to_hub(repo_id, token=token)
        print(f"Successfully pushed tokenizer files to {repo_id}.")
    except Exception as e:
        print(f"Error processing tokenizer: {e}")

    print("\n--- Auxiliary File Upload Complete ---")

def main():
    parser = argparse.ArgumentParser(description="Upload model index and tokenizer to HF Hub.")
    parser.add_argument('--repo_id', type=str, default='silx-ai/QuasarV4-400B-1M', help='Hugging Face repository ID.')
    parser.add_argument('--model_dir', type=str, default='./model', help='Local directory where the model index is located.')
    parser.add_argument('--tokenizer_path', type=str, default='deepseek-ai/DeepSeek-V3-0324', help='Source repository for the tokenizer.')
    parser.add_argument('--hf_token', type=str, required=True, help='Your Hugging Face Hub token.')
    
    args = parser.parse_args()
    
    upload_index_and_tokenizer(args.repo_id, args.model_dir, args.tokenizer_path, args.hf_token)

if __name__ == "__main__":
    main()
