"""
Script to upload the QuasarV4-400B-1M model to Hugging Face Hub.
"""

import os
from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer

# Configuration
HF_REPO = "silx-ai/QuasarV4-400B-1M"
HF_TOKEN = "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"
MODEL_DIR = "./model"  # Directory containing the model files
TOKENIZER_PATH = "deepseek-ai/DeepSeek-V3-0324"

def upload_model():
    print("\nUploading model to Hugging Face Hub...")
    
    # Create repository if it doesn't exist
    try:
        create_repo(HF_REPO, token=HF_TOKEN, repo_type="model", exist_ok=True)
        print(f"Repository '{HF_REPO}' created or already exists.")
    except Exception as e:
        print(f"Could not create repository: {e}")
        return

    # Upload model files
    print("\nUploading model files...")
    api = HfApi()
    
    # Get all model files sorted by number
    model_files = []
    for file in os.listdir(MODEL_DIR):
        if file.startswith('model-') and file.endswith('.safetensors'):
            model_files.append(file)
    
    # Sort files by their number (e.g., model-00001-of-00355.safetensors)
    model_files.sort(key=lambda x: int(x.split('-')[1].split('.')[0]))
    
    # Upload each file starting from file 14
    for file in model_files:
        # Extract the file number (e.g., '00013' from 'model-00013-of-00355.safetensors')
        file_num = int(file.split('-')[1].split('.')[0])
        
        # Skip files that have already been uploaded (1-13)
        if file_num <= 13:
            print(f"Skipping already uploaded file: {file}")
            continue
            
        file_path = os.path.join(MODEL_DIR, file)
        print(f"Uploading {file}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=file,
            repo_id=HF_REPO,
            token=HF_TOKEN,
            repo_type="model"
        )
        
        # Add a small delay between uploads to avoid rate limiting
        import time
        time.sleep(1)

    # Upload tokenizer
    print("\nUploading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, token=HF_TOKEN)
    tokenizer.save_pretrained(HF_REPO, push_to_hub=True, token=HF_TOKEN)

    print(f"\nModel successfully uploaded to {HF_REPO}")

def main():
    upload_model()

if __name__ == '__main__':
    main()
