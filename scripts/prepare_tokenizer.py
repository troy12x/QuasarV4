# c:\quasarv4\scripts\prepare_tokenizer.py

from transformers import AutoTokenizer
import os

def main():
    # Define the tokenizer to use and the local path to save it
    tokenizer_name = "avey-ai/avey1-1.5B-base-preview-100BT"
    save_directory = "/home/jovyan/work/quasarv4"

    print(f"Downloading tokenizer '{tokenizer_name}'...")
    
    # Download and save the tokenizer files
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        tokenizer.save_pretrained(save_directory)
        print(f"Tokenizer successfully downloaded and saved to '{save_directory}'.")
        print("The files 'tokenizer.json' and 'tokenizer_config.json' are now populated.")
    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure you are logged into the Hugging Face Hub if the model is private.")
        print("You can log in using: huggingface-cli login")

if __name__ == '__main__':
    main()
