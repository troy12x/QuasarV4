import os
import argparse
from huggingface_hub import HfApi

def push_to_hub(repo_id: str, folder_path: str):
    """
    Uploads all content from a local folder to a Hugging Face repository.

    Args:
        repo_id (str): The ID of the repository to push to (e.g., 'username/repo-name').
        folder_path (str): The path to the local folder to upload.
    """
    if not os.path.isdir(folder_path):
        print(f"Error: The folder '{folder_path}' was not found.")
        print("Please make sure you have run the training and the output directory exists.")
        return

    print(f"Uploading folder '{folder_path}' to repository '{repo_id}'...")
    
    # HfApi provides a programmatic interface to the Hugging Face Hub.
    api = HfApi()

    # The upload_folder function handles the entire upload process.
    # It will automatically use your cached token from `huggingface-cli login`.
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload fine-tuned LoRA model from {os.path.basename(folder_path)}"
    )

    print("\nUpload complete!")
    print(f"Check out your model at: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    # --- IMPORTANT --- #
    # Before running this script, please log in to your Hugging Face account
    # by opening a terminal and running the following command:
    # huggingface-cli login
    # Then, paste your new Hugging Face token when prompted.
    # ----------------- #

    parser = argparse.ArgumentParser(description="Push a local model folder to the Hugging Face Hub.")
    
    parser.add_argument(
        "--repo_id",
        type=str,
        default="eyad-silx/tars",
        help="The ID of the Hugging Face repository (e.g., 'your-username/your-repo-name')."
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        # The user specified the output is in /home/jovyan/work/quasarv4/pretrain_output
        # Assuming the script is run from /home/jovyan/work/quasarv4, this is the relative path.
        default="pretrain_output/checkpoint-6000",
        help="The path to the local folder containing the model files."
    )

    args = parser.parse_args()

    push_to_hub(repo_id=args.repo_id, folder_path=args.folder_path)
