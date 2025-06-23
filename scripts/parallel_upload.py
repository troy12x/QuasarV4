"""
Uploads model shards to the Hugging Face Hub in parallel to accelerate the process.
"""
import argparse
import os
import concurrent.futures
from huggingface_hub import HfApi, HfFolder
from tqdm.auto import tqdm

def main(args):
    """Main function to handle the parallel upload process."""
    hf_token = args.hf_token or HfFolder.get_token()
    if not hf_token:
        raise ValueError("Hugging Face token not found. Please login via `huggingface-cli login` or pass --hf_token.")

    api = HfApi()

    # Get local files to upload
    if not os.path.isdir(args.model_path):
        print(f"Error: Local model path '{args.model_path}' not found.")
        return

    # Include common model/tokenizer files like .json and .md
    local_files = sorted([f for f in os.listdir(args.model_path) if f.endswith(('.safetensors', '.json', '.md'))])

    # Get remote files to skip already uploaded ones
    try:
        print(f"Checking for existing files in repo: {args.repo_id}...")
        remote_files = {f.rfilename for f in api.list_repo_files_info(args.repo_id, token=hf_token)}
        print(f"Found {len(remote_files)} files on the Hub.")
    except Exception as e:
        print(f"Warning: Could not list remote files, will attempt to upload all local files. Error: {e}")
        remote_files = set()

    files_to_upload = [f for f in local_files if f not in remote_files]

    if not files_to_upload:
        print("All model files are already present on the Hub. Nothing to upload.")
        return

    print(f"\nFound {len(files_to_upload)} new files to upload. Submitting to the upload queue...")

    # Use the library's built-in async upload mechanism.
    # HfApi manages its own thread pool (default size is 5 workers).
    futures = []
    for filename in files_to_upload:
        file_path = os.path.join(args.model_path, filename)
        # The upload_file function will show its own progress bar for each file.
        # We will just track the completion of these uploads.
        future = api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=filename,
            repo_id=args.repo_id,
            token=hf_token,
            run_as_future=True, # This is the key to parallel uploads
        )
        futures.append(future)

    print(f"All {len(futures)} files have been submitted for upload. Waiting for completion...")

    # Wait for all uploads to complete and show an overall progress bar
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Overall Upload Progress"):
        try:
            # result() will raise an exception if the upload failed
            result = future.result()
        except Exception as e:
            # The uploader already prints detailed errors, so we just note that a failure occurred.
            print(f"An upload failed. See logs above for details. Error: {e}")

    print("\n--- Upload process finished. ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload model shards to Hugging Face Hub in parallel.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the directory containing model shards.")
    parser.add_argument("--repo_id", type=str, default="silx-ai/QuasarV4-400B-1M", help="Hugging Face Hub repository ID.")
    parser.add_argument("--hf_token", type=str, default="hf_EYcsIFrnfjkZwFCZVLvYpEyMGjSywQTw", help="Hugging Face Hub token.")

    args = parser.parse_args()
    main(args)
