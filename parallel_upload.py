"""
Script to upload the QuasarV4-400B-1M model to Hugging Face Hub with a rich,
multi-file parallel progress display.
"""

import os
import concurrent.futures
from functools import partialmethod

from huggingface_hub import HfApi, create_repo
from transformers import AutoTokenizer
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TransferSpeedColumn,
    TimeRemainingColumn,
)
from tqdm.auto import tqdm

# --- Configuration ---
HF_REPO = "silx-ai/QuasarV4-400B-1M"
HF_TOKEN = "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"
MODEL_DIR = "./model"
TOKENIZER_PATH = "deepseek-ai/DeepSeek-V3-0324"
MAX_WORKERS = 4  # Number of files to upload at the same time
START_SHARD = 94 # Start uploading from the shard AFTER this one

class ProgressCallbackFile:
    """A file-like object that updates a Rich progress bar on each read."""
    def __init__(self, path, progress, task_id):
        self._file = open(path, 'rb')
        self._progress = progress
        self._task_id = task_id
        self._size = os.path.getsize(path)
        self._progress.update(self._task_id, total=self._size)
        self._progress.start_task(self._task_id)

    def read(self, size=-1):
        chunk = self._file.read(size)
        if chunk:
            self._progress.update(self._task_id, advance=len(chunk))
        return chunk

    def __len__(self):
        return self._size

    def __getattr__(self, attr):
        return getattr(self._file, attr)

    def __enter__(self):
        self._file.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._file.__exit__(exc_type, exc_val, exc_tb)


def upload_model():
    # --- Setup ---
    print("\nInitializing Hugging Face API...")
    api = HfApi()
    try:
        create_repo(HF_REPO, token=HF_TOKEN, repo_type="model", exist_ok=True)
        print(f"Repository '{HF_REPO}' confirmed.")
    except Exception as e:
        print(f"Could not create repository: {e}")
        return

    # --- File Filtering ---
    all_files = sorted([f for f in os.listdir(MODEL_DIR) if f.startswith('model-') and f.endswith('.safetensors')])
    files_to_upload = [f for f in all_files if int(f.split('-')[1]) > START_SHARD]
    print(f"Found {len(files_to_upload)} files to upload, starting after shard {START_SHARD}.")

    if not files_to_upload:
        print("No new files to upload.")
    else:
        # --- Parallel Upload with Rich Progress ---
        progress = Progress(
            TextColumn("[bold blue]{task.fields[filename]}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%", "•",
            TransferSpeedColumn(), "•",
            TimeRemainingColumn(),
        )
            try:
                # result() will raise an exception if the upload failed
                future.result()
            except Exception as e:
                # The uploader usually prints detailed errors. This is a fallback.
                print(f"An upload failed. See logs above for details. Error: {e}")

    # Upload tokenizer
    print("\nUploading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, token=HF_TOKEN)
    tokenizer.save_pretrained(HF_REPO, push_to_hub=True, token=HF_TOKEN)

    print(f"\nModel successfully uploaded to {HF_REPO}")

def main():
    upload_model()

if __name__ == '__main__':
    main()
