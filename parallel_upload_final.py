"""
Script to upload the QuasarV4-400B-1M model to Hugging Face Hub with a rich,
multi-file parallel progress display.

Run this script to see multiple, simultaneous upload bars.
"""

import os
import io
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
MODEL_DIR = "./temp_model"
TOKENIZER_PATH = "deepseek-ai/DeepSeek-V3-0324"
MAX_WORKERS = 10  # Number of files to upload at the same time
START_SHARD = 285 # Start uploading from the shard AFTER this one

class ProgressCallbackFile(io.BufferedIOBase):
    """A file-like object that updates a Rich progress bar on each read."""
    def __init__(self, path, progress, task_id, filename):
        self._file = open(path, 'rb')
        self._progress = progress
        self._task_id = task_id
        self._filename = filename
        self._size = os.path.getsize(path)
        self._progress.update(self._task_id, total=self._size)
        self._progress.start_task(self._task_id)

    def read(self, size=-1):
        chunk = self._file.read(size)
        if chunk:
            self._progress.update(self._task_id, advance=len(chunk))
        else:
            # When read is finished, update the description to show it's finalizing
            self._progress.update(self._task_id, description=f"[yellow]Finalizing: {self._filename}")
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

    def seek(self, offset, whence=0):
        return self._file.seek(offset, whence)

    def tell(self):
        return self._file.tell()


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
    files_to_upload = [f for f in all_files if int(f.split('-')[1].split('.')[0]) > START_SHARD]
    print(f"Found {len(files_to_upload)} files to upload, starting after shard {START_SHARD}.")

    if not files_to_upload:
        print("No new files to upload.")
    else:
        # --- Parallel Upload with Rich Progress ---
        progress = Progress(
            TextColumn("{task.description}", justify="right"),
            BarColumn(bar_width=None),
            "[progress.percentage]{task.percentage:>3.1f}%", "•",
            TransferSpeedColumn(), "•",
            TimeRemainingColumn(),
        )

        # Temporarily disable the default tqdm bars to avoid visual glitches
        original_tqdm_init = tqdm.__init__
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)

        with progress:
            with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_task = {}
                for filename in files_to_upload:
                    # Set initial state to "Sending"
                    task_id = progress.add_task(description=f"[cyan]Sending:   {filename}", start=False)
                    file_path = os.path.join(MODEL_DIR, filename)
                    # Pass filename to the callback so it can update the state
                    wrapped_file = ProgressCallbackFile(file_path, progress, task_id, filename)
                    
                    future = executor.submit(api.upload_file, path_or_fileobj=wrapped_file, path_in_repo=filename, repo_id=HF_REPO, token=HF_TOKEN)
                    future_to_task[future] = (task_id, filename)

                for future in concurrent.futures.as_completed(future_to_task):
                    task_id, filename = future_to_task[future]
                    try:
                        future.result()  # Wait for the upload to be fully confirmed by the Hub
                        # Set final state to "Completed"
                        progress.update(task_id, description=f"[bold green]✅ Completed: {filename}")
                    except Exception as e:
                        progress.update(task_id, description=f"[bold red]❌ Failed:    {filename}")
        
        # Restore the original tqdm behavior
        tqdm.__init__ = original_tqdm_init

    # --- Upload Tokenizer (sequentially) ---
    print("\nUploading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, token=HF_TOKEN)
        tokenizer.save_pretrained(HF_REPO, push_to_hub=True, token=HF_TOKEN)
        print("Tokenizer uploaded successfully.")
    except Exception as e:
        print(f"Could not upload tokenizer: {e}")

    print(f"\n--- Process Complete ---")

if __name__ == '__main__':
    # You may need to install rich: pip install rich
    upload_model()
    