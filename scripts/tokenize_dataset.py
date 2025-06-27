import argparse
import os
from itertools import chain
from datasets import Dataset, Features, Value, Sequence
from transformers import AutoTokenizer
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

def stream_parquet_split(repo_id, split, text_column, max_files=None):
    """
    A general-purpose generator that lists all Parquet files in a dataset split,
    but only downloads and yields the specified text column from the first `max_files`.
    """
    api = HfApi()
    repo_info = api.repo_info(repo_id, repo_type="dataset")

    # Discover Parquet files by looking for files under the 'data/' directory
    # that also contain the split name in their path. This is a robust way
    # to handle different dataset structures on the Hub.
    prefix = "data/"
    all_parquet_files = [f.rfilename for f in repo_info.siblings if f.rfilename.endswith(".parquet")]
    files = sorted([f for f in all_parquet_files if f.startswith(prefix) and split in f])

    if not files:
        raise ValueError(
            f"No Parquet files found for split '{split}' under the '{prefix}' directory in repo '{repo_id}'. "
            f"Please check the dataset repository to confirm the file structure."
        )

    print(f"[stream_parquet_split] Found {len(files)} total Parquet files for split '{split}'.")

    files_to_process = files
    if max_files is not None:
        print(f"[stream_parquet_split] --- Limiting to the first {max_files} files. ---")
        files_to_process = files[:max_files]

    for filename in files_to_process:
        print(f"[stream_parquet_split] Processing file: {filename}")
        try:
            local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
            # Efficiently read only the required column
            table = pq.read_table(local_path, columns=[text_column])
            for batch in table.to_batches():
                # Yield each text entry as a dictionary
                for text in batch.to_pydict()[text_column]:
                    if text: # Ensure we don't yield empty strings
                        yield {text_column: text}
        except Exception as e:
            print(f"[stream_parquet_split] WARNING: Failed to process {filename}. Error: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize a generic Parquet dataset and save it to disk.")
    parser.add_argument("--tokenizer_name", type=str, required=True, help="Tokenizer to use.")
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset to tokenize.")
    parser.add_argument("--dataset_split", type=str, required=True, help="Dataset split to use.")
    parser.add_argument("--text_column", type=str, default="content", help="The column in the dataset containing the text to tokenize.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the tokenized dataset.")
    parser.add_argument("--sequence_length", type=int, default=4096, help="Sequence length for packing.")
    parser.add_argument("--num_proc", type=int, default=os.cpu_count(), help="Number of CPU processes for tokenization.")
    parser.add_argument("--max_files_to_process", type=int, default=None, help="Limit processing to the first N files to create a sample.")

    args = parser.parse_args()

    print(f"Loading tokenizer: {args.tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, trust_remote_code=True)

    print(f"Starting data streamer for: {args.dataset_name} (Split: {args.dataset_split}, Column: {args.text_column})")

    # The features now dynamically use the specified text column
    features = Features({
        args.text_column: Value('string'),
    })

    raw_dataset = Dataset.from_generator(
        lambda: stream_parquet_split(
            repo_id=args.dataset_name,
            split=args.dataset_split,
            text_column=args.text_column,
            max_files=args.max_files_to_process
        ),
        features=features
    )
    print("Dataset stream loaded. Starting tokenization...")

    def tokenize_function(examples):
        return tokenizer(examples[args.text_column], truncation=False, padding=False)

    tokenized_dataset = raw_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_proc,
        remove_columns=[args.text_column]
    )

    def pack_iterator(dataset, sequence_length, batch_size=1000):
        """Packs a dataset of tokenized examples into fixed-length sequences."""
        buffer = []
        for batch in dataset.iter(batch_size=batch_size):
            tokens = list(chain.from_iterable(batch['input_ids']))
            buffer.extend(tokens)
            while len(buffer) >= sequence_length:
                chunk = buffer[:sequence_length]
                buffer = buffer[sequence_length:]
                yield {"input_ids": chunk, "labels": chunk.copy()}

    print("Packing dataset into fixed-length sequences...")

    packed_features = Features({
        'input_ids': Sequence(feature=Value(dtype='int64')),
        'labels': Sequence(feature=Value(dtype='int64')),
    })

    packed_dataset = Dataset.from_generator(
        lambda: pack_iterator(tokenized_dataset, args.sequence_length),
        features=packed_features
    )

    print(f"Saving processed dataset to {args.output_dir}")
    packed_dataset.save_to_disk(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
