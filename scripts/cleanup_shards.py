import os
import re
import argparse

def cleanup_model_shards(directory, start_keep, end_keep, execute=False):
    """
    Deletes model shard files (.safetensors) in a directory that are outside
    a specified range of shard numbers.

    Args:
        directory (str): The path to the directory containing the model shards.
        start_keep (int): The starting shard number of the range to keep.
        end_keep (int): The ending shard number of the range to keep.
        execute (bool): If True, perform deletion. If False, perform a dry run.
    """
    # Regex to find the shard number, e.g., "model-00123-of-..."
    shard_pattern = re.compile(r"model-(\d{5})-of-\d{5}\.safetensors")
    
    print(f"--- Shard Cleanup (Range Mode) ---")
    print(f"Target Directory: {directory}")
    print(f"Keeping Shards: From {start_keep} to {end_keep}")
    print(f"Mode: {'EXECUTE' if execute else 'DRY RUN'}")
    print("----------------------------------")

    if not os.path.isdir(directory):
        print(f"Error: Directory not found at '{directory}'")
        return

    files_to_delete = []
    for filename in os.listdir(directory):
        match = shard_pattern.match(filename)
        if match:
            shard_number = int(match.group(1))
            if not (start_keep <= shard_number <= end_keep):
                files_to_delete.append(filename)

    if not files_to_delete:
        print("No shards found outside the specified range. Nothing to do.")
        return

    for filename in sorted(files_to_delete):
        file_path = os.path.join(directory, filename)
        if execute:
            try:
                os.remove(file_path)
                print(f"DELETED: {filename}")
            except OSError as e:
                print(f"Error deleting {filename}: {e}")
        else:
            print(f"WOULD DELETE: {filename}")
            
    if not execute:
        print(f"\nDry run complete. To delete these {len(files_to_delete)} files, run again with the --execute flag.")
    else:
        print(f"\nDeletion complete. {len(files_to_delete)} files removed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean up model shards outside a specified range.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "directory",
        type=str,
        help="The directory containing the .safetensors model shards."
    )
    parser.add_argument(
        "start_keep",
        type=int,
        help="The starting shard number of the range to KEEP."
    )
    parser.add_argument(
        "end_keep",
        type=int,
        help="The ending shard number of the range to KEEP."
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete the files. If not provided, will only perform a dry run."
    )

    args = parser.parse_args()
    cleanup_model_shards(args.directory, args.start_keep, args.end_keep, args.execute)
