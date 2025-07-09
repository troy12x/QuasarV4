import argparse
import os
import logging
import torch
import shutil
import json
from safetensors.torch import save_file
from huggingface_hub import HfApi
from pathlib import Path

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_and_push_model(output_dir: str, push_to_hub: bool, token: str = None, push_repo_id: str = None, checkpoint_dir: str = None, hub_model_id_source: str = None):
    """
    Prepares a model for the Hub from either a local checkpoint or another Hub repository,
    then optionally pushes it.

    Args:
        output_dir (str): Directory to save the final safetensors model.
        push_to_hub (bool): If True, push the converted model to the Hub.
        token (str, optional): Your Hugging Face API token.
        push_repo_id (str, optional): The target repository ID. Required if push_to_hub is True.
        checkpoint_dir (str, optional): Path to the local checkpoint directory.
        hub_model_id_source (str, optional): The source repository ID on the Hub.
    """
    output_path = Path(output_dir)
    source_path = None
    weights = None

    try:
        if hub_model_id_source:
            # Mode 1: Start from a Hub repository
            logger.info(f"Starting from Hub model: '{hub_model_id_source}'")
            source_path = Path(snapshot_download(repo_id=hub_model_id_source, token=token))
            
            # Load weights from either safetensors or bin
            safetensors_file = source_path / "model.safetensors"
            bin_file = source_path / "pytorch_model.bin"
            if safetensors_file.exists():
                logger.info("Loading weights from model.safetensors...")
                # This doesn't load the full model, just the weights.
                from safetensors.torch import load_file
                weights = load_file(safetensors_file, device="cpu")
            elif bin_file.exists():
                logger.info("Loading weights from pytorch_model.bin...")
                weights = torch.load(bin_file, map_location="cpu")
            else:
                raise FileNotFoundError("Could not find model weights (.safetensors or .bin) in the source repository.")

        elif checkpoint_dir:
            # Mode 2: Start from a local checkpoint
            logger.info(f"Starting from local checkpoint: '{checkpoint_dir}'")
            source_path = Path(checkpoint_dir)
            training_state_file = source_path / "training_state.pt"
            if not training_state_file.exists():
                raise FileNotFoundError(f"'training_state.pt' not found in '{checkpoint_dir}'.")
            
            logger.info(f"Loading weights from '{training_state_file}'...")
            state = torch.load(training_state_file, map_location="cpu")
            if 'model_state_dict' not in state:
                raise KeyError("'model_state_dict' not found in 'training_state.pt'.")
            weights = state['model_state_dict']
        
        # Step 1: Create output directory and copy all files from source
        logger.info(f"Copying files from '{source_path}' to '{output_path}'...")
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(source_path, output_path)

        # Step 2: Save weights in safetensors format (ensures consistency)
        safetensors_path = output_path / "model.safetensors"
        logger.info(f"Saving weights to '{safetensors_path}'...")
        save_file(weights, safetensors_path)
        logger.info("Successfully created .safetensors file.")
        
        # Step 3: Create the model.safetensors.index.json file
        logger.info("Generating model.safetensors.index.json...")
        index_path = output_path / "model.safetensors.index.json"
        total_size = sum(tensor.numel() * tensor.element_size() for tensor in weights.values())
        index_json = {
            "metadata": {"total_size": total_size},
            "weight_map": {k: "model.safetensors" for k in weights.keys()}
        }
        with open(index_path, 'w') as f:
            json.dump(index_json, f, indent=4)
        logger.info(f"Successfully created '{index_path.name}'.")

        # Step 4: Update config.json for Hub compatibility
        logger.info("Updating config.json for Hub compatibility...")
        config_path = output_path / "config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            config["model_type"] = "lnn"
            config["auto_map"] = {"AutoModel": "quasar.lnn.LNNModel"}
            
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            logger.info("config.json updated successfully.")
        else:
            logger.warning("config.json not found, cannot update.")


        # Step 5: Clean up unnecessary files from the output directory
        logger.info("Cleaning up unnecessary checkpoint files...")
        for file_to_delete in ["training_state.pt", "optimizer.pt", "scheduler.pt", "pytorch_model.bin", "optimizer.bin", "scheduler.bin"]:
            if (output_path / file_to_delete).exists():
                os.remove(output_path / file_to_delete)

        # Step 6: Push to Hub if requested
        if push_to_hub:
            if not push_repo_id:
                logger.error("FATAL: Must provide --push_repo_id to push to the Hub.")
                return
                
            logger.info(f"Preparing to push the final model to '{push_repo_id}'...")
            api = HfApi(token=token)
            api.upload_folder(
                folder_path=str(output_path),
                repo_id=push_repo_id,
                repo_type="model",
                commit_message="Add safetensors model from local checkpoint"
            )
            logger.info(f"Successfully pushed to '{push_repo_id}'.")
        else:
            logger.info("Skipping push to Hub. The final model is ready at:")
            logger.info(f"  {output_path.resolve()}")

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(
        description="Convert a local training checkpoint or Hub model to a final, Hub-ready format."
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Path to the local checkpoint directory containing 'training_state.pt'."
    )
    parser.add_argument(
        "--hub_model_id_source",
        type=str,
        default=None,
        help="The source repository ID on the Hub to start from."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory where the final model will be saved."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Set this flag to push the converted model to the Hub."
    )
    parser.add_argument(
        "--push_repo_id",
        type=str,
        default=None,
        help="The repository ID to push the converted model to. Required if --push_to_hub is set."
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Your Hugging Face API token. Needed for pushing to a private repo."
    )
    args = parser.parse_args()

    if args.push_to_hub and not args.push_repo_id:
        parser.error("--push_repo_id is required when --push_to_hub is set.")
    
    if not args.checkpoint_dir and not args.hub_model_id_source:
        parser.error("Either --checkpoint_dir or --hub_model_id_source must be provided.")
    
    if args.checkpoint_dir and args.hub_model_id_source:
        parser.error("Provide either --checkpoint_dir or --hub_model_id_source, not both.")

    prepare_and_push_model(
        output_dir=args.output_dir,
        push_to_hub=args.push_to_hub,
        token=args.token,
        push_repo_id=args.push_repo_id,
        checkpoint_dir=args.checkpoint_dir,
        hub_model_id_source=args.hub_model_id_source
    )

if __name__ == "__main__":
    main()