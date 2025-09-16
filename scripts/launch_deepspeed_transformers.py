#!/usr/bin/env python3
"""
DeepSpeed Launch Script for Standard Transformer Multi-GPU Training

This script launches the standard transformer pretraining with DeepSpeed across multiple GPUs.
It handles distributed training setup and configuration management.

Usage:
    # Single GPU
    python launch_deepspeed_transformers.py --gpus 1
    
    # Multi-GPU (2 GPUs)
    python launch_deepspeed_transformers.py --gpus 2
    
    # Multi-GPU (4 GPUs) 
    python launch_deepspeed_transformers.py --gpus 4
    
    # Custom config
    python launch_deepspeed_transformers.py --gpus 2 --config custom_ds_config.json
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch Standard Transformer training with DeepSpeed')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--config', type=str, default='scripts/ds_config.json', 
                       help='DeepSpeed config file path')
    parser.add_argument('--master_port', type=int, default=29500, 
                       help='Master port for distributed training')
    parser.add_argument('--script', type=str, default='scripts/small_transformers.py',
                       help='Training script to run')
    parser.add_argument('--mix_dataset', action='store_true', 
                       help='Enable mixed dataset training')
    parser.add_argument('--dataset_ratios', type=str, default='0.7,0.3',
                       help='Dataset mixing ratios (e.g., 0.7,0.3)')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume from (e.g., checkpoints/best_model_step_84000)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: DeepSpeed config file not found: {config_path}")
        sys.exit(1)
    
    # Validate training script exists
    script_path = Path(args.script)
    if not script_path.exists():
        print(f"❌ Error: Training script not found: {script_path}")
        sys.exit(1)
    
    print("🚀 Standard Transformer DeepSpeed Multi-GPU Training Launcher")
    print("=" * 60)
    print(f"📊 GPUs: {args.gpus}")
    print(f"⚙️  Config: {args.config}")
    print(f"🐍 Script: {args.script}")
    print(f"🔌 Master Port: {args.master_port}")
    print("=" * 60)
    
    # Set environment variables
    os.environ['MASTER_PORT'] = str(args.master_port)
    
    # Build base command
    if args.gpus == 1:
        # Single GPU training
        print("🔥 Launching single GPU training...")
        cmd = [
            sys.executable, args.script,
            '--deepspeed_config', args.config,
            '--local_rank', '0'
        ]
    else:
        # Multi-GPU training with DeepSpeed launcher
        print(f"🔥 Launching standard transformer multi-GPU training on {args.gpus} GPUs...")
        cmd = [
            'deepspeed',
            '--num_gpus', str(args.gpus),
            '--master_port', str(args.master_port),
            args.script,
            '--deepspeed_config', args.config
        ]
    
    # Add mixed dataset arguments if specified
    if args.mix_dataset:
        cmd.extend(['--mix_dataset'])
        cmd.extend(['--dataset_ratios', args.dataset_ratios])
        print(f"🎯 Mixed dataset enabled with ratios: {args.dataset_ratios}")
    
    # Add checkpoint resumption if specified
    if args.resume_from_checkpoint:
        # Validate checkpoint exists
        checkpoint_path = Path(args.resume_from_checkpoint)
        print(f"🔍 DEBUG: Checking checkpoint path: {checkpoint_path}")
        print(f"🔍 DEBUG: Absolute path: {checkpoint_path.absolute()}")
        print(f"🔍 DEBUG: Path exists: {checkpoint_path.exists()}")
        if checkpoint_path.exists():
            print(f"🔍 DEBUG: Directory contents: {list(checkpoint_path.iterdir())}")
        
        if not checkpoint_path.exists():
            print(f"❌ Error: Checkpoint not found: {checkpoint_path}")
            print(f"❌ Absolute path checked: {checkpoint_path.absolute()}")
            sys.exit(1)
        cmd.extend(['--resume_from_checkpoint', args.resume_from_checkpoint])
        print(f"🔄 Resuming from checkpoint: {args.resume_from_checkpoint}")
    
    print(f"💻 Command: {' '.join(cmd)}")
    print("🚀 Starting training...")
    print("=" * 60)
    
    try:
        # Launch the training
        result = subprocess.run(cmd, check=True)
        print("✅ Training completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Training failed with exit code: {e.returncode}")
        sys.exit(e.returncode)
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
