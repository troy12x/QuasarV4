#!/usr/bin/env python3
"""
DeepSpeed Launch Script for TrueEvolving Multi-GPU Training

This script launches the TrueEvolving pretraining with DeepSpeed across multiple GPUs.
It handles distributed training setup and configuration management.

Usage:
    # Single GPU
    python launch_deepspeed.py --gpus 1
    
    # Multi-GPU (2 GPUs)
    python launch_deepspeed.py --gpus 2
    
    # Multi-GPU (4 GPUs) 
    python launch_deepspeed.py --gpus 4
    
    # Custom config
    python launch_deepspeed.py --gpus 2 --config custom_ds_config.json
"""

import os
import sys
import subprocess
import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Launch TrueEvolving training with DeepSpeed')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs to use')
    parser.add_argument('--config', type=str, default='scripts/ds_config.json', 
                       help='DeepSpeed config file path')
    parser.add_argument('--master_port', type=int, default=29500, 
                       help='Master port for distributed training')
    parser.add_argument('--script', type=str, default='scripts/small_scale_pretraining.py',
                       help='Training script to run')
    parser.add_argument('--mix_dataset', action='store_true', 
                       help='Enable mixed dataset training')
    parser.add_argument('--dataset_ratios', type=str, default='0.7,0.3',
                       help='Dataset mixing ratios (e.g., 0.7,0.3)')
    
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
    
    print("🚀 TrueEvolving DeepSpeed Multi-GPU Training Launcher")
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
        print(f"🔥 Launching multi-GPU training on {args.gpus} GPUs...")
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
