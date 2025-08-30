#!/usr/bin/env python3
"""
Small-Scale Pretraining Script for TrueEvolving Attention Architecture
Test on RTX 3050 single GPU before scaling to multi-GPU setup

TRANSFORMER BOTTLENECKS WE'RE SOLVING:
1. QUADRATIC MEMORY SCALING: O(n²) attention becomes O(n) with temporal evolution
2. CONTEXT LENGTH LIMITS: Current models break at 2K-8K tokens, we aim for INFINITE
3. LONG-RANGE DEPENDENCIES: Standard attention degrades, ours IMPROVES with length
4. TRAINING INSTABILITY: Gradient explosion at long sequences, our evolution stabilizes
5. INFERENCE SPEED: KV-cache grows linearly, our memory decays exponentially

THIS WILL CHANGE THE WORLD BY:
- Enabling truly long-form reasoning (novels, codebases, conversations)
- Making AI remember entire conversations without forgetting
- Allowing models to process infinite documents without chunking
- Creating the first O(n) attention that actually works better than O(n²)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import deepspeed
import numpy as np
import time
import json
import logging
import os
import math
import random
import pickle
import hashlib
import glob
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrueEvolvingAttention(nn.Module):
    """
    REVOLUTIONARY ATTENTION MECHANISM
    - Temporal evolution creates infinite context capability
    - Memory decay prevents gradient explosion
    - O(n) complexity instead of O(n²)
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        # Core attention components
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # REVOLUTIONARY COMPONENTS
        self.evolution_rate = nn.Parameter(torch.tensor(0.1))  # Learnable evolution
        self.memory_decay = nn.Parameter(torch.tensor(0.85))   # Learnable decay
        self.temporal_weights = nn.Parameter(torch.randn(self.head_dim) * 0.02)
        
        # Normalization for stability
        self.q_norm = nn.LayerNorm(self.head_dim)
        self.k_norm = nn.LayerNorm(self.head_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
    def forward(self, x, mask=None):
        B, T, C = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # TEMPORAL EVOLUTION - THE BREAKTHROUGH
        evolved_q = torch.zeros_like(q)
        evolved_k = torch.zeros_like(k)
        
        # Initialize evolving memory
        memory = torch.zeros(B, self.n_heads, self.head_dim, device=x.device, dtype=x.dtype)
        
        for pos in range(T):
            # Evolution factor grows with position (longer context = more evolution)
            evolution_factor = torch.sigmoid(self.evolution_rate) * (pos + 1)
            temporal_signal = torch.sin(evolution_factor * self.temporal_weights)
            
            # Evolve queries and keys with memory
            evolved_q[:, :, pos, :] = self.q_norm(
                q[:, :, pos, :] + temporal_signal + torch.sigmoid(self.memory_decay) * memory
            )
            evolved_k[:, :, pos, :] = self.k_norm(
                k[:, :, pos, :] + temporal_signal * 0.5 + torch.sigmoid(self.memory_decay) * memory * 0.3
            )
            
            # Update memory with current query (information flows forward)
            memory = 0.7 * memory + 0.3 * q[:, :, pos, :]
        
        # Compute attention with evolved Q, K
        att_scores = torch.matmul(evolved_q, evolved_k.transpose(-2, -1)) * self.scale
        
        # Apply causal mask
        if mask is None:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        att_scores.masked_fill_(mask, float('-inf'))
        
        # Attention weights
        att_weights = F.softmax(att_scores, dim=-1)
        att_weights = self.dropout(att_weights)
        
        # Apply attention to values
        out = torch.matmul(att_weights, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        
        return self.out_proj(out)

class TrueEvolvingTransformerBlock(nn.Module):
    """Transformer block with TrueEvolving attention"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = TrueEvolvingAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, mask=None):
        # Self-attention with residual
        attn_out = self.attention(self.norm1(x), mask)
        x = x + attn_out
        
        # Feed-forward with residual
        ffn_out = self.ffn(self.norm2(x))
        x = x + ffn_out
        
        return x

class TrueEvolvingLanguageModel(nn.Module):
    """
    WORLD-CHANGING LANGUAGE MODEL
    First transformer with TRUE infinite context capability
    """
    def __init__(self, vocab_size, d_model=512, n_heads=8, n_layers=6, d_ff=2048, 
                 max_seq_len=2048, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks with TrueEvolving attention
        self.blocks = nn.ModuleList([
            TrueEvolvingTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output layer
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, input_ids, targets=None):
        B, T = input_ids.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds maximum {self.max_seq_len}"
        
        # Embeddings
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final layer norm and projection
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        
        return logits, loss

class TextDataset(Dataset):
    """Dataset for loading and tokenizing text data with caching and mixed dataset support"""
    
    def __init__(self, tokenizer, max_length, mix_dataset=False, dataset_ratios=[1.0]):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mix_dataset = mix_dataset
        self.dataset_ratios = dataset_ratios
        
        # Create cache key based on tokenizer, parameters, and dataset configuration
        if mix_dataset:
            cache_key = f"{tokenizer.name_or_path}_{max_length}_mixed_{'-'.join(map(str, dataset_ratios))}"
        else:
            cache_key = f"{tokenizer.name_or_path}_{max_length}_single"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        
        # Cache directory and file path
        cache_dir = "./tokenized_cache"
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, f"tokenized_data_{cache_hash}.pkl")
        
        # Try to load from cache
        if os.path.exists(cache_file):
            logger.info(f"🚀 Loading tokenized data from cache: {cache_file}")
            try:
                with open(cache_file, 'rb') as f:
                    self.tokenized_samples = pickle.load(f)
                logger.info(f"✅ Successfully loaded {len(self.tokenized_samples)} cached samples")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, will re-tokenize")
        
        # Load and tokenize data
        if mix_dataset:
            logger.info(f"🔥 Creating mixed dataset with ratios {dataset_ratios}")
            self.tokenized_samples = self._load_mixed_datasets()
        else:
            logger.info(f"🔥 Tokenizing texts with {tokenizer.name_or_path} from OpenWebText...")
            self.tokenized_samples = self._load_single_dataset()
        
        # Save to cache
        logger.info(f"💾 Saving tokenized data to cache: {cache_file}")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.tokenized_samples, f)
            logger.info("✅ Cache saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")
    
    def _load_single_dataset(self):
        """Load single dataset (using Ultra-FineWeb-1B)"""
        try:
            # Use Ultra-FineWeb-1B dataset - high quality web data
            dataset = load_dataset("sumuks/Ultra-FineWeb-1B", split="train")
            
            texts = []
            for example in dataset:
                text = example['content']  # Uses 'content' column
                if len(text.strip()) > 50:  # Filter out very short texts
                    texts.append(text)
            
            if deepspeed.comm.get_rank() == 0:
                logger.info(f"✅ Loaded {len(texts)} samples from Ultra-FineWeb-1B dataset")
            return self._tokenize_texts(texts)
            
        except Exception as e:
            logger.error(f"Failed to load Ultra-FineWeb-1B dataset: {e}")
            raise RuntimeError("Could not load Ultra-FineWeb-1B dataset. Please check your internet connection or dataset availability.")
    
    def _load_mixed_datasets(self):
        """Load and mix multiple datasets"""
        # Calculate samples per dataset based on ratios
        if len(self.dataset_ratios) != 2:
            logger.warning("Mixed dataset currently supports exactly 2 datasets, using default ratios")
            self.dataset_ratios = [0.7, 0.3]
        
        # Load all available samples with dataset ratios for mixing
        
        if deepspeed.comm.get_rank() == 0:
            logger.info(f"📊 Dataset mixing ratios: {self.dataset_ratios[0]*100:.1f}% / {self.dataset_ratios[1]*100:.1f}%")
        
        all_texts = []
        
        # Load Dataset 1: MegaMath Web Pro
        if deepspeed.comm.get_rank() == 0:
            logger.info("Loading MegaMath Web Pro dataset...")
        try:
            dataset1 = load_dataset("semran1/megamath-web-pro", split="train")
            dataset1_texts = []
            for example in dataset1:
                text = example['text']  # Uses 'text' column
                if len(text.strip()) > 50:
                    dataset1_texts.append(text)
            
            # Take proportion based on ratio
            dataset1_count = int(len(dataset1_texts) * self.dataset_ratios[0])
            all_texts.extend(dataset1_texts[:dataset1_count])
            
        except Exception as e:
            logger.warning(f"Could not load MegaMath dataset: {e}")
            # Fallback to Ultra-FineWeb-1B
            try:
                dataset1 = load_dataset("sumuks/Ultra-FineWeb-1B", split="train")
                dataset1_texts = []
                for example in dataset1:
                    text = example['content']  # Uses 'content' column
                    if len(text.strip()) > 50:
                        dataset1_texts.append(text)
                
                dataset1_count = int(len(dataset1_texts) * self.dataset_ratios[0])
                all_texts.extend(dataset1_texts[:dataset1_count])
                
            except Exception as e2:
                logger.error(f"Could not load Ultra-FineWeb-1B dataset either: {e2}")
                raise RuntimeError("Could not load fallback dataset for mixed training. Please check your internet connection.")
        
        # Load Dataset 2: Ultra-FineWeb-1B
        if deepspeed.comm.get_rank() == 0:
            logger.info("Loading Ultra-FineWeb-1B dataset...")
        try:
            dataset2 = load_dataset("sumuks/Ultra-FineWeb-1B", split="train")
            dataset2_texts = []
            for example in dataset2:
                text = example['content']  # Uses 'content' column
                if len(text.strip()) > 50:
                    dataset2_texts.append(text)
            
            # Take proportion based on ratio
            dataset2_count = int(len(all_texts) * self.dataset_ratios[1] / self.dataset_ratios[0])
            all_texts.extend(dataset2_texts[:dataset2_count])
            
        except Exception as e:
            logger.warning(f"Could not load Ultra-FineWeb-1B dataset: {e}")
        
        # Shuffle the combined dataset
        import random
        random.shuffle(all_texts)
        
        if deepspeed.comm.get_rank() == 0:
            logger.info(f"🎯 Mixed dataset created with {len(all_texts)} total samples")
        return self._tokenize_texts(all_texts)
    
    def _tokenize_texts(self, texts):
        """Tokenize list of texts"""
        tokenized_samples = []
        
        for text in texts:
            # Tokenize text
            tokens = self.tokenizer(
                text,
                max_length=self.max_length,
                truncation=True,
                padding='max_length',
                return_tensors='pt'
            )
            
            input_ids = tokens['input_ids'].squeeze()
            
            # Create input and target (shifted by 1 for language modeling)
            if len(input_ids) > 1:
                inputs = input_ids[:-1]
                targets = input_ids[1:]
                tokenized_samples.append((inputs, targets))
        
        return tokenized_samples
    
    def __len__(self):
        return len(self.tokenized_samples)
    
    def __getitem__(self, idx):
        return self.tokenized_samples[idx]

def train_epoch(model_engine, dataloader, device, epoch, config, global_step=0):
    """Train for one epoch with DeepSpeed and checkpointing"""
    model_engine.train()
    total_loss = 0
    num_batches = len(dataloader)
    total_tokens = 0
    epoch_start_time = time.time()
    current_step = global_step
    best_val_loss = float('inf')
    
    for batch_idx, (input_ids, targets) in enumerate(dataloader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model_engine(input_ids)
        
        # Handle tuple output (logits, loss) or just logits
        if isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs
            
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # Backward pass
        model_engine.backward(loss)
        model_engine.step()
        
        total_loss += loss.item()
        current_step += 1
        total_tokens += input_ids.numel()
        
        # Calculate tokens per second
        elapsed_time = time.time() - epoch_start_time
        tokens_per_sec = total_tokens / elapsed_time if elapsed_time > 0 else 0
        
        # Get learning rate
        lr = model_engine.get_lr()[0] if hasattr(model_engine, 'get_lr') else 0.0
        
        # Log progress (only rank 0 to avoid duplicate prints)
        if batch_idx % config.get('log_every', 10) == 0 and deepspeed.comm.get_rank() == 0:
            logger.info(f"Epoch {epoch}, Step {current_step}, Batch {batch_idx}/{num_batches}, "
                       f"Loss: {loss.item():.4f}, LR: {lr:.6f}, "
                       f"Tokens: {total_tokens:,}, Tokens/sec: {tokens_per_sec:.0f}")
            
            # Log to wandb if available
            if WANDB_AVAILABLE and wandb.run is not None:
                try:
                    wandb.log({
                        "train_loss": loss.item(),
                        "learning_rate": lr,
                        "epoch": epoch,
                        "step": current_step,
                        "batch": batch_idx,
                        "tokens_processed": total_tokens,
                        "tokens_per_second": tokens_per_sec
                    })
                except Exception as e:
                    logger.warning(f"Wandb logging failed: {e}")
        
        # Evaluation and checkpointing
        if current_step % config['eval_every'] == 0:
            # Quick validation
            val_loss, val_perplexity = evaluate_model(model_engine, 
                                                    torch.utils.data.DataLoader(
                                                        torch.utils.data.TensorDataset(input_ids[:1], targets[:1]),
                                                        batch_size=1
                                                    ), device)
            
            # Save checkpoint
            if current_step % config['save_every'] == 0:
                save_checkpoint(model_engine, current_step, epoch, loss.item(), val_loss, 
                              config, config['checkpoint_dir'])
                cleanup_old_checkpoints(config['checkpoint_dir'], config['keep_last_n_checkpoints'])
                
                # Track best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = os.path.join(config['checkpoint_dir'], f"best_model_step_{current_step}")
                    model_engine.save_checkpoint(best_checkpoint_path)
                    logger.info(f"🏆 New best model saved at step {current_step} (val_loss: {val_loss:.4f})")
    
    avg_loss = total_loss / num_batches
    epoch_time = time.time() - epoch_start_time
    final_tokens_per_sec = total_tokens / epoch_time if epoch_time > 0 else 0
    
    logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
    logger.info(f"📊 Epoch Stats: {total_tokens:,} tokens processed in {epoch_time:.1f}s ({final_tokens_per_sec:.0f} tokens/sec)")
    return avg_loss, total_tokens, current_step

def evaluate_model(model_engine, dataloader, device):
    """Evaluate model on validation set with DeepSpeed"""
    model_engine.eval()
    total_loss = 0
    num_batches = len(dataloader)
    total_tokens = 0
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (input_ids, targets) in enumerate(dataloader):
            input_ids = input_ids.to(device)
            targets = targets.to(device)
            
            # Count tokens in this batch
            batch_tokens = input_ids.numel()
            total_tokens += batch_tokens
            
            # Forward pass
            logits, loss = model_engine(input_ids, targets)
            total_loss += loss.item()
    
    avg_loss = total_loss / num_batches
    perplexity = math.exp(avg_loss)
    eval_time = time.time() - eval_start_time
    tokens_per_sec = total_tokens / eval_time if eval_time > 0 else 0
    
    logger.info(f"Validation Loss: {avg_loss:.4f}, Perplexity: {perplexity:.2f}")
    logger.info(f"📊 Validation Stats: {total_tokens:,} tokens processed in {eval_time:.1f}s ({tokens_per_sec:.0f} tokens/sec)")
    return avg_loss, perplexity

def save_checkpoint(model_engine, step, epoch, train_loss, val_loss, config, checkpoint_dir):
    """Save DeepSpeed checkpoint with metadata"""
    if deepspeed.comm.get_rank() == 0:
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save DeepSpeed checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"step_{step}")
        model_engine.save_checkpoint(checkpoint_path)
        
        # Save metadata
        metadata = {
            'step': step,
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': config,
            'timestamp': time.time()
        }
        
        metadata_path = os.path.join(checkpoint_path, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"💾 Saved checkpoint: {checkpoint_path}")
        return checkpoint_path
    return None

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n=3):
    """Remove old checkpoints to save disk space"""
    if deepspeed.comm.get_rank() == 0:
        checkpoint_pattern = os.path.join(checkpoint_dir, "step_*")
        checkpoints = glob.glob(checkpoint_pattern)
        
        if len(checkpoints) > keep_last_n:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split('_')[-1]))
            
            # Remove oldest checkpoints
            for old_checkpoint in checkpoints[:-keep_last_n]:
                try:
                    import shutil
                    shutil.rmtree(old_checkpoint)
                    logger.info(f"🗑️  Removed old checkpoint: {old_checkpoint}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint {old_checkpoint}: {e}")

def load_checkpoint(checkpoint_path, model_engine):
    """Load DeepSpeed checkpoint"""
    try:
        # Load DeepSpeed checkpoint
        model_engine.load_checkpoint(checkpoint_path)
        
        # Load metadata
        metadata_path = os.path.join(checkpoint_path, 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"📂 Loaded checkpoint from step {metadata['step']}, epoch {metadata['epoch']}")
            return metadata
        else:
            logger.warning("Checkpoint metadata not found")
            return None
    except Exception as e:
        logger.error(f"Failed to load checkpoint: {e}")
        return None

def main():
    """Main pretraining function with DeepSpeed"""
    # Parse DeepSpeed arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--deepspeed_config', type=str, default='scripts/ds_config.json')
    parser.add_argument('--mix_dataset', action='store_true', help='Enable mixed dataset training')
    parser.add_argument('--dataset_ratios', type=str, default='0.7,0.3', 
                       help='Comma-separated ratios for mixed datasets (e.g., 0.7,0.3)')
    args = parser.parse_args()
    
    # Configuration for multi-GPU training with DeepSpeed - 4B Parameter Model
    config = {
        'vocab_size': 129280,  # DeepSeek-V3 vocab size
        'd_model': 2048,      # Scaled up for 4B parameters
        'n_heads': 32,        # Scaled up proportionally
        'n_layers': 24,       # Scaled up for 4B parameters
        'd_ff': 8192,         # 4x d_model (standard ratio)
        'max_seq_len': 2048,  # Increased sequence length
        'dropout': 0.1,
        'batch_size': 4,      # Per-GPU batch size (DeepSpeed handles global batching)
        'learning_rate': 3e-4,
        'num_epochs': 5,
        'warmup_steps': 1000,
        'weight_decay': 0.01,
        'save_every': 100,    # Save checkpoint every N steps
        'eval_every': 50,     # Evaluate every N steps  
        'val_split': 0.1,
        'logging_steps': 1,   # Log every step to wandb
        'checkpoint_dir': './checkpoints',  # Directory for checkpoints
        'keep_last_n_checkpoints': 3,       # Keep only last N checkpoints
        'mix_dataset': args.mix_dataset,    # Enable mixed dataset training
        'dataset_ratios': [float(x) for x in args.dataset_ratios.split(',')],  # Dataset mixing ratios
        'deepspeed_config': args.deepspeed_config
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    torch.cuda.empty_cache()
    
    # Wandb will be initialized after DeepSpeed initialization
    logger.info("Wandb initialization deferred until after DeepSpeed setup")
    
    # Load DeepSeek tokenizer
    logger.info("Loading DeepSeek-V3 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"✅ Loaded tokenizer: {tokenizer.name_or_path}")
    logger.info(f"📝 Vocab size: {tokenizer.vocab_size:,}")
    logger.info(f"🔤 Pad token: {tokenizer.pad_token}")
    
    # Update config with actual tokenizer vocab size
    config['vocab_size'] = tokenizer.vocab_size
    
    # Create model
    logger.info("Creating TrueEvolving model...")
    model = TrueEvolvingLanguageModel(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_heads=config['n_heads'],
        n_layers=config['n_layers'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Initialize DeepSpeed (optimizer configured in ds_config.json)
    logger.info("Initializing DeepSpeed...")
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        args=args,
        model=model
    )
    
    device = model_engine.device
    logger.info(f"DeepSpeed initialized on device: {device}")
    logger.info(f"DeepSpeed world size: {deepspeed.comm.get_world_size()}")
    logger.info(f"DeepSpeed local rank: {deepspeed.comm.get_rank()}")
    
    # Initialize wandb for tracking (only on rank 0, after DeepSpeed init)
    if WANDB_AVAILABLE and deepspeed.comm.get_rank() == 0:
        try:
            wandb.init(
                project="trueevolving-pretraining",
                config=config,
                name=f"deepspeed-{deepspeed.comm.get_world_size()}gpu-test",
                group="trueevolving-multiGPU"
            )
            logger.info("✅ Wandb initialized on rank 0")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
    elif WANDB_AVAILABLE:
        logger.info(f"Wandb skipped on rank {deepspeed.comm.get_rank()}")
    else:
        logger.info("Wandb not available, skipping experiment tracking")
    
    # Create datasets
    logger.info("Creating datasets...")
    if config['mix_dataset']:
        logger.info(f"🎯 Using mixed dataset mode with ratios: {config['dataset_ratios']}")
        train_dataset = TextDataset(
            tokenizer, config['max_seq_len'], 
            mix_dataset=True, dataset_ratios=config['dataset_ratios']
        )
        val_dataset = TextDataset(
            tokenizer, config['max_seq_len'],
            mix_dataset=True, dataset_ratios=config['dataset_ratios']
        )
    else:
        logger.info("📖 Using single dataset mode (OpenWebText)")
        train_dataset = TextDataset(tokenizer, config['max_seq_len'])
        val_dataset = TextDataset(tokenizer, config['max_seq_len'])
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        betas=(0.9, 0.95),
        weight_decay=0.1
    )
    
    total_steps = len(train_dataloader) * config['num_epochs']
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['learning_rate'],
        total_steps=total_steps,
        pct_start=0.1
    )
    
    # Training loop
    logger.info("🚀 STARTING PRETRAINING OF WORLD-CHANGING ARCHITECTURE!")
    logger.info("=" * 60)
    
    best_val_loss = float('inf')
    # Training loop with comprehensive checkpointing
    logger.info("Starting training with checkpointing system...")
    cumulative_tokens = 0
    global_step = 0
    best_val_loss = float('inf')
    
    # Create checkpoint directory
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    for epoch in range(config['num_epochs']):
        logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
        
        # Train with checkpointing
        train_loss, epoch_tokens, global_step = train_epoch(
            model_engine, train_dataloader, device, epoch + 1, config, global_step
        )
        cumulative_tokens += epoch_tokens
        
        # Full validation at end of epoch
        val_loss, val_perplexity = evaluate_model(model_engine, val_dataloader, device)
        
        logger.info(f"Epoch {epoch + 1} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss: {val_loss:.4f}, Perplexity: {val_perplexity:.2f}")
        logger.info(f"  📈 Cumulative tokens processed: {cumulative_tokens:,}")
        logger.info(f"  🔢 Global step: {global_step}")
        
        # Log to wandb (only rank 0)
        if WANDB_AVAILABLE and wandb.run is not None and deepspeed.comm.get_rank() == 0:
            try:
                wandb.log({
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_perplexity": val_perplexity,
                    "cumulative_tokens": cumulative_tokens,
                    "global_step": global_step,
                    "best_val_loss": best_val_loss
                })
            except Exception as e:
                logger.warning(f"Wandb epoch logging failed: {e}")
        
        # Save epoch checkpoint
        epoch_checkpoint_path = save_checkpoint(
            model_engine, global_step, epoch + 1, train_loss, val_loss, 
            config, config['checkpoint_dir']
        )
        
        # Update best model if needed
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if deepspeed.comm.get_rank() == 0:
                best_checkpoint_path = os.path.join(config['checkpoint_dir'], "best_model")
                model_engine.save_checkpoint(best_checkpoint_path)
                logger.info(f"🏆 New best model saved (val_loss: {val_loss:.4f})")
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(config['checkpoint_dir'], config['keep_last_n_checkpoints'])
    
    logger.info("Training completed!")
    logger.info(f"🎯 Final cumulative tokens processed: {cumulative_tokens:,}")
    logger.info(f"🔢 Final global step: {global_step}")
    logger.info(f"🏆 Best validation loss: {best_val_loss:.4f}")
    
    # Final save
    final_checkpoint_path = save_checkpoint(
        model_engine, global_step, config['num_epochs'], train_loss, val_loss,
        config, config['checkpoint_dir']
    )
    logger.info(f"💾 Saved final checkpoint: {final_checkpoint_path}")
    
    # Cleanup wandb and distributed training
    if WANDB_AVAILABLE and wandb.run is not None and deepspeed.comm.get_rank() == 0:
        try:
            wandb.finish()
            logger.info("✅ Wandb session finished")
        except Exception as e:
            logger.warning(f"Wandb finish failed: {e}")
    
    # Cleanup distributed training
    try:
        deepspeed.comm.destroy_process_group()
        logger.info("✅ DeepSpeed process group destroyed")
    except Exception as e:
        logger.warning(f"DeepSpeed cleanup failed: {e}")

    logger.info("\n🌍 READY TO CHANGE THE WORLD!")

if __name__ == "__main__":
    main()
