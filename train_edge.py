"""
Training Script for Hybrid SWA and MLA Transformers.

Based on the proven train_gpt2.py but adapted for:
- Both Hybrid (Gemma-style) and MLA (DeepSeek-style) models
- Automatic kernel selection based on hardware
- Integration with edu_fineweb10B.py data pipeline

Usage:
    # Single GPU
    python train_edge.py --model_type hybrid
    
    # Multi-GPU DDP
    torchrun --standalone --nproc_per_node=8 train_edge.py --model_type mla
"""

import os
import math
import time
import argparse
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

import numpy as np
import tiktoken

# Import our models
from models.config import ModelConfig, HYBRID_500M, MLA_500M
from models.transformer import Transformer


# =============================================================================
# DATA LOADING (from train_gpt2.py)
# =============================================================================

def load_tokens(filename):
    """Load tokenized data using memory mapping for zero-latency access."""
    npt = np.memmap(filename, dtype=np.uint16, mode='r')
    ptt = torch.from_numpy(npt.astype(np.int64))
    return ptt


class DataLoaderLite:
    """Lightweight data loader for sharded binary token files."""
    
    def __init__(self, B, T, process_rank, num_processes, split, data_root, master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.master_process = master_process
        assert split in {'train', 'val'}

        # Get shard filenames
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s and s.endswith('.bin')]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split} in {data_root}"
        
        if master_process:
            print(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]
        x = (buf[:-1]).view(B, T)
        y = (buf[1:]).view(B, T)
        
        # Advance position
        self.current_position += B * T * self.num_processes
        
        # Move to next shard if needed
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
            
        return x, y


# =============================================================================
# LEARNING RATE SCHEDULE
# =============================================================================

def get_lr(step, warmup_steps, max_steps, max_lr, min_lr):
    """Cosine learning rate schedule with linear warmup."""
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    # After max_steps, return min_lr
    if step > max_steps:
        return min_lr
    
    # Cosine decay
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


# =============================================================================
# MAIN TRAINING LOOP
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train Hybrid or MLA Transformer")
    parser.add_argument("--model_type", type=str, default="hybrid", choices=["hybrid", "mla"],
                        help="Model architecture: 'hybrid' (Gemma-style) or 'mla' (DeepSeek-style)")
    parser.add_argument("--shards_dir", type=str, default="/lambda/nfs/riddy/edu_fineweb10B",
                        help="Directory containing tokenized shards (Lambda: /lambda/nfs/riddy/edu_fineweb10B)")
    parser.add_argument("--log_dir", type=str, default="log",
                        help="Directory for logs and checkpoints")
    parser.add_argument("--max_steps", type=int, default=19073,
                        help="Maximum training steps (~1 epoch for 10B tokens)")
    parser.add_argument("--val_interval", type=int, default=250,
                        help="Validation interval")
    parser.add_argument("--save_interval", type=int, default=5000,
                        help="Checkpoint save interval")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Micro batch size per GPU (16 for 500M on H100, 8 for local 3090)")
    parser.add_argument("--total_batch_size", type=int, default=524288,
                        help="Total batch size in tokens (~0.5M)")
    parser.add_argument("--max_lr", type=float, default=6e-4,
                        help="Maximum learning rate")
    parser.add_argument("--min_lr", type=float, default=6e-5,
                        help="Minimum learning rate")
    parser.add_argument("--warmup_steps", type=int, default=715,
                        help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1,
                        help="Weight decay")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile")
    args = parser.parse_args()

    # =========================================================================
    # DDP SETUP
    # =========================================================================
    ddp = int(os.environ.get('RANK', -1)) != -1
    
    if ddp:
        assert torch.cuda.is_available(), "DDP requires CUDA"
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0
    else:
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        if master_process:
            print(f"Using device: {device}")

    device_type = "cuda" if device.startswith("cuda") else "cpu"
    
    # Reproducibility
    torch.manual_seed(1337)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)

    # =========================================================================
    # MODEL SETUP
    # =========================================================================
    if master_process:
        print(f"\n{'='*60}")
        print(f"Training {args.model_type.upper()} Transformer")
        print(f"{'='*60}\n")
    
    # Select config based on model type
    if args.model_type == "hybrid":
        config = HYBRID_500M
    else:
        config = MLA_500M
    
    # Determine mode based on hardware
    mode = "train"  # Will use optimized kernels if available
    
    model = Transformer(config, mode=mode)
    model.to(device)
    
    if args.compile:
        if master_process:
            print("Compiling model with torch.compile...")
        model = torch.compile(model)
    
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    
    raw_model = model.module if ddp else model

    # =========================================================================
    # DATA LOADER SETUP
    # =========================================================================
    B = args.batch_size
    T = config.block_size
    
    assert args.total_batch_size % (B * T * ddp_world_size) == 0, \
        "total_batch_size must be divisible by B * T * world_size"
    grad_accum_steps = args.total_batch_size // (B * T * ddp_world_size)
    
    if master_process:
        print(f"Total batch size: {args.total_batch_size:,} tokens")
        print(f"Micro batch size: {B} sequences x {T} tokens = {B*T:,} tokens/GPU")
        print(f"Gradient accumulation steps: {grad_accum_steps}")
        print(f"Effective batch: {B * T * grad_accum_steps * ddp_world_size:,} tokens")

    train_loader = DataLoaderLite(
        B=B, T=T, 
        process_rank=ddp_rank, 
        num_processes=ddp_world_size,
        split="train", 
        data_root=args.shards_dir,
        master_process=master_process
    )
    val_loader = DataLoaderLite(
        B=B, T=T,
        process_rank=ddp_rank,
        num_processes=ddp_world_size,
        split="val",
        data_root=args.shards_dir,
        master_process=master_process
    )

    # =========================================================================
    # OPTIMIZER SETUP
    # =========================================================================
    torch.set_float32_matmul_precision('high')
    optimizer = raw_model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.max_lr,
        device_type=device_type
    )

    # =========================================================================
    # LOGGING SETUP
    # =========================================================================
    os.makedirs(args.log_dir, exist_ok=True)
    log_file = os.path.join(args.log_dir, f"log_{args.model_type}.txt")
    if master_process:
        with open(log_file, "w") as f:
            f.write(f"Training {args.model_type} model\n")
            f.write(f"Config: {asdict(config)}\n\n")

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    enc = tiktoken.get_encoding("gpt2")
    
    for step in range(args.max_steps):
        t0 = time.time()
        last_step = (step == args.max_steps - 1)

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        if step % args.val_interval == 0 or last_step:
            model.eval()
            val_loader.reset()
            
            with torch.no_grad():
                val_loss_accum = 0.0
                val_loss_steps = 20
                
                for _ in range(val_loss_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    
                    with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                        _, loss, _ = model(x, targets=y)
                    
                    loss = loss / val_loss_steps
                    val_loss_accum += loss.detach()
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            
            if master_process:
                print(f"Step {step:5d} | val loss: {val_loss_accum.item():.4f}")
                with open(log_file, "a") as f:
                    f.write(f"{step} val {val_loss_accum.item():.4f}\n")

        # ---------------------------------------------------------------------
        # CHECKPOINTING
        # ---------------------------------------------------------------------
        if master_process and step > 0 and (step % args.save_interval == 0 or last_step):
            checkpoint_path = os.path.join(args.log_dir, f"{args.model_type}_{step:05d}.pt")
            checkpoint = {
                'model': raw_model.state_dict(),
                'config': asdict(config),
                'step': step,
                'val_loss': val_loss_accum.item() if 'val_loss_accum' in dir() else None,
            }
            torch.save(checkpoint, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

        # ---------------------------------------------------------------------
        # SAMPLE GENERATION
        # ---------------------------------------------------------------------
        if master_process and ((step > 0 and step % args.val_interval == 0) or last_step):
            model.eval()
            prompt = "The future of artificial intelligence"
            tokens = enc.encode(prompt)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
            
            with torch.no_grad():
                generated = raw_model.generate(tokens, max_new_tokens=50, temperature=0.8)
            
            text = enc.decode(generated[0].tolist())
            print(f"\nGenerated: {text}\n")

        # ---------------------------------------------------------------------
        # TRAINING STEP
        # ---------------------------------------------------------------------
        model.train()
        optimizer.zero_grad()
        loss_accum = torch.zeros(1, device=device)
        
        for micro_step in range(grad_accum_steps):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            
            if ddp:
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                _, loss, _ = model(x, targets=y)
            
            loss = loss / grad_accum_steps
            loss_accum += loss.detach()
            loss.backward()
        
        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
        
        # Gradient clipping
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Learning rate schedule
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        optimizer.step()
        
        if device_type == "cuda":
            torch.cuda.synchronize()
        
        t1 = time.time()
        dt = t1 - t0
        tokens_processed = B * T * grad_accum_steps * ddp_world_size
        tokens_per_sec = tokens_processed / dt
        
        if master_process:
            print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.2e} | "
                  f"norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:,.0f}")
            with open(log_file, "a") as f:
                f.write(f"{step} train {loss_accum.item():.6f}\n")

    # =========================================================================
    # CLEANUP
    # =========================================================================
    if ddp:
        destroy_process_group()
    
    if master_process:
        print("\nTraining complete!")


if __name__ == "__main__":
    main()
