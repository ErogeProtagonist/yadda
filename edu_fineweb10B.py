"""
FineWeb-Edu Data Preparation Script (Cluster-Grade)

This script is designed for high-end clusters (H100/B200). 
It streaming-downloads, parallel-tokenizes, and raw-binary-shards the 
FineWeb-Edu (10B token) dataset.
"""

import os
import numpy as np
import multiprocessing as mp
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# --- Config ---
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "edu_fineweb10B")
SHARD_SIZE = int(1e8) # 100M tokens per shard
TOKENIZER_NAME = "gpt2"
DTYPE = np.uint16 
REMOTE_NAME = "sample-10BT" # The 10 billion token subset

# 1. Setup
os.makedirs(DATA_CACHE_DIR, exist_ok=True)
enc = tiktoken.get_encoding(TOKENIZER_NAME)
eot_token = enc.eot_token

def tokenize(doc):
    # Separate function for multiprocessing pool
    # Adds EOT token at the start of each document
    tokens = [eot_token]
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # Ensure it fits in uint16 (GPT-2 vocab is 50k, so it's fine)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all()
    return tokens_np.astype(np.uint16)

def write_shard(filename, tokens_np):
    # Save as raw binary file for direct memmap access
    with open(filename, "wb") as f:
        f.write(tokens_np.tobytes())

def prepare_data():
    print(f"Loading dataset '{REMOTE_NAME}' from FineWeb-Edu...")
    # Streaming mode handles massive datasets without crashing RAM
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=REMOTE_NAME, split="train", streaming=True)

    # multiprocessing pool for tokenization
    # Note: Using half the cores is usually enough as I/O is the bottleneck
    n_procs = max(1, os.cpu_count() // 2)
    
    with mp.Pool(n_procs) as pool:
        shard_idx = 0
        # Preallocate buffer for current shard (100M tokens = ~200MB)
        all_tokens_np = np.empty((SHARD_SIZE,), dtype=DTYPE)
        token_count = 0
        progress_bar = None
        
        print(f"Starting parallel processing on {n_procs} cores...")
        
        # We iterate over documents in chunks to keep the pool busy
        for tokens in pool.imap(tokenize, fw, chunksize=16):
            
            # Check if current document fits in this shard
            if token_count + len(tokens) < SHARD_SIZE:
                all_tokens_np[token_count:token_count+len(tokens)] = tokens
                token_count += len(tokens)
                if progress_bar is None:
                    progress_bar = tqdm(total=SHARD_SIZE, unit="tokens", desc=f"Shard {shard_idx}")
                progress_bar.update(len(tokens))
            else:
                # This doc overflows the shard. Split it.
                remainder = SHARD_SIZE - token_count
                progress_bar.update(remainder)
                all_tokens_np[token_count:SHARD_SIZE] = tokens[:remainder]
                
                # Write completed shard
                split = "val" if shard_idx == 0 else "train"
                filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_idx:06d}.bin")
                write_shard(filename, all_tokens_np)
                
                shard_idx += 1
                progress_bar = None
                
                # Start new shard with leftovers
                leftover = tokens[remainder:]
                all_tokens_np[0:len(leftover)] = leftover
                token_count = len(leftover)

        # Write final shard
        if token_count > 0:
            split = "val" if shard_idx == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_idx:06d}.bin")
            write_shard(filename, all_tokens_np[:token_count])

    print("\nData preparation complete!")

if __name__ == "__main__":
    prepare_data()
