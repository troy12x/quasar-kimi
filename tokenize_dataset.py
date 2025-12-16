"""
Tokenize MegaMath-Web-Pro dataset for Kimi Linear continual pretraining.
HIGH PERFORMANCE VERSION: Uses multiprocessing and dataset mapping for maximum speed.
"""

import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing as mp

# Global tokenizer for multiprocessing
tokenizer = None

def tokenize_batch(examples, tokenizer_path):
    """Tokenize a batch of texts."""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, 
            trust_remote_code=True,
            use_fast=True
        )
        
    return tokenizer(
        examples["text"], 
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )

def group_texts(examples, max_seq_length, pad_id, eos_id):
    """
    Concatenate texts and chunk them into max_seq_length.
    Adds EOS token between documents.
    """
    # Concatenate all texts
    concatenated_examples = {k: [] for k in examples.keys()}
    
    for input_ids in examples["input_ids"]:
        concatenated_examples["input_ids"].extend(input_ids)
        concatenated_examples["input_ids"].append(eos_id)

    total_length = len(concatenated_examples["input_ids"])
    
    # Drop the last chunk if it's too small
    if total_length >= max_seq_length:
        total_length = (total_length // max_seq_length) * max_seq_length
        
    # Split by chunks of max_len
    result = {
        "input_ids": [
            concatenated_examples["input_ids"][i : i + max_seq_length]
            for i in range(0, total_length, max_seq_length)
        ]
    }
    return result

def save_tokenized_dataset(
    dataset_name: str,
    split: str,
    text_column: str,
    output_dir: str,
    tokenizer_name: str,
    max_seq_length: int,
    num_proc: int = None,
    max_samples: int = None,
):
    start_total = time.time()
    
    # 1. Setup
    if num_proc is None:
        num_proc = mp.cpu_count()
        
    print(f"\n{'='*60}")
    print(f"High-Performance Tokenization Configuration:")
    print(f"  - CPU cores: {num_proc}")
    print(f"  - Target chunk size: {max_seq_length:,} tokens (2M)")
    print(f"  - Output directory: {output_dir}")
    print(f"  - Tokenizer: {tokenizer_name}")
    print(f"{'='*60}\n")
    
    # 2. Load Dataset
    print(f"Loading dataset {dataset_name}...")
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples:
        print(f"Selecting first {max_samples} samples for testing...")
        dataset = dataset.select(range(max_samples))
        
    # 3. Rename column if needed
    if text_column != "text" and text_column in dataset.column_names:
        dataset = dataset.rename_column(text_column, "text")

    # 4. Tokenize (Parallel)
    print(f"Phase 1: Tokenizing {len(dataset):,} documents (Parallel)...")
    # We need to initialize the tokenizer in the main process first for basic info
    main_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True, use_fast=True)
    eos_id = main_tokenizer.eos_token_id
    pad_id = main_tokenizer.pad_token_id
    vocab_size = main_tokenizer.vocab_size

    tokenized_dataset = dataset.map(
        tokenize_batch,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
        desc="Tokenizing",
        fn_kwargs={"tokenizer_path": tokenizer_name}
    )

    # 5. Grouping (Parallel)
    print(f"Phase 2: Grouping into {max_seq_length:,} token chunks (Parallel)...")
    grouped_dataset = tokenized_dataset.map(
        partial(group_texts, max_seq_length=max_seq_length, pad_id=pad_id, eos_id=eos_id),
        batched=True,
        num_proc=num_proc,
        desc="Grouping",
    )

    # 6. Save to Binary
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokens_file = output_path / "tokens.bin"
    metadata_path = output_path / "metadata.json"
    
    print(f"Phase 3: Saving to binary file {tokens_file}...")
    
    total_chunks = len(grouped_dataset)
    total_tokens = total_chunks * max_seq_length
    
    # Pre-allocate file? No, just write iteratively.
    # To be extremely fast, we could assume dataset is in arrow and memory mapped, 
    # but we need a flat binary file for the training script.
    
    # We can iterate and write.
    with open(tokens_file, "wb") as f:
        for i, batch in enumerate(tqdm(grouped_dataset, desc="Writing", unit=" chunks")):
            # array is list of ints
            chunk_array = np.array(batch["input_ids"], dtype=np.uint32)
            chunk_array.tofile(f)

    # 7. Metadata
    metadata = {
        'dataset_name': dataset_name,
        'split': split,
        'tokenizer_name': tokenizer_name,
        'max_seq_length': max_seq_length,
        'num_chunks': total_chunks,
        'total_tokens': total_tokens,
        'vocab_size': vocab_size,
        'eos_token_id': eos_id,
        'pad_token_id': pad_id,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
        
    duration = time.time() - start_total
    print(f"\n{'='*60}")
    print(f"✓ DONE in {duration:.1f} seconds ({duration/60:.1f} minutes)")
    print(f"  - Total chunks: {total_chunks:,}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Speed: {total_tokens/duration:,.0f} tokens/sec")
    print(f"{'='*60}\n")

from functools import partial
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="High Performance Tokenizer")
    parser.add_argument("--dataset_name", type=str, default="semran1/megamath-web-pro")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--text_column", type=str, default="text")
    parser.add_argument("--output_dir", type=str, default="./data/tokenized_2m")
    parser.add_argument("--tokenizer_name", type=str, default="silx-ai/Quasar-2M-Base")
    parser.add_argument("--max_seq_length", type=int, default=2097152)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--max_samples", type=int, default=1000000)
    
    args = parser.parse_args()
    
    save_tokenized_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        text_column=args.text_column,
        output_dir=args.output_dir,
        tokenizer_name=args.tokenizer_name,
        max_seq_length=args.max_seq_length,
        num_proc=args.num_proc,
        max_samples=args.max_samples,
    )

if __name__ == "__main__":
    main()
