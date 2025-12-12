"""
Tokenize MegaMath-Web-Pro dataset for Kimi Linear continual pretraining.
Preprocesses the 40-100B token dataset and saves it in memory-mapped format for efficient training.
"""

import os
import argparse
import json
from pathlib import Path
from typing import Iterator, List, Dict
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
import multiprocessing as mp
from functools import partial


def tokenize_function(examples: Dict, tokenizer, text_column: str = "content"):
    """Tokenize a batch of examples."""
    return tokenizer(examples[text_column], add_special_tokens=False)


def chunk_sequences(token_ids: List[int], max_length: int, stride: int = None):
    """
    Chunk a sequence of token IDs into overlapping windows.
    
    Args:
        token_ids: List of token IDs
        max_length: Maximum sequence length
        stride: Overlap between chunks (if None, no overlap)
    
    Yields:
        Chunks of token IDs
    """
    if stride is None:
        stride = max_length  # No overlap
    
    for i in range(0, len(token_ids), stride):
        chunk = token_ids[i:i + max_length]
        if len(chunk) == max_length:  # Only yield full chunks
            yield chunk


def save_tokenized_chunks(
    dataset_name: str,
    split: str,
    text_column: str,
    output_dir: str,
    tokenizer_name: str,
    max_seq_length: int,
    num_proc: int = None,
    streaming: bool = True,
    max_samples: int = None,
):
    """
    Tokenize dataset and save as memory-mapped numpy array.
    
    Args:
        dataset_name: HuggingFace dataset name
        split: Dataset split to use
        text_column: Name of text column
        output_dir: Directory to save tokenized data
        tokenizer_name: HuggingFace tokenizer name or path
        max_seq_length: Maximum sequence length (2M for target)
        num_proc: Number of processes for tokenization
        streaming: Whether to use streaming mode
        max_samples: Maximum number of samples to process (for testing)
    """
    print(f"Loading tokenizer from {tokenizer_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name, 
        trust_remote_code=True,
        use_fast=True
    )
    
    print(f"Loading dataset {dataset_name} (split: {split})...")
    dataset = load_dataset(
        dataset_name,
        split=split,
        streaming=streaming,
        trust_remote_code=True
    )
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Metadata
    metadata_path = output_path / "metadata.json"
    
    # If resuming, load existing metadata
    start_idx = 0
    if metadata_path.exists():
        print("Found existing metadata, resuming tokenization...")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            start_idx = metadata.get('num_chunks', 0)
    
    # Output files
    tokens_file = output_path / "tokens.bin"
    
    # Determine number of processes
    if num_proc is None:
        num_proc = max(1, mp.cpu_count() // 2)
    
    print(f"Starting tokenization with {num_proc} processes...")
    print(f"Target sequence length: {max_seq_length:,} tokens")
    print(f"Output directory: {output_path}")
    
    chunk_count = start_idx
    total_tokens = 0
    
    # Process in batches
    batch_size = 1000
    buffer = []
    
    # Open file in append mode
    mode = 'ab' if start_idx > 0 else 'wb'
    
    with open(tokens_file, mode) as f:
        iterator = iter(dataset)
        
        if max_samples:
            from itertools import islice
            iterator = islice(iterator, max_samples)
        
        pbar = tqdm(desc="Tokenizing", unit=" docs")
        
        for example in iterator:
            # Tokenize single document
            text = example[text_column]
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            
            total_tokens += len(token_ids)
            
            # Add separator token between documents
            token_ids.append(tokenizer.eos_token_id)
            buffer.extend(token_ids)
            
            # When buffer is large enough, chunk it
            while len(buffer) >= max_seq_length:
                chunk = buffer[:max_seq_length]
                buffer = buffer[max_seq_length:]
                
                # Write chunk to file
                chunk_array = np.array(chunk, dtype=np.uint32)
                chunk_array.tofile(f)
                
                chunk_count += 1
                
                if chunk_count % 100 == 0:
                    pbar.set_postfix({
                        'chunks': chunk_count,
                        'total_tokens': f"{total_tokens:,}",
                        'buffer': len(buffer)
                    })
            
            pbar.update(1)
        
        pbar.close()
        
        # Handle remaining buffer (if any)
        if len(buffer) >= max_seq_length // 2:  # Only save if at least half full
            # Pad to max_seq_length
            buffer.extend([tokenizer.pad_token_id] * (max_seq_length - len(buffer)))
            chunk_array = np.array(buffer[:max_seq_length], dtype=np.uint32)
            chunk_array.tofile(f)
            chunk_count += 1
    
    # Save metadata
    metadata = {
        'dataset_name': dataset_name,
        'split': split,
        'text_column': text_column,
        'tokenizer_name': tokenizer_name,
        'max_seq_length': max_seq_length,
        'num_chunks': chunk_count,
        'total_tokens': total_tokens,
        'vocab_size': tokenizer.vocab_size,
        'eos_token_id': tokenizer.eos_token_id,
        'pad_token_id': tokenizer.pad_token_id,
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✓ Tokenization complete!")
    print(f"  - Total chunks: {chunk_count:,}")
    print(f"  - Total tokens: {total_tokens:,}")
    print(f"  - Tokens file: {tokens_file}")
    print(f"  - Metadata: {metadata_path}")


def verify_tokenized_data(output_dir: str):
    """Verify tokenized data integrity."""
    output_path = Path(output_dir)
    
    # Load metadata
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Load tokens
    tokens_file = output_path / "tokens.bin"
    tokens = np.fromfile(tokens_file, dtype=np.uint32)
    
    expected_size = metadata['num_chunks'] * metadata['max_seq_length']
    actual_size = len(tokens)
    
    print(f"Verification:")
    print(f"  - Expected tokens: {expected_size:,}")
    print(f"  - Actual tokens: {actual_size:,}")
    print(f"  - Match: {'✓' if expected_size == actual_size else '✗'}")
    
    if expected_size == actual_size:
        # Reshape and check first chunk
        tokens_reshaped = tokens.reshape(-1, metadata['max_seq_length'])
        print(f"  - Shape: {tokens_reshaped.shape}")
        print(f"  - First 10 tokens: {tokens_reshaped[0, :10].tolist()}")
        print(f"  - Last 10 tokens: {tokens_reshaped[0, -10:].tolist()}")


def main():
    parser = argparse.ArgumentParser(description="Tokenize Ultra-FineWeb dataset for Kimi Linear")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="semran1/megamath-web-pro",
        help="HuggingFace dataset name"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split"
    )
    parser.add_argument(
        "--text_column",
        type=str,
        default="text",
        help="Name of text column"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/tokenized_2m",
        help="Output directory for tokenized data"
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default="moonshotai/Kimi-Linear-48B-A3B-Base",
        help="Tokenizer name or path"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=2097152,  # 2M tokens
        help="Maximum sequence length"
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None,
        help="Number of processes for tokenization"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples (for testing)"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing tokenized data"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_tokenized_data(args.output_dir)
    else:
        save_tokenized_chunks(
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
