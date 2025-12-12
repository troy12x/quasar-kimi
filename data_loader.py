"""
Data loader for pre-tokenized Kimi Linear training data.
Supports progressive sequence length curriculum and memory-efficient loading.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from pathlib import Path
from typing import Optional, Dict


class TokenizedDataset(IterableDataset):
    """
    Memory-mapped dataset for pre-tokenized data.
    Supports progressive sequence length training.
    """
    
    def __init__(
        self,
        data_dir: str,
        sequence_length: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
    ):
        """
        Args:
            data_dir: Directory containing tokenized data
            sequence_length: Current sequence length (for curriculum)
            rank: Process rank for distributed training
            world_size: Total number of processes
            seed: Random seed for shuffling
            shuffle: Whether to shuffle data
        """
        super().__init__()
        
        self.data_dir = Path(data_dir)
        self.sequence_length = sequence_length
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        
        # Load metadata
        metadata_path = self.data_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.max_seq_length = self.metadata['max_seq_length']
        self.num_chunks = self.metadata['num_chunks']
        
        # Memory-map the tokens file
        tokens_file = self.data_dir / "tokens.bin"
        self.tokens_mmap = np.memmap(
            tokens_file,
            dtype=np.uint32,
            mode='r',
            shape=(self.num_chunks, self.max_seq_length)
        )
        
        # Calculate chunks per process
        self.chunks_per_rank = self.num_chunks // self.world_size
        self.start_chunk = self.rank * self.chunks_per_rank
        self.end_chunk = self.start_chunk + self.chunks_per_rank
        
        print(f"[Rank {self.rank}] Loaded {self.num_chunks:,} chunks")
        print(f"[Rank {self.rank}] Processing chunks {self.start_chunk:,} to {self.end_chunk:,}")
        print(f"[Rank {self.rank}] Current sequence length: {self.sequence_length:,}")
    
    def __iter__(self):
        """Iterate over chunks."""
        # Create index array for this rank
        indices = np.arange(self.start_chunk, self.end_chunk)
        
        if self.shuffle:
            # Shuffle with seed for reproducibility
            rng = np.random.RandomState(self.seed + self.rank)
            rng.shuffle(indices)
        
        for idx in indices:
            # Load chunk from memory map
            chunk = self.tokens_mmap[idx]
            
            # Truncate to current sequence length (for curriculum)
            chunk = chunk[:self.sequence_length]
            
            # Convert to tensor
            input_ids = torch.from_numpy(chunk.astype(np.int64))
            
            # Create labels (shifted by 1 for causal LM)
            labels = input_ids.clone()
            
            yield {
                'input_ids': input_ids,
                'labels': labels,
                'attention_mask': torch.ones_like(input_ids),
            }
    
    def set_sequence_length(self, new_length: int):
        """Update sequence length for curriculum learning."""
        assert new_length <= self.max_seq_length, \
            f"New length {new_length} exceeds max {self.max_seq_length}"
        
        old_length = self.sequence_length
        self.sequence_length = new_length
        print(f"[Rank {self.rank}] Sequence length: {old_length:,} → {new_length:,}")


class ProgressiveCurriculumDataLoader:
    """
    DataLoader wrapper that supports progressive sequence length curriculum.
    Automatically transitions from 1M → 1.5M → 2M tokens.
    """
    
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        curriculum_steps: Dict[int, int],
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        num_workers: int = 0,
    ):
        """
        Args:
            data_dir: Directory containing tokenized data
            batch_size: Batch size (should be 1 for long sequences)
            curriculum_steps: Dict mapping step to sequence length
                              e.g., {0: 1048576, 1000: 1572864, 2000: 2097152}
            rank: Process rank
            world_size: Total processes
            seed: Random seed
            num_workers: DataLoader workers
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.curriculum_steps = sorted(curriculum_steps.items())  # Sort by step
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.num_workers = num_workers
        
        # Start with first sequence length
        initial_length = self.curriculum_steps[0][1]
        
        # Create dataset
        self.dataset = TokenizedDataset(
            data_dir=data_dir,
            sequence_length=initial_length,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
        
        # Create dataloader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
        
        self.current_curriculum_idx = 0
        self.current_step = 0
    
    def update_curriculum(self, step: int):
        """Update sequence length based on current training step."""
        self.current_step = step
        
        # Find the appropriate curriculum stage
        target_length = self.curriculum_steps[0][1]  # Default to first
        
        for curriculum_step, seq_length in self.curriculum_steps:
            if step >= curriculum_step:
                target_length = seq_length
            else:
                break
        
        # Update if changed
        if target_length != self.dataset.sequence_length:
            self.dataset.set_sequence_length(target_length)
            
            # Recreate dataloader to reset iteration
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            )
    
    def __iter__(self):
        """Iterate over dataloader."""
        return iter(self.dataloader)
    
    def get_current_seq_length(self) -> int:
        """Get current sequence length."""
        return self.dataset.sequence_length


def create_dataloader(
    data_dir: str,
    batch_size: int,
    rank: int = 0,
    world_size: int = 1,
    seed: int = 42,
    num_workers: int = 0,
    use_curriculum: bool = True,
    curriculum_steps: Optional[Dict[int, int]] = None,
) -> ProgressiveCurriculumDataLoader:
    """
    Factory function to create dataloader.
    
    Args:
        data_dir: Directory containing tokenized data
        batch_size: Batch size per device
        rank: Process rank
        world_size: Total processes
        seed: Random seed
        num_workers: Number of workers
        use_curriculum: Whether to use progressive curriculum
        curriculum_steps: Custom curriculum dict {step: seq_length}
                         If None and use_curriculum=True, uses research-based defaults
    
    Returns:
        DataLoader instance
    """
    if use_curriculum:
        # Research-based curriculum for 1M→2M extrapolation
        # Based on "Extending Context Length via Length Extrapolation" principles:
        # - Train on base length for stability
        # - Gradual 25% increases to allow adaptation
        # - Each stage sees ~10-20B tokens before increasing
        # 
        # Assumes effective batch size of 256-512 (global)
        # At batch_size=256: 39M tokens/step, ~256 steps = 10B tokens per stage
        # At batch_size=512: 78M tokens/step, ~128 steps = 10B tokens per stage
        #
        # Conservative approach: more tokens at each length for stability
        if curriculum_steps is None:
            curriculum_steps = {
                0: 1_048_576,        # 1M - Start with pretrained length (train ~20-30B tokens)
                2500: 1_310_720,     # 1.25M - 25% increase (train ~15-20B tokens)  
                5000: 1_572_864,     # 1.5M - another 25% (train ~15-20B tokens)
                7500: 1_835_008,     # 1.75M - gradual approach (train ~10-15B tokens)
                10000: 2_097_152,    # 2M - final target (train to convergence)
            }
        
        return ProgressiveCurriculumDataLoader(
            data_dir=data_dir,
            batch_size=batch_size,
            curriculum_steps=curriculum_steps,
            rank=rank,
            world_size=world_size,
            seed=seed,
            num_workers=num_workers,
        )
    else:
        # Fixed 2M length - only use if you have a model already trained at 2M
        dataset = TokenizedDataset(
            data_dir=data_dir,
            sequence_length=2_097_152,
            rank=rank,
            world_size=world_size,
            seed=seed,
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
        )
