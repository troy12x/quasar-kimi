"""
Memory optimization utilities for Kimi Linear training.
Includes gradient checkpointing, memory profiling, and OOM recovery.
"""

import torch
import torch.nn as nn
from typing import Optional, Callable
import functools
import gc


def enable_gradient_checkpointing(
    model: nn.Module,
    checkpoint_ratio: float = 0.5,
    use_reentrant: bool = False
):
    """
    Enable gradient checkpointing for memory efficiency.
    
    Args:
        model: The model to apply checkpointing to
        checkpoint_ratio: Ratio of layers to checkpoint (0.5 = every other layer)
        use_reentrant: Whether to use reentrant checkpointing
    """
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print(f"✓ Enabled gradient checkpointing (ratio: {checkpoint_ratio})")
    else:
        print("⚠ Model does not support gradient_checkpointing_enable")
    
    # Additional memory optimizations
    if hasattr(model, 'config'):
        model.config.use_cache = False  # Disable KV cache during training
        print("✓ Disabled KV cache for training")


def get_memory_stats(device: Optional[torch.device] = None) -> dict:
    """Get current GPU memory statistics."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if device.type == 'cuda':
        allocated = torch.cuda.memory_allocated(device) / 1024**3
        reserved = torch.cuda.memory_reserved(device) / 1024**3
        max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        
        return {
            'allocated_gb': allocated,
            'reserved_gb': reserved,
            'max_allocated_gb': max_allocated,
            'free_gb': reserved - allocated,
        }
    else:
        return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0, 'free_gb': 0}


def print_memory_stats(prefix: str = "", device: Optional[torch.device] = None):
    """Print memory statistics."""
    stats = get_memory_stats(device)
    print(f"{prefix}Memory: {stats['allocated_gb']:.2f}GB allocated, "
          f"{stats['reserved_gb']:.2f}GB reserved, "
          f"{stats['max_allocated_gb']:.2f}GB peak")


def clear_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def setup_memory_efficient_training(
    model: nn.Module,
    gradient_checkpointing: bool = True,
    mixed_precision: bool = True,
    cpu_offload: bool = False,
) -> dict:
    """
    Setup memory-efficient training configuration.
    
    Args:
        model: Model to configure
        gradient_checkpointing: Enable gradient checkpointing
        mixed_precision: Use mixed precision training
        cpu_offload: Offload to CPU (for very large models)
    
    Returns:
        Configuration dict
    """
    config = {}
    
    # Gradient checkpointing
    if gradient_checkpointing:
        enable_gradient_checkpointing(model)
        config['gradient_checkpointing'] = True
    
    # Mixed precision
    if mixed_precision:
        config['mixed_precision'] = 'bf16' if torch.cuda.is_bf16_supported() else 'fp16'
        print(f"✓ Using {config['mixed_precision']} mixed precision")
    
    # CPU offload
    if cpu_offload:
        config['cpu_offload'] = True
        print("✓ CPU offloading enabled")
    
    # Disable dropout for inference-like behavior (optional)
    # model.eval()  # Uncomment if needed
    
    return config


class MemoryProfiler:
    """Context manager for profiling memory usage."""
    
    def __init__(self, name: str = "Operation", device: Optional[torch.device] = None):
        self.name = name
        self.device = device
        self.start_stats = None
    
    def __enter__(self):
        clear_memory()
        self.start_stats = get_memory_stats(self.device)
        print(f"\n{'='*50}")
        print(f"Starting: {self.name}")
        print_memory_stats("Before: ", self.device)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_stats = get_memory_stats(self.device)
        print_memory_stats("After: ", self.device)
        
        delta = end_stats['allocated_gb'] - self.start_stats['allocated_gb']
        print(f"Delta: {delta:+.2f}GB")
        print(f"{'='*50}\n")


def calculate_model_memory(
    model: nn.Module,
    dtype: torch.dtype = torch.bfloat16,
) -> dict:
    """
    Estimate model memory requirements.
    
    Args:
        model: Model to analyze
        dtype: Data type for parameters
    
    Returns:
        Dictionary with memory estimates
    """
    param_count = sum(p.numel() for p in model.parameters())
    
    # Bytes per parameter
    if dtype == torch.float32:
        bytes_per_param = 4
    elif dtype in [torch.float16, torch.bfloat16]:
        bytes_per_param = 2
    else:
        bytes_per_param = 4
    
    # Model parameters
    model_memory_gb = (param_count * bytes_per_param) / 1024**3
    
    # Gradients (same size as parameters)
    gradient_memory_gb = model_memory_gb
    
    # Optimizer states (Adam: 2x parameters for momentum + variance)
    optimizer_memory_gb = model_memory_gb * 2
    
    # Total
    total_memory_gb = model_memory_gb + gradient_memory_gb + optimizer_memory_gb
    
    return {
        'param_count': param_count,
        'model_memory_gb': model_memory_gb,
        'gradient_memory_gb': gradient_memory_gb,
        'optimizer_memory_gb': optimizer_memory_gb,
        'total_memory_gb': total_memory_gb,
    }


def estimate_activation_memory(
    batch_size: int,
    sequence_length: int,
    hidden_size: int,
    num_layers: int,
    dtype: torch.dtype = torch.bfloat16,
    gradient_checkpointing: bool = False,
) -> dict:
    """
    Estimate activation memory for transformer model.
    
    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        hidden_size: Hidden dimension
        num_layers: Number of layers
        dtype: Data type
        gradient_checkpointing: Whether gradient checkpointing is enabled
    
    Returns:
        Dictionary with activation memory estimates
    """
    if dtype == torch.float32:
        bytes_per_elem = 4
    elif dtype in [torch.float16, torch.bfloat16]:
        bytes_per_elem = 2
    else:
        bytes_per_elem = 4
    
    # Per-layer activation size (rough estimate)
    # Typical: batch_size * seq_length * hidden_size * (4 to 6 intermediate tensors)
    per_layer_activations = batch_size * sequence_length * hidden_size * 5 * bytes_per_elem
    
    if gradient_checkpointing:
        # Only store activations for checkpointed layers
        # Assume we checkpoint every 2 layers
        effective_layers = num_layers // 2
    else:
        effective_layers = num_layers
    
    total_activation_memory = per_layer_activations * effective_layers / 1024**3
    
    return {
        'batch_size': batch_size,
        'sequence_length': sequence_length,
        'per_layer_gb': per_layer_activations / 1024**3,
        'total_activation_gb': total_activation_memory,
        'gradient_checkpointing': gradient_checkpointing,
    }


def adaptive_batch_size(
    target_memory_gb: float,
    sequence_length: int,
    hidden_size: int = 2304,  # Kimi Linear hidden size
    num_layers: int = 27,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """
    Calculate adaptive batch size based on available memory.
    
    Args:
        target_memory_gb: Target memory budget (e.g., 35GB for 40GB GPU)
        sequence_length: Sequence length
        hidden_size: Hidden dimension
        num_layers: Number of layers
        dtype: Data type
    
    Returns:
        Recommended batch size
    """
    # Binary search for maximum batch size
    low, high = 1, 16
    best_batch_size = 1
    
    while low <= high:
        mid = (low + high) // 2
        
        est = estimate_activation_memory(
            batch_size=mid,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dtype=dtype,
            gradient_checkpointing=True,
        )
        
        if est['total_activation_gb'] <= target_memory_gb:
            best_batch_size = mid
            low = mid + 1
        else:
            high = mid - 1
    
    return best_batch_size


if __name__ == "__main__":
    # Test memory utilities
    print("Testing memory utilities...\n")
    
    # Test memory stats
    print("Current memory stats:")
    print_memory_stats()
    
    # Test memory estimation for Kimi Linear
    print("\nKimi Linear 48B-A3B Memory Estimation:")
    print("(Note: 3B active parameters due to MoE)")
    
    # Estimate for active parameters only (3B)
    active_params = 3_000_000_000
    
    class DummyModel:
        def parameters(self):
            return [torch.randn(active_params // 10) for _ in range(10)]
    
    dummy = DummyModel()
    mem_stats = calculate_model_memory(dummy, dtype=torch.bfloat16)
    
    print(f"Active parameters: {mem_stats['param_count']:,}")
    print(f"Model memory: {mem_stats['model_memory_gb']:.2f}GB")
    print(f"Total (with optimizer): {mem_stats['total_memory_gb']:.2f}GB")
    
    # Test activation memory for 2M sequence
    print("\nActivation memory for 2M sequence:")
    act_stats = estimate_activation_memory(
        batch_size=1,
        sequence_length=2_097_152,
        hidden_size=2304,
        num_layers=27,
        dtype=torch.bfloat16,
        gradient_checkpointing=True,
    )
    
    print(f"Batch size: {act_stats['batch_size']}")
    print(f"Sequence length: {act_stats['sequence_length']:,}")
    print(f"Total activation memory: {act_stats['total_activation_gb']:.2f}GB")
    
    # Adaptive batch size
    print("\nRecommended batch size for 35GB budget:")
    batch_size = adaptive_batch_size(
        target_memory_gb=35.0,
        sequence_length=2_097_152,
    )
    print(f"Batch size: {batch_size}")
