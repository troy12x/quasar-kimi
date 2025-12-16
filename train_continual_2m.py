"""
Continual Pretraining Script for Kimi Linear: 1M → 2M Sequence Extension
Supports both DeepSpeed ZeRO-3 and FSDP for memory-efficient training.
"""

import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Optional, Dict
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    get_scheduler,
    set_seed,
)
from transformers.trainer_pt_utils import get_parameter_names

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs

import wandb
from tqdm import tqdm

# Local imports
from data_loader import create_dataloader
from memory_utils import (
    enable_gradient_checkpointing,
    print_memory_stats,
    clear_memory,
    calculate_model_memory,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Continual pretraining for Kimi Linear")
    
    # Model arguments
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="silx-ai/Quasar-2M-Base",
        help="Model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code",
    )
    
    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Directory with tokenized data",
    )
    parser.add_argument(
        "--use_curriculum",
        action="store_true",
        default=True,
        help="Use progressive sequence length curriculum",
    )
    
    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=10000,
        help="Total number of training steps",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=1,
        help="Batch size per device (should be 1 for 2M sequences)",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate (lower for continual pretraining)",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "constant"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Max gradient norm for clipping",
    )
    
    # Optimization arguments
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        default=True,
        help="Use gradient checkpointing",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        default=True,
        help="Use bfloat16 mixed precision",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use float16 mixed precision",
    )
    
    # DeepSpeed/FSDP
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="DeepSpeed config file",
    )
    parser.add_argument(
        "--fsdp",
        action="store_true",
        default=False,
        help="Use FSDP instead of DeepSpeed",
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every X steps",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X steps",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every X steps",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="kimi-linear-2m",
        help="Weights & Biases project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Weights & Biases run name",
    )
    
    # Misc
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    
    # Distributed training arguments (added by launchers like DeepSpeed)
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (set by launcher)",
    )
    
    args = parser.parse_args()
    return args


def setup_model_and_tokenizer(args, accelerator):
    """Load model and tokenizer."""
    print(f"Loading model from {args.model_name_or_path}...")
    
    # Load config
    config = AutoConfig.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
    )
    
    # Update config for 2M training
    if hasattr(config, 'model_max_length'):
        config.model_max_length = 2_097_152
        print(f"Updated model_max_length to {config.model_max_length:,}")
        
    # Fix for DeepSpeed ZeRO-3: padding_idx can be out of bounds if embedding is sharded
    if config.pad_token_id is not None:
        print(f"Clearing config.pad_token_id (was {config.pad_token_id}) to avoid ZeRO-3 sharding issues")
        config.pad_token_id = None
    
    # Disable KV cache for training
    if hasattr(config, 'use_cache'):
        config.use_cache = False
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    
    # Load model with ZeRO-3 Deferred Initialization
    if accelerator.is_main_process:
        print("Loading model with Deepspeed ZeRO-3 Init...")
        print_memory_stats("Before model load: ")

    import deepspeed
    # Use the DeepSpeed context to initialize weights directly on meta/partitioned directly
    # This prevents the full model from exploding in RAM
    ds_config_path = args.deepspeed
    
    # We need to ensure we have the config available
    if not ds_config_path:
        # Fallback if config not passed directly (rare in accelerate)
        print("Warning: No deepspeed config found in args, loading normally might OOM")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            config=config,
            trust_remote_code=args.trust_remote_code,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
    else:
        # Patch "auto" values in config because zero.Init doesn't auto-resolve them
        with open(ds_config_path, "r") as f:
            ds_config_dict = json.load(f)
        
        # Calculate train batch size
        total_batch_size = (
            args.per_device_train_batch_size * 
            args.gradient_accumulation_steps * 
            accelerator.num_processes
        )
        
        # Override auto with explicit values
        ds_config_dict["train_micro_batch_size_per_gpu"] = args.per_device_train_batch_size
        ds_config_dict["gradient_accumulation_steps"] = args.gradient_accumulation_steps
        ds_config_dict["train_batch_size"] = total_batch_size
        
        print(f"DeepSpeed Deferred Init with Batch Size: {total_batch_size}")

        with deepspeed.zero.Init(config_dict_or_path=ds_config_dict):
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name_or_path,
                config=config,
                trust_remote_code=args.trust_remote_code,
                torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
    
    if accelerator.is_main_process:
        print_memory_stats("After model load: ")
        
        # Calculate memory requirements
        mem_stats = calculate_model_memory(model, dtype=torch.bfloat16 if args.bf16 else torch.float16)
        print(f"\nModel Memory Estimation:")
        print(f"  Parameters: {mem_stats['param_count']:,}")
        print(f"  Model: {mem_stats['model_memory_gb']:.2f}GB")
        print(f"  Total (with optimizer): {mem_stats['total_memory_gb']:.2f}GB")
    
    # Enable gradient checkpointing
    if args.gradient_checkpointing:
        enable_gradient_checkpointing(model)
    
    return model, tokenizer, config


def setup_optimizer(args, model):
    """Setup optimizer with weight decay."""
    # Separate parameters that should/shouldn't have weight decay
    decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]
    
    import bitsandbytes as bnb
    print("Using 8-bit AdamW optimizer")
    
    optimizer = bnb.optim.AdamW8bit(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    
    return optimizer


def train(args):
    """Main training loop."""
    
    # Initialize accelerator
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    
    # When using DeepSpeed, don't let accelerator handle device placement
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision='bf16' if args.bf16 else ('fp16' if args.fp16 else 'no'),
        log_with="wandb" if args.wandb_project else None,
        kwargs_handlers=[ddp_kwargs],
        deepspeed_plugin=None if not args.deepspeed else None,  # Let DeepSpeed handle everything
    )
    
    # Set seed
    set_seed(args.seed)
    
    # Setup logging
    if accelerator.is_main_process:
        if args.wandb_project:
            run_name = args.wandb_run_name or f"kimi-2m-{args.seed}"
            accelerator.init_trackers(
                project_name=args.wandb_project,
                config=vars(args),
                init_kwargs={"wandb": {"name": run_name}},
            )
    
    # Load model and tokenizer
    model, tokenizer, config = setup_model_and_tokenizer(args, accelerator)
    
    # Setup dataloader
    if accelerator.is_main_process:
        print(f"\nSetting up dataloader from {args.data_dir}...")
    
    curriculum_dataloader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.per_device_train_batch_size,
        rank=accelerator.process_index,
        world_size=accelerator.num_processes,
        seed=args.seed,
        num_workers=0,
        use_curriculum=args.use_curriculum,
    )
    
    # Extract the actual dataloader for accelerator.prepare()
    if hasattr(curriculum_dataloader, 'dataloader'):
        # It's a ProgressiveCurriculumDataLoader wrapper
        train_dataloader = curriculum_dataloader.dataloader
    else:
        # It's already a DataLoader
        train_dataloader = curriculum_dataloader
    
    # Setup optimizer
    optimizer = setup_optimizer(args, model)
    
    # Setup scheduler
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_steps,
    )
    
    # Prepare everything with accelerator
    # For large models, we need to avoid accelerator.prepare() moving the model to GPU
    # Instead, manually move to device and wrap with DDP
    # Prepare everything with accelerator
    # For ZeRO-3, the model must be prepared by accelerator to handle sharding correctly
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Store curriculum wrapper for later use
    if hasattr(curriculum_dataloader, 'dataloader'):
        curriculum_dataloader.dataloader = train_dataloader
    
    # Resume from checkpoint if specified
    starting_step = 0
    if args.resume_from_checkpoint:
        if accelerator.is_main_process:
            print(f"Resuming from checkpoint: {args.resume_from_checkpoint}")
        
        accelerator.load_state(args.resume_from_checkpoint)
        
        # Extract step number from checkpoint path
        checkpoint_name = os.path.basename(args.resume_from_checkpoint)
        if "step" in checkpoint_name:
            starting_step = int(checkpoint_name.split("step")[1].split("-")[0])
    
    # Training loop
    if accelerator.is_main_process:
        print(f"\n{'='*60}")
        print(f"Starting training from step {starting_step}")
        print(f"Total steps: {args.num_train_steps}")
        print(f"Effective batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
        print(f"{'='*60}\n")
    
    model.train()
    global_step = starting_step
    
    # Progress bar
    progress_bar = tqdm(
        total=args.num_train_steps,
        initial=starting_step,
        disable=not accelerator.is_local_main_process,
        desc="Training",
    )
    
    train_iterator = iter(train_dataloader)
    
    while global_step < args.num_train_steps:
        try:
            # Update curriculum if using progressive training
            if args.use_curriculum and hasattr(curriculum_dataloader, 'update_curriculum'):
                curriculum_dataloader.update_curriculum(global_step)
                # Reset iterator after curriculum update
                train_iterator = iter(train_dataloader)
            
            try:
                batch = next(train_iterator)
            except StopIteration:
                # Reset iterator if dataset is exhausted
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)
            
            with accelerator.accumulate(model):
                # Forward pass
                outputs = model(**batch)
                loss = outputs.loss
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
        except Exception as e:
            if accelerator.is_main_process:
                print(f"\nError at step {global_step}:")
                print(traceback.format_exc())
            raise
        
        # Update step counter
        if accelerator.sync_gradients:
            global_step += 1
            progress_bar.update(1)
            
            # Logging
            if global_step % args.logging_steps == 0:
                metrics = {
                    "loss": loss.item(),
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "step": global_step,
                }
                
                # Add sequence length info
                if hasattr(curriculum_dataloader, 'get_current_seq_length'):
                    metrics["sequence_length"] = curriculum_dataloader.get_current_seq_length()
                
                if accelerator.is_main_process:
                    progress_bar.set_postfix(metrics)
                    
                    if args.wandb_project:
                        accelerator.log(metrics, step=global_step)
            
            # Save checkpoint
            if global_step % args.save_steps == 0:
                output_dir = Path(args.output_dir) / f"checkpoint-step{global_step}"
                
                if accelerator.is_main_process:
                    print(f"\nSaving checkpoint to {output_dir}")
                
                accelerator.save_state(output_dir)
                
                # Save tokenizer and config
                if accelerator.is_main_process:
                    tokenizer.save_pretrained(output_dir)
                    config.save_pretrained(output_dir)
    
    # Final checkpoint
    if accelerator.is_main_process:
        print(f"\nTraining complete! Saving final checkpoint...")
        
        final_dir = Path(args.output_dir) / "final"
        accelerator.save_state(final_dir)
        tokenizer.save_pretrained(final_dir)
        config.save_pretrained(final_dir)
        
        print(f"Final checkpoint saved to {final_dir}")
    
    # End tracking
    if args.wandb_project:
        accelerator.end_training()
    
    progress_bar.close()


if __name__ == "__main__":
    args = parse_args()
    try:
        train(args)
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR in training process:")
        print(f"{'='*60}")
        print(traceback.format_exc())
        print(f"{'='*60}\n")
        sys.exit(1)
