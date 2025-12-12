# Kimi Linear 2M Continual Pretraining

Production-ready training scripts to extend [Kimi-Linear-48B-A3B](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Base) from 1M to 2M sequence length using MegaMath-Web-Pro dataset (40-100B tokens).

## Quick Start

### 1. Install Dependencies

```bash
# Install flash-linear-attention with KDA kernel (CRITICAL!)
pip install flash-linear-attention

# Install other dependencies
pip install -r requirements.txt
```

### 2. Tokenize Dataset

```bash
# Tokenize MegaMath-Web-Pro (40-100B tokens - much faster than Ultra-FineWeb!)
python tokenize_dataset.py \
  --dataset_name semran1/megamath-web-pro \
  --split train \
  --text_column text \
  --output_dir ./data/tokenized_2m \
  --max_seq_length 2097152

# Verify tokenized data
python tokenize_dataset.py --output_dir ./data/tokenized_2m --verify
```

### 3. Run Training

**For 2x H100 (80GB each):**
```bash
accelerate launch --config_file accelerate_config_h100.yaml \
  train_continual_2m.py \
  --data_dir ./data/tokenized_2m \
  --output_dir ./checkpoints/kimi-2m-h100 \
  --deepspeed deepspeed_config_h100.json \
  --num_train_steps 30000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 256 \
  --learning_rate 1e-5 \
  --warmup_steps 1000 \
  --gradient_checkpointing \
  --bf16 \
  --wandb_project kimi-linear-2m
```

**For 4x L20 (40GB each):**
```bash
accelerate launch --config_file accelerate_config_l20.yaml \
  train_continual_2m.py \
  --data_dir ./data/tokenized_2m \
  --output_dir ./checkpoints/kimi-2m-l20 \
  --deepspeed deepspeed_config_l20.json \
  --num_train_steps 30000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 512 \
  --learning_rate 1e-5 \
  --warmup_steps 1000 \
  --gradient_checkpointing \
  --bf16 \
  --wandb_project kimi-linear-2m
```

## Training Details

### Progressive Curriculum

The training uses research-based length extrapolation:
- **Step 0-2500:** 1M tokens (existing capability)
- **Step 2500-5000:** 1.25M tokens (+25% increase)
- **Step 5000-7500:** 1.5M tokens 
- **Step 7500-10000:** 1.75M tokens
- **Step 10000-30000:** 2M tokens (final target)

Each stage sees 10-20B tokens for stable adaptation.

### Hardware Requirements

| Setup | Memory/GPU | Global Batch | Tokens/Step | Speed |
|-------|-----------|--------------|-------------|-------|
| 2x H100 | 80GB | 512 sequences | ~1B tokens | ~1.5 steps/sec |
| 4x L20 | 40GB | 2048 sequences | ~4B tokens | ~0.7 steps/sec |

### Training Parameters

- **Learning Rate:** 1e-5 (conservative for continual pretraining)
- **Weight Decay:** 0.1 (standard for large models)
- **Warmup:** 1000 steps (~1-4B tokens depending on hardware)
- **Total Training:** ~100-150B tokens
- **Mixed Precision:** BF16
- **Optimization:** DeepSpeed ZeRO-3 + CPU offloading

## File Structure

```
├── train_continual_2m.py          # Main training script
├── tokenize_dataset.py            # Dataset preprocessing
├── data_loader.py                 # Progressive curriculum dataloader
├── memory_utils.py                # Memory profiling utilities
├── deepspeed_config_h100.json     # DeepSpeed config for H100
├── deepspeed_config_l20.json      # DeepSpeed config for L20
├── config_h100_2x.yaml            # Training config for H100
├── config_l20_4x40gb.yaml         # Training config for L20
└── requirements.txt               # Dependencies
```

## Monitoring

Training metrics are logged to Weights & Biases. Key metrics to watch:
- **Loss trajectory:** Should remain smooth during length transitions
- **GPU memory:** Should stay under 75GB (H100) or 38GB (L20)
- **Throughput:** Steps/second should remain consistent

## Resume Training

```bash
python train_continual_2m.py \
  --resume_from_checkpoint ./checkpoints/kimi-2m-h100/checkpoint-step5000 \
  [... other args ...]
```

## Cost Estimate

- **2x H100:** ~$200-300 for 30k steps (~20-30 hours)
- **4x L20:** ~$150-250 for 30k steps (~40-50 hours)

## Notes

- **KDA Kernel:** Uses official `flash-linear-attention` library (~2× faster than naive implementation)
- **No RoPE adjustment needed:** Model uses NoPE (No Positional Encoding) design
- **MoE efficiency:** Only 3B/48B params active → much more memory-efficient than dense models
