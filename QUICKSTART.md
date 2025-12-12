# Quick Reference: MegaMath-Web-Pro Training

## Dataset: semran1/megamath-web-pro
- **Size:** 40-100B tokens (manageable!)
- **Split:** train
- **Column:** text
- **Focus:** High-quality math and reasoning data

## Time & Cost Estimates (4x L20)

| Tokens to train | Steps needed | Time | Cost |
|----------------|--------------|------|------|
| 40B tokens | ~2,350 steps | ~20-40 hours | ~$500-1,000 |
| 60B tokens | ~3,500 steps | ~30-60 hours | ~$750-1,500 |
| 100B tokens | ~5,850 steps | ~50-100 hours | ~$1,250-2,500 |

**Recommended: Train on full 40-100B for best results**

## Commands

**Tokenize:**
```bash
python tokenize_dataset.py \
  --dataset_name semran1/megamath-web-pro \
  --split train \
  --text_column text \
  --output_dir ./data/tokenized_2m
```

**Train (4x L20):**
```bash
accelerate launch --num_processes 4 --mixed_precision bf16 \
  train_continual_2m.py \
  --data_dir ./data/tokenized_2m \
  --deepspeed deepspeed_config_l20.json \
  --output_dir ./checkpoints/kimi-2m \
  --wandb_project kimi-linear-2m
```

**Train (2x H100 - faster):**
```bash
accelerate launch --num_processes 2 --mixed_precision bf16 \
  train_continual_2m.py \
  --data_dir ./data/tokenized_2m \
  --deepspeed deepspeed_config_h100.json \
  --output_dir ./checkpoints/kimi-2m \
  --wandb_project kimi-linear-2m
```

## Why MegaMath-Web-Pro?

✅ **Manageable size** - 40-100B vs 2T tokens  
✅ **High quality** - Math and reasoning focus  
✅ **Complete in days** - Not weeks  
✅ **Cost effective** - $500-2,500 vs $10k+  
✅ **Sufficient for length extension** - Don't need 2T tokens for continual pretraining
