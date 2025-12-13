#!/bin/bash
# Prune Kimi-Linear-48B-A3B-Base using REAP
# Reduces 256 experts to 128 (50% compression) with near-lossless performance

# Configuration
MODEL="moonshotai/Kimi-Linear-48B-A3B-Base"
DATASET="theblackcat102/evol-codealpaca-v1"  # REAP default calibration dataset
COMPRESSION=0.5  # 50% pruning (256 → 128 experts)
PRUNING_METHOD="reap"
SEED=42
NUM_SAMPLES=1024

# GPU configuration
export CUDA_VISIBLE_DEVICES=0  # Adjust based on your setup
FIRST_DEVICE=0
PORT=8000

OUTPUT_FILE="observations_${NUM_SAMPLES}_cosine-seed_${SEED}.pt"
SERVER_LOG="reap-kimi-pruning.log"

echo "========================================="
echo "REAP Expert Pruning for Kimi Linear"
echo "========================================="
echo "Model: $MODEL"
echo "Dataset: $DATASET"
echo "Compression: ${COMPRESSION} (256 → 128 experts)"
echo "Method: $PRUNING_METHOD"
echo "========================================="

# Step 1: Add Kimi support to REAP
echo "Adding Kimi model support to REAP..."
python add_kimi_to_reap.py

# Step 2: Run REAP pruning
echo "Running expert pruning (this will take ~30-60 minutes)..."
cd reap

python /reap/prune.py \
    --model-name "$MODEL" \
    --dataset-name "$DATASET" \
    --compression-ratio $COMPRESSION \
    --prune-method $PRUNING_METHOD \
    --profile false \
    --vllm_port $PORT \
    --server-log-file-name "$SERVER_LOG" \
    --do-eval false \
    --distance_measure cosine \
    --seed $SEED \
    --output_file_name "${OUTPUT_FILE}" \
    --singleton_super_experts false \
    --singleton_outlier_experts false \
    --samples_per_category $NUM_SAMPLES \
    --record_pruning_metrics_only true

# Extract pruned model directory
SHORT_MODEL="Kimi-Linear-48B-A3B-Base"
SHORT_DATASET="evol-codealpaca-v1"
PRUNED_MODEL_DIR="artifacts/${SHORT_MODEL}/${SHORT_DATASET}/pruned_models/reap-seed_${SEED}-${COMPRESSION}"

echo "========================================="
echo "✓ Pruning complete!"
echo "Pruned model saved to: $PRUNED_MODEL_DIR"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. The pruned model is now 128 experts (was 256)"
echo "2. Use this pruned model for continual pretraining"
echo "3. Training will be ~2× faster and use ~50% less memory"
echo ""
echo "To use the pruned model in training, update train_continual_2m.py:"
echo "  --model_name_or_path=\"$PRUNED_MODEL_DIR\""
