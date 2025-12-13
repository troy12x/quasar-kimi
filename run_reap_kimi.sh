#!/bin/bash
# Prune Kimi-Linear-48B-A3B-Base using REAP
# 50% compression (256 -> 128 experts)

# Configuration
# Configuration
# 1. HuggingFace Mirror (Enabled by default for stability)
export HF_ENDPOINT=https://hf-mirror.com

# 2. Model Selection
MODEL_ID="moonshotai/Kimi-Linear-48B-A3B-Base"
LOCAL_MODEL_PATH="$(pwd)/model_data/Kimi-Linear-48B-A3B-Base"

# Check for ROBUST local download (from download_model.py)
if [ -f "$LOCAL_MODEL_PATH/config.json" ]; then
    echo "ℹ️ Found downloaded model in: $LOCAL_MODEL_PATH"
    MODEL="$LOCAL_MODEL_PATH"
# Fallback check for current directory (must have config AND code)
elif [ -f "config.json" ] && [ -f "configuration_kimi.py" ]; then
    echo "ℹ️ Found valid local model (config + code) in current directory."
    echo "ℹ️ Using local model: $(pwd)"
    MODEL="$(pwd)"
else
     echo "⚠️ No complete local model found."
     echo "ℹ️ Will attempt download from HuggingFace cache: $MODEL_ID"
     MODEL="$MODEL_ID"
fi
DATASET="sumukshashidhar-archive/Ultra-FineWeb-100M"
COMPRESSION=0.5
PRUNING_METHOD="reap"
SEED=42
NUM_SAMPLES=15
# GPU configuration
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
FIRST_DEVICE=$(echo $CUDA_VISIBLE_DEVICES | cut -d',' -f1)
PORT=$((8000 + FIRST_DEVICE))
export REAP_OUTPUT_DIR="/shared/quasar_artifacts"
export TRITON_CACHE_DIR="/shared/.triton/cache"
mkdir -p "$TRITON_CACHE_DIR"

OUTPUT_FILE="observations_${NUM_SAMPLES}_cosine-seed_${SEED}.pt"
SERVER_LOG="reap-kimi-pruning.log"

echo "========================================="
echo "REAP Expert Pruning for Kimi Linear"
echo "========================================="

# 1. Check for REAP and Install
if [ ! -d "reap" ]; then
    echo "⚠️ REAP directory not found. Cloning from GitHub..."
    git clone https://github.com/cerebras/reap.git
    cd reap
    # Initialize submodules because pyproject.toml references third-party/ directories!
    git submodule update --init --recursive
    cd ..
else
    echo "✓ REAP directory found."
    # Ensure submodules are populated if they were missed
    if [ -d "reap/third-party" ] && [ -z "$(ls -A reap/third-party)" ]; then
         echo "⚠️ Populating missing submodules..."
         cd reap
         git submodule update --init --recursive
         cd ..
    fi
fi

echo "📦 Fixing dependencies..."
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
pip install scikit-learn

echo "🔧 Enforcing correct 'fla-core' installation (fixing import error)..."
# Force reinstall to ensure the correct package provides 'fla.layers'
# and to fix potential 'already installed' corruption or namespace conflicts

echo "📦 Installing REAP in editable mode..."
# We use --no-build-isolation to avoid metadata build failures often caused by missing build deps
#pip install -e reap --no-build-isolation

# 2. Run Pruning via Custom Runner
echo "🚀 Running pruning (incorporates Kimi/UltraFineWeb registration)..."
# We run from current directory, python path should pick up installed 'reap' package
python run_reap_custom.py \
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

echo "========================================="
echo "✓ Pruning process finished."
echo "Check $SERVER_LOG for details."
echo "========================================="
