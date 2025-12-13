
#!/bin/bash
set -e

echo "🚀 Starting Optimized Flash Attention Installation..."
echo "ℹ️  PyTorch 2.9.1+cu128 detected - Source build required (no wheels available)."

# 1. Install Build Dependencies
echo "📦 Installing build accelerators (Ninja)..."
pip install ninja packaging

# 2. Configure Build Environment
# Use all available cores for compilation
export MAX_JOBS=$(nproc)
export RAM_LIMIT_GB=32 # Prevent OOM checks if needed, but mostly rely on jobs
echo "⚙️  Setting MAX_JOBS=$MAX_JOBS for parallel compilation."

# 3. Install Flash Attention
# --no-build-isolation: uses current env packages (like ninja) instead of fresh empty venv
echo "🔥 Building Flash Attention (this will still take a few minutes, but faster)..."
pip install flash-attn --no-build-isolation

echo "✅ Flash Attention Installation Complete!"
