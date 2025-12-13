
import os
import sys
import time

# ---------------------------------------------------------------------------
# FORCE MIRROR CONFIGURATION
# ---------------------------------------------------------------------------
# As verified with fix_tokenizer.py, this solves the connection stall.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("🌐 Forced HF_ENDPOINT=https://hf-mirror.com for stability.")

# Import after setting env var
from huggingface_hub import snapshot_download

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Base"
LOCAL_DIR = os.path.join(os.getcwd(), "model_data", "Kimi-Linear-48B-A3B-Base")
MAX_RETRIES = 10
WORKERS = 4  # Conservative worker count for stability

print(f"🚀 Starting Full Model Download: {MODEL_ID}")
print(f"📂 Target: {LOCAL_DIR}")

# ---------------------------------------------------------------------------
# DOWNLOAD LOOP WITH RETRY
# ---------------------------------------------------------------------------
for i in range(1, MAX_RETRIES + 1):
    try:
        print(f"\n⬇️  Attempt {i}/{MAX_RETRIES}...")
        
        path = snapshot_download(
            repo_id=MODEL_ID,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,  # Ensure real files
            resume_download=True,          # Resume partial downloads
            max_workers=WORKERS,
            etag_timeout=60
        )
        
        print(f"\n✅ SUCCESS! Full model downloaded to:\n{path}")
        print("\nReady to prune! Run: bash run_reap_kimi.sh")
        sys.exit(0)
        
    except Exception as e:
        print(f"⚠️  Error during download: {e}")
        if i < MAX_RETRIES:
            wait_time = 15
            print(f"⏳ Connection stalled. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        else:
            print("\n❌ Failed after multiple attempts.")
            sys.exit(1)
