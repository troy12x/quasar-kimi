
import os
from huggingface_hub import upload_folder, login

# Configuration
REPO_ID = "eyad-silx/Quasar-40B"
TOKEN = "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"
MODEL_DIR = "/shared/quasar_artifacts/Kimi-Linear-48B-A3B-Base/Ultra-FineWeb-100M/pruned_models/reap-seed_42-0.50"

print(f"🚀 Preparing to upload pruned model to {REPO_ID}...")
print(f"📂 Source directory: {MODEL_DIR}")

if not os.path.exists(MODEL_DIR):
    print(f"❌ Error: Model directory not found at {MODEL_DIR}")
    exit(1)

# Login
print("🔑 Logging in to Hugging Face...")
login(token=TOKEN)

# Upload
print("⬆️  Starting upload (this may take a while depending on size)...")
try:
    upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model"
    )
    print("✅ Upload complete! Access your model here:")
    print(f"https://huggingface.co/{REPO_ID}")
except Exception as e:
    print(f"❌ Upload failed: {e}")
