
import os
import sys

# FORCE MIRROR: It seems your connection to huggingface.co is unstable/blocked.
# This often fixes "stuck" downloads.
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
print("🌐 Forced HF_ENDPOINT=https://hf-mirror.com for stability.")

from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

MODEL_ID = "moonshotai/Kimi-Linear-48B-A3B-Base"
LOCAL_DIR = os.path.join(os.getcwd(), "model_data", "Kimi-Linear-48B-A3B-Base")

FILES_TO_FETCH = [
    "tiktoken.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "tokenization_kimi.py",
    "config.json",
    "configuration_kimi.py"
]

print(f"🚀 Downloading tokenizer files for {MODEL_ID}...")
os.makedirs(LOCAL_DIR, exist_ok=True)

for filename in FILES_TO_FETCH:
    try:
        print(f"⬇️  Fetching {filename}...")
        hf_hub_download(
            repo_id=MODEL_ID,
            filename=filename,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            force_download=True  # Ensure we get a good copy
        )
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")

print("\n✅ Tokenizer files downloaded.")

# Verification Step
print("\n🧪 Verifying tokenizer loading...")
try:
    # Explicitly pointing to the vocab file helps if auto-discovery fails
    vocab_path = os.path.join(LOCAL_DIR, "tiktoken.model")
    if not os.path.exists(vocab_path):
        print(f"⚠️  WARNING: {vocab_path} does not exist!")
    
    # Try loading normally first
    print("Attempt 1: AutoTokenizer.from_pretrained(LOCAL_DIR)")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_DIR, trust_remote_code=True)
    print("✅ Successfully loaded tokenizer!")

except Exception as e:
    print(f"❌ Load failed: {e}")
    print("\nAttempt 2: Checking if explicitly passing vocab_file helps...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            LOCAL_DIR, 
            trust_remote_code=True,
            vocab_file=os.path.join(LOCAL_DIR, "tiktoken.model")
        )
        print("✅ Successfully loaded tokenizer with explicit vocab_file arg!")
    except Exception as e2:
        print(f"❌ Attempt 2 failed also: {e2}")

