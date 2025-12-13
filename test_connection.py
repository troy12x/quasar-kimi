
import os
import sys

print("🌐 Testing connection to HuggingFace...")

try:
    from huggingface_hub import hf_hub_download
    print(f"hf_hub_download imported successfully.")
    
    # Try to fetch a tiny file (config.json) from the Kimi repo
    # This mimics exactly what transformers does
    print("⬇️ Attempting to download config.json from moonshotai/Kimi-Linear-48B-A3B-Base...")
    
    path = hf_hub_download(
        repo_id="moonshotai/Kimi-Linear-48B-A3B-Base",
        filename="config.json",
        force_download=True
    )
    print(f"✅ SUCCESS! File downloaded to: {path}")
    print("Connection works perfectly.")

except Exception as e:
    print(f"\n❌ CONNECTION FAILED: {e}")
    print("\nTroubleshooting:")
    print("1. Check if 'export HF_ENDPOINT=https://hf-mirror.com' helps.")
    print("2. Check if your container allows outbound internet access.")
    print("3. Check DNS settings.")
