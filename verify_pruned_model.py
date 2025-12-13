
import os

# Use HF Mirror for faster download
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# Redirect Triton cache to shared drive to avoid disk space errors
os.environ["TRITON_CACHE_DIR"] = "/shared/.triton/cache"
os.makedirs("/shared/.triton/cache", exist_ok=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model ID on Hugging Face
MODEL_ID = "eyad-silx/Quasar-40B"

print(f"🚀 Loading model from HF Hub: {MODEL_ID}...")
print("Note: This requires the model to be fully uploaded.")

try:
    # Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    
    # Load Model (using auto device map for GPU usage if available)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype="auto"
    )
    print("✅ Model and Tokenizer loaded successfully!")
except Exception as e:
    print(f"❌ Critical Error loading model: {e}")
    print("Tip: Ensure 'python upload_to_hf.py' finished successfully.")
    exit(1)

# -------------------------------------------------------------
# 1. Verify Pruning (Expert Count)
# -------------------------------------------------------------
print("\n🔍 Checking Pruning Status...")
EXPECTED_EXPERTS = 128

# Check Config
if hasattr(model.config, "num_experts"):
    print(f"   Config 'num_experts': {model.config.num_experts}")
    if model.config.num_experts == EXPECTED_EXPERTS:
        print("   ✅ Config matches expected expert count.")
    else:
        print(f"   ⚠️  Config shows {model.config.num_experts} experts (Desired: {EXPECTED_EXPERTS})")

# Check Actual Layer Structure
try:
    # Kimi is a Hybrid model; Layer 0 might be dense. Scan layers until we find MoE.
    found_moe = False
    for i, layer in enumerate(model.model.layers[:10]): # Check first 10 layers
        if hasattr(layer, "block_sparse_moe"):
            print(f"   ℹ️  Found MoE block at Layer {i}")
            experts = layer.block_sparse_moe.experts
            num_actual_experts = len(experts)
            print(f"   Actual experts in Layer {i}: {num_actual_experts}")
            
            if num_actual_experts == EXPECTED_EXPERTS:
                print("   ✅ Layer structure confirms correct pruning!")
            else:
                print(f"   ⚠️  Layer has {num_actual_experts} experts!")
            found_moe = True
            break
    
    if not found_moe:
        print("   ⚠️  Could not find 'block_sparse_moe' in first 10 layers.")

except Exception as e:
    print(f"   ⚠️  Error inspecting model structure: {e}")

# -------------------------------------------------------------
# 2. Smoke Test (Generation)
# -------------------------------------------------------------
print("\n📝 Running Generation Smoke Test...")
prompt = "The future of Artificial General Intelligence is"
print(f"   Prompt: '{prompt}'")

try:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"\n🤖 Model Output:\n{'-'*40}\n{output_text}\n{'-'*40}")
    print("\n✅ Verification script finished.")

except Exception as e:
    print(f"\n❌ Generation failed: {e}")
