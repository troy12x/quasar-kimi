
import os
import sys

# Define the polyfills for the missing functions
# NOTE: The original code has this import inside a try/except block indented by 4 spaces.
# So we must indent our polyfill code to match, OR ensure we close the try block properly.
# However, defining functions inside a try block is weird.
# Better strategy: Inspect the file content first to see indentation.
# Assuming standard 4 spaces:

POLYFILLS = r"""
    import torch

    # POLYFILL: Functions missing from fla-core 0.4.0
    def get_unpad_data(attention_mask):
        seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
        indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
        max_seqlen_in_batch = seqlens_in_batch.max().item()
        cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
        return indices, cu_seqlens, max_seqlen_in_batch

    def pad_input(hidden_states, indices, batch, seq_len):
        # Inverse of unpad: scatter back to padded tensor
        dim = hidden_states.shape[-1]
        output = torch.zeros((batch * seq_len, dim), device=hidden_states.device, dtype=hidden_states.dtype)
        output[indices] = hidden_states
        return output.view(batch, seq_len, dim)

    def index_first_axis(params, indices):
        return params[indices]
    # END POLYFILL
"""

# Targets
TARGET_DIR = os.path.join("model_data", "Kimi-Linear-48B-A3B-Base")
TARGET_FILE = "modeling_kimi.py"
FULL_PATH = os.path.join(TARGET_DIR, TARGET_FILE)

print(f"🔧 Attempting to patch {FULL_PATH}...")

if not os.path.exists(FULL_PATH):
    print(f"❌ File not found at: {FULL_PATH}")
    # Try searching?
    print("Searching recursively...")
    for root, dirs, files in os.walk("."):
        if TARGET_FILE in files:
            FULL_PATH = os.path.join(root, TARGET_FILE)
            print(f"✅ Found at: {FULL_PATH}")
            break
    else:
        print("❌ Could not find file. Exiting.")
        sys.exit(1)

with open(FULL_PATH, 'r', encoding='utf-8') as f:
    content = f.read()

# Check against original import
BROKEN_IMPORT = "from fla.layers.utils import get_unpad_data, index_first_axis, pad_input"

if BROKEN_IMPORT in content:
    print("Found broken import. replacing...")
    new_content = content.replace(BROKEN_IMPORT, POLYFILLS)
    
    # Also comment out the 'pip install' error raise that Kimi adds
    # "raise ImportError("Plese run `pip install -U fla-core`")"
    ERROR_RAISE = 'raise ImportError("Plese run `pip install -U fla-core`")'
    if ERROR_RAISE in new_content:
        print("Disabling the ImportError check...")
        new_content = new_content.replace(ERROR_RAISE, '# ' + ERROR_RAISE)
        
    with open(FULL_PATH, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("✅ Patch applied successfully.")
else:
    if "def get_unpad_data" in content:
        print("⚠️  File already patched (polyfills found).")
    else:
        print("❌ Could not find the specific import string to replace.")
        print("First 500 chars of file:")
        print(content[:500])

