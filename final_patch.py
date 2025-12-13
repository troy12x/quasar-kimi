
import os

# 1. READ clean source
SOURCE_FILE = "modeling_kimi.py"  # The one user just created in root
TARGET_FILE = os.path.join("model_data", "Kimi-Linear-48B-A3B-Base", "modeling_kimi.py")

print(f"📖 Reading source from: {SOURCE_FILE}")
with open(SOURCE_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# 2. DEFINE the Block to Replace
# We'll target the text exactly as it appears in the clean file
BLOCK_TO_REPLACE = r"""try:
    from fla.layers.utils import get_unpad_data, index_first_axis, pad_input
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    raise ImportError("Plese run `pip install -U fla-core`")"""

# 3. DEFINE the Replacement (Imports + Polyfills)
# We move this to top-level to avoid indentation headaches
NEW_BLOCK = r"""
# --- PATCH START: REAP Compatibility ---
try:
    # These usually exist
    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.gate import fused_kda_gate
except ImportError:
    pass

import torch

# POLYFILL: Functions missing from fla-core 0.4.0 (for variable length sequences)
def get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return indices, cu_seqlens, max_seqlen_in_batch

def pad_input(hidden_states, indices, batch, seq_len):
    dim = hidden_states.shape[-1]
    output = torch.zeros((batch * seq_len, dim), device=hidden_states.device, dtype=hidden_states.dtype)
    output[indices] = hidden_states
    return output.view(batch, seq_len, dim)

def index_first_axis(params, indices):
    return params[indices]
# --- PATCH END ---
"""

# 4. SWAP
if BLOCK_TO_REPLACE in content:
    print("✅ Found the target block. Replacing...")
    new_content = content.replace(BLOCK_TO_REPLACE, NEW_BLOCK)
    
    # 5. WRITE to target
    print(f"💾 Writing patched version to: {TARGET_FILE}")
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("✅ Patch Complete!")
else:
    print("❌ Could not find the exact target block in source file. Double check spacing.")
    # Fallback: exact match might fail due to whitespace, let's try a softer match
    print("Attempting loose match...")
    start_tag = "try:"
    end_tag = 'raise ImportError("Plese run `pip install -U fla-core`")'
    p1 = content.find(start_tag)
    p2 = content.find(end_tag)
    
    if p1 != -1 and p2 != -1:
        # includes the end tag length
        end_idx = p2 + len(end_tag)
        # Check if "fla.layers.utils" is in between (safety check)
        substr = content[p1:end_idx]
        if "fla.layers.utils" in substr:
            print("✅ Found via substring indices. Replacing...")
            new_content = content[:p1] + NEW_BLOCK + content[end_idx:]
            with open(TARGET_FILE, "w", encoding="utf-8") as f:
                f.write(new_content)
            print("✅ Patch Complete (Loose Match)!")
        else:
            print("❌ Found tags but content didn't look right.")
    else:
        print("❌ Could not locate block even loosely.")
