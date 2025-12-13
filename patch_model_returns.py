
import os
import re

TARGET_FILE = os.path.join("model_data", "Kimi-Linear-48B-A3B-Base", "modeling_kimi.py")
print(f"🔧 Patching {TARGET_FILE} for REAP compatibility (returning router logits)...")

with open(TARGET_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# -------------------------------------------------------------------------
# 1. KimiMoEGate: Return logits
# -------------------------------------------------------------------------
# Find the return statement in KimiMoEGate.forward
# It currently looks like: return topk_idx, topk_weight
# We want: return topk_idx, topk_weight, logits

# Be careful not to replace other returns. Use context.
# We look for the last return in KimiMoEGate logic.
# It uses "routed_scaling_factor" right before return usually.
pattern_gate_return = r"(topk_weight = topk_weight \* self\.routed_scaling_factor\s+return topk_idx, topk_weight)"
replacement_gate_return = r"topk_weight = topk_weight * self.routed_scaling_factor\n        return topk_idx, topk_weight, logits"

if re.search(pattern_gate_return, content):
    content = re.sub(pattern_gate_return, replacement_gate_return, content)
    print("✅ Patched KimiMoEGate to return logits.")
else:
    print("⚠️  KimiMoEGate return pattern not found (already patched?)")


# -------------------------------------------------------------------------
# 2. KimiSparseMoeBlock: Capture and Return logits
# -------------------------------------------------------------------------
# A. Capture: topk_idx, topk_weight = self.gate(hidden_states)
# -> topk_idx, topk_weight, router_logits = self.gate(hidden_states)

pattern_moe_call = r"(topk_idx, topk_weight = self\.gate\(hidden_states\))"
replacement_moe_call = r"topk_idx, topk_weight, router_logits = self.gate(hidden_states)"

if re.search(pattern_moe_call, content):
    content = re.sub(pattern_moe_call, replacement_moe_call, content)
    print("✅ Patched KimiSparseMoeBlock to capture logits.")
else:
    print("⚠️  KimiSparseMoeBlock gate call pattern not found.")

# B. Return: return y -> return y, router_logits
# Need context. It usually ends with:
# if self.config.num_shared_experts is not None:
#    y = y + self.shared_experts(identity)
# return y

pattern_moe_return = r"(return y)(\s+@torch\.no_grad\(\))" # Lookahead to next method def
# Or just simple replacement if unique enough in that class.
# Let's use the line before it: "y = y + self.shared_experts(identity)" is optional.
# "return y" inside KimiSparseMoeBlock.
# Let's search for the whole block end structure to be safe.

# We'll use a specific replacement for the return y inside the forward method of SparseMoe
# It is preceded by "if self.config.num_shared_experts is not None:" check block.
pattern_moe_ret_ctx = r"(if self\.config\.num_shared_experts is not None:\s+y = y \+ self\.shared_experts\(identity\)\s+return y)"
replacement_moe_ret_ctx = r"if self.config.num_shared_experts is not None:\n            y = y + self.shared_experts(identity)\n        return y, router_logits"

if re.search(pattern_moe_ret_ctx, content):
    content = re.sub(pattern_moe_ret_ctx, replacement_moe_ret_ctx, content)
    print("✅ Patched KimiSparseMoeBlock to return (y, router_logits).")
else:
    print("⚠️  KimiSparseMoeBlock return pattern not found.")


# -------------------------------------------------------------------------
# 3. KimiDecoderLayer: Unpack output
# -------------------------------------------------------------------------
# if hasattr(self, "block_sparse_moe"):
#     hidden_states = self.block_sparse_moe(hidden_states)
# else:

pattern_decoder = r"(hidden_states = self\.block_sparse_moe\(hidden_states\))"
replacement_decoder = r"hidden_states = self.block_sparse_moe(hidden_states)\n            if isinstance(hidden_states, tuple):\n                hidden_states = hidden_states[0]"

if re.search(pattern_decoder, content):
    content = re.sub(pattern_decoder, replacement_decoder, content)
    print("✅ Patched KimiDecoderLayer to unpack tuple output.")
else:
    print("⚠️  KimiDecoderLayer unpack pattern not found.")

# Save
with open(TARGET_FILE, "w", encoding="utf-8") as f:
    f.write(content)
print("💾 File saved.")
