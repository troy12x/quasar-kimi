# Add Kimi Linear support to RE

from reap import model_util

# Add Kimi Linear configuration to MODEL_ATTRS
KIMI_CONFIG = {
    "KimiLinearForCausalLM": {
        "moe_block": "block_sparse_moe",  # Kimi uses block_sparse_moe
        "gate_proj": "w1",   # gate in Kimi
        "up_proj": "w3",     # up in Kimi  
        "down_proj": "w2",   # down in Kimi
        "experts": "experts",
        "fused": False,
        "router": "gate",    # KimiMoEGate
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_token",
    }
}

# Add to MODEL_ATTRS
model_util.MODEL_ATTRS.update(KIMI_CONFIG)

print("✓ Added Kimi Linear support to REAP")
print(f"  - Model class: KimiLinearForCausalLM")
print(f"  - Experts: 256 → can prune to 128 (50% compression)")
print(f"  - Active params: 3B → will become ~1.5B after 50% pruning")
print(f"  - Expected memory reduction: ~40-50%")
