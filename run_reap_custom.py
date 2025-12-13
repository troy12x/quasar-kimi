# Custom runner for REAP expert pruning
# Registers Kimi model and Ultra-FineWeb dataset before running pruning

import sys
import os

# Ensure we can import from reap
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'reap/src'))

from reap import model_util
from reap import data
from reap import prune

# 1. Register Kimi Linear support
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
model_util.MODEL_ATTRS.update(KIMI_CONFIG)
print(f"✓ Registered Kimi config: {list(KIMI_CONFIG.keys())}")

# 2. Register Ultra-FineWeb dataset support
class UltraFineWebDataset(data.LMDatasetProcessor):
    """Dataset processor for Ultra-FineWeb with 'content' column."""
    category_field: str = None  # No categories
    text_field: str = "content"  # Ultra-FineWeb uses 'content' not 'text'
    
    @staticmethod
    def _map_fn(sample: dict) -> dict:
        return {"text": sample.get("content", sample.get("text", ""))}

data.DATASET_REGISTRY["sumukshashidhar-archive/Ultra-FineWeb-100M"] = UltraFineWebDataset
print(f"✓ Registered dataset: sumukshashidhar-archive/Ultra-FineWeb-100M")

# 3. Run REAP Pruning Main Function
if __name__ == "__main__":
    print("🚀 Starting REAP pruning...")
    prune.main()
