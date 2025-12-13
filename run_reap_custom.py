# Consolidate REAP runner
# Registers Kimi model and Ultra-FineWeb dataset, then runs pruning

import sys
import os

print("DEBUG: helping python find REAP...")
# Force-insert local reap sources to ensure we use the local files
# This fixes "ImportError: (unknown location)" and ensures we use the cloned code
project_root = os.path.dirname(os.path.abspath(__file__))
reap_src = os.path.join(project_root, 'reap', 'src')

if os.path.exists(reap_src):
    if reap_src not in sys.path:
        sys.path.insert(0, reap_src)
        print(f"DEBUG: Added {reap_src} to sys.path")
else:
    print(f"⚠️ Warning: {reap_src} does not exist. Relying on installed package.")

try:
    import reap
    print(f"DEBUG: reap imported from: {getattr(reap, '__file__', 'unknown')}")
except ImportError as e:
    print(f"❌ Critical Error: Could not import reap: {e}")
    sys.exit(1)

from reap import model_util
from reap import data
from reap import prune

print("✓ REAP submodules loaded successfully")

# ---------------------------------------------------------
# 1. Register Kimi Linear Model Support
# ---------------------------------------------------------
KIMI_CONFIG = {
    "KimiLinearForCausalLM": {
        "moe_block": "block_sparse_moe",
        "gate_proj": "w1",
        "up_proj": "w3",
        "down_proj": "w2",
        "experts": "experts",
        "fused": False,
        "router": "gate",
        "num_experts": "num_experts",
        "num_experts_per_tok": "num_experts_per_token",
    }
}
model_util.MODEL_ATTRS.update(KIMI_CONFIG)
print(f"✓ Registered Model Config: KimiLinearForCausalLM")

# ---------------------------------------------------------
# 2. Register Ultra-FineWeb Dataset Support
# ---------------------------------------------------------
class UltraFineWebDataset(data.LMDatasetProcessor):
    """Dataset processor for Ultra-FineWeb with 'content' column."""
    category_field: str = None
    text_field: str = "content"
    
    @staticmethod
    def _map_fn(sample: dict) -> dict:
        return {"text": sample.get("content", sample.get("text", ""))}

NEW_DATASET_NAME = "sumukshashidhar-archive/Ultra-FineWeb-100M"
data.DATASET_REGISTRY[NEW_DATASET_NAME] = UltraFineWebDataset
print(f"✓ Registered Dataset: {NEW_DATASET_NAME}")

# ---------------------------------------------------------
# --- 3. Patch REAP Observer for Kimi ---
try:
    from reap.observer import MoETransformerObserverConfig, OBSERVER_CONFIG_REGISTRY
    from dataclasses import dataclass
    from typing import Optional

    @dataclass
    class KimiObserverHookConfig(MoETransformerObserverConfig):
        module_class_name_to_hook_regex: Optional[str] = "KimiSparseMoeBlock"
        num_experts_attr_name: str = "num_experts"
        top_k_attr_name: str = "top_k"
        fused_experts: bool = False

    OBSERVER_CONFIG_REGISTRY["KimiLinearForCausalLM"] = KimiObserverHookConfig
    print("✅ Registered KimiObserverHookConfig")

except ImportError as e:
    print(f"⚠️  Could not patch observer (this might fail later if REAP not installed fully): {e}")

# --- 4. Patch REAP Arguments (Dataset) ---
# We monkeypatch the DatasetArgs to allow our custom dataset choice
try:
    from reap.args import DatasetArgs
    # We just override the choices metadata to allow anything, or at least our custom one
    # But since HfArgumentParser checks choices, we need to append to the list in metadata.
    # A quick hack is to just replace the field definition or just ignore it if we can.
    # Easier: modify the metadata of the existing field if mutable.
    
    # Access the choices list from the dataclass field metadata
    field_metadata = DatasetArgs.__dataclass_fields__["dataset_name"].metadata
    if "choices" in field_metadata:
        choices = field_metadata["choices"]
        if NEW_DATASET_NAME not in choices:
            choices.append(NEW_DATASET_NAME)
            print(f"✓ Patched DatasetArgs choices: Added {NEW_DATASET_NAME}")
    else:
        print("⚠️ Warning: No choices found in DatasetArgs metadata to patch.")
except Exception as e:
    print(f"⚠️ Warning: Failed to patch DatasetArgs choices: {e}")
    # We continue anyway, hoping validation isn't strict or we patched the right object

# ---------------------------------------------------------
# 3. Execution
# ---------------------------------------------------------
if __name__ == "__main__":
    print(f"\n🚀 Starting REAP Pruning Process...")
    # This calls the main function from reap.prune which parses args and runs logic
    prune.main()
