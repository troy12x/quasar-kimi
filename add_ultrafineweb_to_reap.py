# Add Ultra-FineWeb dataset support to REAP
# Ultra-FineWeb-100M has raw text in "content" column

import sys
sys.path.insert(0, 'reap/src')

from reap.data import LMDatasetProcessor, DATASET_REGISTRY

class UltraFineWebDataset(LMDatasetProcessor):
    """Dataset processor for Ultra-FineWeb with 'content' column."""
    
    category_field: str = None  # No categories
    text_field: str = "content"  # Ultra-FineWeb uses 'content' not 'text'
    
    @staticmethod
    def _map_fn(sample: dict) -> dict:
        # Ultra-FineWeb already has content in right format
        # Just rename to 'text' for compatibility
        return {"text": sample.get("content", sample.get("text", ""))}

# Register the dataset
DATASET_REGISTRY["sumukshashidhar-archive/Ultra-FineWeb-100M"] = UltraFineWebDataset

print("✓ Added Ultra-FineWeb-100M support to REAP")
print("  - Dataset: sumukshashidhar-archive/Ultra-FineWeb-100M")
print("  - Text column: content")
print("  - Type: Language Modeling (raw web text)")
