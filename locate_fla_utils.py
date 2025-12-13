
import fla
import inspect
import sys

print("🔍 Searching for 'get_unpad_data' in fla package...")

found = False

# Check top level
if hasattr(fla, 'get_unpad_data'):
    print("✅ Found in 'fla'")
    found = True

# Check utils
try:
    import fla.utils
    if hasattr(fla.utils, 'get_unpad_data'):
        print("✅ Found in 'fla.utils'")
        found = True
except ImportError:
    pass

# Check ops
try:
    import fla.ops
    # recursive inspection...
except ImportError:
    pass

if not found:
    print("❌ Could not find 'get_unpad_data' in standard locations.")
    print("Listing fla.utils contents:")
    import fla.utils
    print(dir(fla.utils))
else:
    print("\n💡 Recommendation: Update imports in modeling_kimi.py to match the above location.")
