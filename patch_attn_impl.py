
import os
import re

TARGET_FILE = os.path.join("model_data", "Kimi-Linear-48B-A3B-Base", "modeling_kimi.py")
print(f"🔧 Patching {TARGET_FILE} to un-force Flash Attention 2...")

with open(TARGET_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# The forcing block looks like this:
#         if getattr(config, "_attn_implementation", None) is not None:
#             if config._attn_implementation != "flash_attention_2":
#                 logger.warning_once(
#                     f"Ignoring the provided attention implementation {config._attn_implementation}")
#                 logger.warning_once("Using flash_attention_2 backend instead.")
#                 config._attn_implementation = "flash_attention_2"
#         else:
#             config._attn_implementation = "flash_attention_2"

# We want to essentially comment this out or make it allow "eager".

# Regex to match the block. It's multi-line.
# We'll just replace the specific line that overwrites it.

# Pattern 1: config._attn_implementation = "flash_attention_2" inside the if
p1 = r'(config\._attn_implementation = "flash_attention_2")'

# We can just change it to use "eager" or just strictly pass what we want.
# Actually, the safest bet is to default to "eager" if we are crashing.

# Let's replace the whole init logic block for attn implementation.
# We search for:
# if getattr(config, "_attn_implementation", None) is not None:
# ...
# else:
#     config._attn_implementation = "flash_attention_2"

# We will replace it with:
# if getattr(config, "_attn_implementation", None) is None:
#     config._attn_implementation = "eager" # Default to eager for stability

block_start = 'if getattr(config, "_attn_implementation", None) is not None:'
block_end_marker = 'self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"'

start_idx = content.find(block_start)
end_idx = content.find(block_end_marker)

if start_idx != -1 and end_idx != -1:
    print("✅ Found attention config block.")
    # Extract the block
    old_block = content[start_idx:end_idx]
    
    # New block: Just respect config or default to eager
    new_block = r"""
        # PATCHED: Don't force FA2 if it's broken
        if getattr(config, "_attn_implementation", None) is None:
             config._attn_implementation = "eager"
        
        logger.info(f"Using attention implementation: {config._attn_implementation}")
        
        """
    new_content = content[:start_idx] + new_block + content[end_idx:]
    
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("✅ Patch applied! Model will now likely use 'eager' attention.")
    
else:
    print("❌ Could not isolate the attention config block exactly.")
    # Fallback: Just replace the assignment lines
    print("Trying naive replace...")
    new_content = content.replace('config._attn_implementation = "flash_attention_2"', 'config._attn_implementation = "eager"')
    if new_content != content:
         with open(TARGET_FILE, "w", encoding="utf-8") as f:
            f.write(new_content)
         print("✅ Naive patch applied (replaced assignments).")
    else:
        print("❌ Naive patch failed too.")

