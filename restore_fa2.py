
import os
import re

TARGET_FILE = os.path.join("model_data", "Kimi-Linear-48B-A3B-Base", "modeling_kimi.py")
print(f"🔧 Re-enabling Flash Attention 2 in {TARGET_FILE}...")

if not os.path.exists(TARGET_FILE):
    print("❌ File not found.")
    exit(1)

with open(TARGET_FILE, "r", encoding="utf-8") as f:
    content = f.read()

# We previously replaced the config assignment with:
# if getattr(config, "_attn_implementation", None) is None:
#      config._attn_implementation = "eager"

# We want to restore the logic to force "flash_attention_2" if available,
# OR just set it to "flash_attention_2" directly since we are installing it.

# Let's search for our patch.
patch_signature = 'config._attn_implementation = "eager"'

if patch_signature in content:
    print("Found patched eager mode. Reverting to FA2...")
    # Replacing the eager block with a simple forced FA2 block
    # Note: Kimi code originally had a check. We'll just enforce it now.
    
    new_block = r"""
        # RESTORED: Trigger FA2 usage
        logger.info("Forcing Flash Attention 2...")
        config._attn_implementation = "flash_attention_2"
        self._use_flash_attention_2 = True
    """
    
    # We need to replace the whole block we inserted previously.
    # The previous block started with: # PATCHED: Don't force FA2 if it's broken
    
    pattern = r"# PATCHED: Don't force FA2 if it's broken.*?config\._attn_implementation\s*=\s*\"eager\".*?logger\.info\(.*?\)"
    
    # Using DOTALL to match newlines
    new_content = re.sub(pattern, new_block, content, flags=re.DOTALL)
    
    # Logic check: if re.sub didn't find the complex block (due to whitespace mismatch),
    # fallback to simple replacement of the eager line.
    if new_content == content:
        print("Complex regex didn't match. Doing simple replacement.")
        new_content = content.replace('config._attn_implementation = "eager"', 'config._attn_implementation = "flash_attention_2"')

    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("✅ Patched: Model is now configured for Flash Attention 2.")

else:
    print("⚠️  'eager' configuration not found. Perhaps it was already FA2?")
    # Check if we have the naive replace from before
    if 'config._attn_implementation = "flash_attention_2"' in content:
        print("✅ Config already looks like FA2.")
    else:
        print("❓ State unclear, but assuming FA2 is desired.")
