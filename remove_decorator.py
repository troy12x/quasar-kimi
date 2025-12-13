
import os

# Define path relative to where script is run (c:\quasar-kimi\)
TARGET_REL_PATH = os.path.join("model_data", "Kimi-Linear-48B-A3B-Base", "modeling_kimi.py")
TARGET_FILE = os.path.abspath(TARGET_REL_PATH)

print(f"🎯 Targeting: {TARGET_FILE}")

if not os.path.exists(TARGET_FILE):
    print("❌ File not found! Checking directory...")
    print(os.listdir(os.path.join("model_data", "Kimi-Linear-48B-A3B-Base")))
    exit(1)

with open(TARGET_FILE, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
found = False
for line in lines:
    # We look for the decorator with some flexibility on whitespace
    if line.strip() == "@check_model_inputs":
        print("✅ Found decorator line. Commenting it out.")
        new_lines.append(f"# {line}") # Comment out but keep indentation
        found = True
    else:
        new_lines.append(line)

if found:
    with open(TARGET_FILE, "w", encoding="utf-8") as f:
        f.writelines(new_lines)
    print("💾 Saved changes.")
else:
    print("⚠️ Decorator not found (maybe already removed?)")
