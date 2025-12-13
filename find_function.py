
import os
import fla
import inspect

package_dir = os.path.dirname(fla.__file__)
print(f"📂 Searching in: {package_dir}")

target = "def get_unpad_data"

found_at = []

for root, dirs, files in os.walk(package_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                    if target in content:
                        rel_path = os.path.relpath(path, package_dir)
                        found_at.append(rel_path)
            except Exception:
                pass

if found_at:
    print(f"✅ Found '{target}' in:")
    for p in found_at:
        print(f" - {p}")
else:
    print(f"❌ '{target}' NOT found in installed fla package.")

# Also check for local model existence to ensure we are patching the right thing
local_model_path = os.path.abspath("model_data/Kimi-Linear-48B-A3B-Base")
if os.path.exists(local_model_path):
    print(f"\n✅ Local model dir exists: {local_model_path}")
    if os.path.exists(os.path.join(local_model_path, "modeling_kimi.py")):
        print("✅ modeling_kimi.py found locally.")
    else:
        print("❌ modeling_kimi.py missing locally.")
else:
    print(f"\n❌ Local model dir NOT found: {local_model_path}")
