
import sys
import os
import importlib.util

print(f"🐍 Python Executable: {sys.executable}")
print(f"🐍 Python Version: {sys.version}")
print("-" * 40)

# Check for 'fla' package
try:
    import fla
    print(f"✅ 'fla' module imported successfully.")
    print(f"📂 Location: {getattr(fla, '__file__', 'unknown')}")
    print(f"ℹ️  Version: {getattr(fla, '__version__', 'unknown')}")
    
    # Check for 'fla.layers'
    try:
        import fla.layers
        print(f"✅ 'fla.layers' imported successfully.")
    except ImportError as e:
        print(f"❌ 'fla.layers' FAILED to import: {e}")
        print("listing dir of fla:")
        if hasattr(fla, '__path__'):
             print(os.listdir(fla.__path__[0]))

except ImportError as e:
    print(f"❌ 'fla' module FAILED to import: {e}")

print("-" * 40)
print("📦 pip freeze | grep fla:")
try:
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "fla" in line.lower():
            print(line)
except Exception as e:
    print(f"Failed to run pip freeze: {e}")
