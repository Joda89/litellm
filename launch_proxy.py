#!/usr/bin/env python3
"""
Launcher LiteLLM Proxy - Compatible Python 3.14
"""

import os
import sys

# Force asyncio instead of uvloop (fixes Python 3.14 compatibility)
os.environ["UVICORN_LOOP"] = "asyncio"

import subprocess

# Configuration
config_file = sys.argv[1] if len(sys.argv) > 1 else "config_1min.yaml"
port = sys.argv[2] if len(sys.argv) > 2 else "4000"

print(f"""
╔══════════════════════════════════════════════════════════════╗
║          🚀 LiteLLM Proxy with 1min.ai                      ║
╚══════════════════════════════════════════════════════════════╝

Config: {config_file}
Port:   {port}
""")

# Vérifier la clé API
if not os.environ.get("ONE_MIN_API_KEY"):
    print("⚠️  Warning: ONE_MIN_API_KEY not set")
    print()

# Lancer litellm
cmd = ["litellm", "--config", config_file, "--port", port, "--host", "0.0.0.0"]
print(f"Running: {' '.join(cmd)}\n")

try:
    subprocess.run(cmd, check=True)
except KeyboardInterrupt:
    print("\n👋 Proxy stopped")
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
