#!/usr/bin/env bash
# ============================================================
# Kaggle Notebook Setup Script
# Run this at the top of every Kaggle Notebook to:
#   1. Clone the GitHub repo
#   2. Install Python dependencies
#   3. Verify GPU availability
#   4. Set up environment variables
#
# Usage (in Kaggle cell):
#   !bash /kaggle/working/AIC_System/scripts/kaggle_setup.sh
#
# Or inline (first notebook cell):
#   import subprocess
#   result = subprocess.run(
#       ["bash", "/kaggle/working/AIC_System/scripts/kaggle_setup.sh"],
#       capture_output=True, text=True
#   )
#   print(result.stdout)
# ============================================================

set -e  # Exit on error

GITHUB_REPO="${GITHUB_REPO:-https://github.com/YOUR_USERNAME/AIC_System.git}"
BRANCH="${BRANCH:-main}" 
WORKDIR="/kaggle/working"
REPO_DIR="$WORKDIR/AIC_System"

echo "=============================================="
echo " AIC System — Kaggle Environment Setup"
echo "=============================================="

# --- Step 1: Clone or pull repo ---
echo "[1/4] Cloning GitHub repository..."
if [ -d "$REPO_DIR/.git" ]; then
    echo "  Repo already exists, pulling latest..."
    git -C "$REPO_DIR" pull origin "$BRANCH"
else
    git clone --branch "$BRANCH" --depth 1 "$GITHUB_REPO" "$REPO_DIR"
    echo "  Cloned to $REPO_DIR"
fi

# --- Step 2: Add repo to Python path ---
echo "[2/4] Setting up Python path..."
export PYTHONPATH="$REPO_DIR:$PYTHONPATH"

# Write .pth file so all subsequent cells can import src.*
python -c "
import site, os
pth = os.path.join(site.getsitepackages()[0], 'aic_system.pth')
with open(pth, 'w') as f:
    f.write('$REPO_DIR\n')
print(f'  Added to site-packages: {pth}')
"

# --- Step 3: Install dependencies ---
echo "[3/4] Installing Python dependencies..."
pip install -q -r "$REPO_DIR/requirements.txt" 2>&1 | tail -5
echo "  Dependencies installed."

# --- Step 4: Verify environment ---
echo "[4/4] Verifying environment..."
python - <<'EOF'
import sys
sys.path.insert(0, "/kaggle/working/AIC_System")

# GPU check
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        print(f"  GPU: {gpu.name} | VRAM: {gpu.total_memory / 1024**3:.1f} GB")
    else:
        print("  WARNING: No GPU detected. Running on CPU.")
except ImportError:
    print("  WARNING: torch not available.")

# FAISS check
try:
    import faiss
    print(f"  FAISS: OK (version {faiss.__version__})")
except Exception as e:
    print(f"  WARNING: FAISS not available: {e}")

# Core imports check
try:
    from src.common.types import KeyframeMeta, TextualKISQuery
    from src.common.enums import QueryType
    from src.common.constants import CLIP32_FEATURE_DIM
    print("  AIC System imports: OK")
except Exception as e:
    print(f"  ERROR: AIC System imports failed: {e}")

print("  Setup complete.")
EOF

echo "=============================================="
echo " Setup finished! Kaggle environment ready."
echo " Repo: $REPO_DIR"
echo "=============================================="
