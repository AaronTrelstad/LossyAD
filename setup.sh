#!/usr/bin/env bash
# setup.sh — Full environment + data setup for LossyAD on Nova (or any GPU server).
#
# Usage:
#   bash setup.sh                        # full setup
#   bash setup.sh --no-sz3               # skip SZ3 native build
#   bash setup.sh --no-serf              # skip SERF native build
#   bash setup.sh --no-datasets          # skip dataset download
#   bash setup.sh --no-model-weights     # skip foundation model weight download
#
# Prerequisites: conda (or mamba), cmake, gcc/g++, CUDA toolkit, internet access
#
# Run this interactively (not via sbatch) — Nova compute nodes may not have
# outbound internet access for large downloads.

set -euo pipefail

BUILD_SZ3=true
BUILD_SERF=true
DOWNLOAD_DATASETS=true
DOWNLOAD_WEIGHTS=true

for arg in "$@"; do
  case $arg in
    --no-sz3)            BUILD_SZ3=false          ;;
    --no-serf)           BUILD_SERF=false         ;;
    --no-datasets)       DOWNLOAD_DATASETS=false  ;;
    --no-model-weights)  DOWNLOAD_WEIGHTS=false   ;;
  esac
done

echo "=== LossyAD Setup ==="
echo "    SZ3:             $BUILD_SZ3"
echo "    SERF:            $BUILD_SERF"
echo "    Download datasets: $DOWNLOAD_DATASETS"
echo "    Download weights:  $DOWNLOAD_WEIGHTS"
echo ""

# ---------------------------------------------------------------------------
# 1. Python packages (pip-installable)
# ---------------------------------------------------------------------------
echo "[setup] Installing pip dependencies..."
pip install -r requirements.txt

# TerseTS (MixPiece) — not on PyPI, install from git
echo "[setup] Installing TerseTS (MixPiece)..."
pip install git+https://github.com/cmcuza/TerseTS.git || \
  echo "[setup] WARNING: TerseTS install failed — MixPiece compressor will be unavailable."

# ---------------------------------------------------------------------------
# 2. TSB-AD datasets
# ---------------------------------------------------------------------------
if [ "$DOWNLOAD_DATASETS" = true ]; then
  echo ""
  echo "[setup] === Downloading TSB-AD datasets ==="

  mkdir -p Datasets/TSB-AD-U Datasets/TSB-AD-M

  # TSB-AD univariate dataset
  # Download from the TSB-AD GitHub releases page or the official mirror.
  # Check https://github.com/thedatumorg/TSB-AD for the latest download link.
  TSB_U_URL="https://thedatumorg.org/datasets/TSB-AD-U.zip"
  TSB_M_URL="https://thedatumorg.org/datasets/TSB-AD-M.zip"

  if [ -z "$(ls -A Datasets/TSB-AD-U 2>/dev/null)" ]; then
    echo "[setup] Downloading TSB-AD-U (univariate)..."
    wget -q --show-progress -O /tmp/TSB-AD-U.zip "$TSB_U_URL" && \
      unzip -q /tmp/TSB-AD-U.zip -d Datasets/ && \
      rm /tmp/TSB-AD-U.zip && \
      echo "[setup] TSB-AD-U extracted to Datasets/TSB-AD-U/" || \
      echo "[setup] WARNING: TSB-AD-U download failed. Download manually and extract to Datasets/TSB-AD-U/"
  else
    echo "[setup] Datasets/TSB-AD-U/ already populated — skipping download."
  fi

  if [ -z "$(ls -A Datasets/TSB-AD-M 2>/dev/null)" ]; then
    echo "[setup] Downloading TSB-AD-M (multivariate)..."
    wget -q --show-progress -O /tmp/TSB-AD-M.zip "$TSB_M_URL" && \
      unzip -q /tmp/TSB-AD-M.zip -d Datasets/ && \
      rm /tmp/TSB-AD-M.zip && \
      echo "[setup] TSB-AD-M extracted to Datasets/TSB-AD-M/" || \
      echo "[setup] WARNING: TSB-AD-M download failed. Download manually and extract to Datasets/TSB-AD-M/"
  else
    echo "[setup] Datasets/TSB-AD-M/ already populated — skipping download."
  fi
else
  echo "[setup] Skipping dataset download (--no-datasets)."
fi

# ---------------------------------------------------------------------------
# 3. Foundation model weights (HuggingFace — must be done with internet access)
# ---------------------------------------------------------------------------
if [ "$DOWNLOAD_WEIGHTS" = true ]; then
  echo ""
  echo "[setup] === Pre-downloading foundation model weights ==="
  echo "        Weights stored in /ptmp/\$USER/hf_cache (not home, avoids 10GB quota)"
  echo "        This prevents timeouts when jobs run on nodes without internet."

  # Mirror the same cache dirs used by the Slurm scripts
  export HF_HOME=/ptmp/$USER/hf_cache
  export TORCH_HOME=/ptmp/$USER/torch_cache
  mkdir -p "$HF_HOME" "$TORCH_HOME"

  python - <<'PYEOF'
import sys

models = [
    ("MOMENT_FT / MOMENT_ZS", "AutonLab/MOMENT-1-large",       "momentfm",  "MOMENTPipeline"),
    ("Chronos",                "amazon/chronos-t5-small",        "chronos",   "ChronosPipeline"),
    ("Lag-Llama",              "time-series-foundation-models/Lag-Llama", "lag_llama", None),
    ("TimesFM",                "google/timesfm-1.0-200m",        "timesfm",   None),
    ("OFA / TimesNet",         None,                             None,        None),  # trained from scratch
]

for name, model_id, pkg, cls in models:
    if model_id is None:
        print(f"  {name}: trained from scratch — no pre-download needed")
        continue
    try:
        if pkg == "momentfm":
            from momentfm import MOMENTPipeline
            MOMENTPipeline.from_pretrained(model_id, task="anomaly_detection", cache_dir=None)
            print(f"  [OK] {name} weights cached")
        elif pkg == "chronos":
            from chronos import ChronosPipeline
            ChronosPipeline.from_pretrained(model_id, device_map="cpu")
            print(f"  [OK] {name} weights cached")
        else:
            # Try generic HuggingFace snapshot_download as fallback
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id)
            print(f"  [OK] {name} weights cached via snapshot_download")
    except ImportError:
        print(f"  [SKIP] {name}: package '{pkg}' not installed (installed via TSB-AD at runtime)")
    except Exception as e:
        print(f"  [WARN] {name}: {e}")
        print(f"         Weights may be downloaded automatically on first job run.")

print("Done with model weight pre-fetch.")
PYEOF

else
  echo "[setup] Skipping model weight download (--no-model-weights)."
fi

# ---------------------------------------------------------------------------
# 4. SZ3 native library (optional)
# ---------------------------------------------------------------------------
if [ "$BUILD_SZ3" = true ]; then
  echo ""
  echo "[setup] === Building SZ3 ==="
  SZ3_DIR="external/SZ3"
  if [ -d "$SZ3_DIR" ]; then
    mkdir -p "$SZ3_DIR/build" "$SZ3_DIR/install"
    cmake -S "$SZ3_DIR" -B "$SZ3_DIR/build" \
          -DCMAKE_INSTALL_PREFIX="$(pwd)/$SZ3_DIR/install" \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=ON 2>&1 | tail -5
    cmake --build "$SZ3_DIR/build" --parallel "$(nproc)" 2>&1 | tail -5
    cmake --install "$SZ3_DIR/build" 2>&1 | tail -5
    echo "[setup] SZ3 built → $SZ3_DIR/install/lib/"
  else
    echo "[setup] external/SZ3 not found — skipping SZ3 build."
    echo "        Clone it with: git clone https://github.com/szcompressor/SZ3 external/SZ3"
  fi
else
  echo "[setup] Skipping SZ3 build (--no-sz3)."
fi

# ---------------------------------------------------------------------------
# 5. SERF native extension (optional)
# ---------------------------------------------------------------------------
if [ "$BUILD_SERF" = true ]; then
  echo ""
  echo "[setup] === Building SERF ==="
  SERF_DIR="external/Serf"
  if [ ! -d "$SERF_DIR" ] || [ -z "$(ls -A $SERF_DIR 2>/dev/null)" ]; then
    echo "[setup] Cloning SERF from GitHub..."
    git clone https://github.com/Spatio-Temporal-Lab/Serf "$SERF_DIR" || {
      echo "[setup] WARNING: Could not clone SERF — SERF compressor will be unavailable."
      BUILD_SERF=false
    }
  fi

  if [ "$BUILD_SERF" = true ] && [ -d "$SERF_DIR" ]; then
    echo "[setup] Building SERF python wrapper..."
    mkdir -p "$SERF_DIR/build"
    cmake -S "$SERF_DIR" -B "$SERF_DIR/build" \
          -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    cmake --build "$SERF_DIR/build" --parallel "$(nproc)" 2>&1 | tail -5

    PYWRAPPER=$(find "$SERF_DIR/build" -name "pyserf*.so" -o -name "pyserf*.pyd" 2>/dev/null | head -1)
    if [ -n "$PYWRAPPER" ]; then
      PYWRAPPER_DIR=$(dirname "$PYWRAPPER")
      echo "[setup] SERF pyserf extension found at: $PYWRAPPER_DIR"
      echo "        Add to your shell profile:"
      echo "        export SERF_PYWRAPPER_PATH=$(pwd)/$PYWRAPPER_DIR"
    else
      echo "[setup] WARNING: Could not locate pyserf shared library after build."
      echo "        Check $SERF_DIR/build/ manually."
    fi
  fi
else
  echo "[setup] Skipping SERF build (--no-serf)."
fi

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Update slurm/*.sh with your ISU email address"
echo "  2. Run the smoke test:  python main.py experiment --dataset-list Datasets/File_List/TSB-AD-U-Smoke.csv --detectors IForest"
echo "  3. Submit all jobs:     bash slurm/submit.sh"
