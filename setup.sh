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

# ---------------------------------------------------------------------------
# Storage root detection
# Prefer /work/classtmp/<user> (Nova HPC storage, accessible from compute nodes).
# Falls back to home directory if not available.
# ---------------------------------------------------------------------------
if [ -d "/work/classtmp/$USER" ]; then
  STORAGE_ROOT="/work/classtmp/$USER/lossyad"
  echo "[setup] Storage: /work/classtmp/$USER/lossyad (HPC storage)"
else
  STORAGE_ROOT="$HOME/.lossyad_data"
  echo "[setup] Storage: $HOME/.lossyad_data (home directory fallback)"
  echo "        NOTE: /work/classtmp/$USER not found. Datasets and model weights"
  echo "        will be stored in home. Watch your 10GB quota."
fi
mkdir -p "$STORAGE_ROOT/datasets" "$STORAGE_ROOT/hf_cache" "$STORAGE_ROOT/torch_cache"

echo "=== LossyAD Setup ==="
echo "    Storage root:      $STORAGE_ROOT"
echo "    SZ3:               $BUILD_SZ3"
echo "    SERF:              $BUILD_SERF"
echo "    Download datasets: $DOWNLOAD_DATASETS"
echo "    Download weights:  $DOWNLOAD_WEIGHTS"
echo ""

# ---------------------------------------------------------------------------
# 1. Python packages (pip-installable)
# ---------------------------------------------------------------------------
echo "[setup] Installing pip dependencies..."
pip install -r requirements.txt

# TerseTS — requires Zig compiler; install ziglang first then build from source
echo "[setup] Installing TerseTS (MixPiece)..."
pip install ziglang && \
  git clone --depth=1 https://github.com/cmcuza/TerseTS.git /tmp/TerseTS_build 2>/dev/null || true
if [ -d "/tmp/TerseTS_build/bindings/python" ]; then
  pip install /tmp/TerseTS_build/bindings/python/ || \
    echo "[setup] WARNING: TerseTS build failed — MP/SWING/SIMPIE/SLIDE/VW compressors will be unavailable."
  rm -rf /tmp/TerseTS_build
else
  echo "[setup] WARNING: TerseTS clone failed — MP/SWING/SIMPIE/SLIDE/VW compressors will be unavailable."
fi

# ---------------------------------------------------------------------------
# 2. TSB-AD datasets
# ---------------------------------------------------------------------------
if [ "$DOWNLOAD_DATASETS" = true ]; then
  echo ""
  echo "[setup] === Downloading TSB-AD datasets ==="
  echo "        Target: $STORAGE_ROOT/datasets/"

  TSB_U_URL="https://www.thedatum.org/datasets/TSB-AD-U.zip"
  TSB_M_URL="https://www.thedatum.org/datasets/TSB-AD-M.zip"

  # --- Univariate ---
  if [ -z "$(ls -A "$STORAGE_ROOT/datasets/TSB-AD-U" 2>/dev/null)" ]; then
    echo "[setup] Downloading TSB-AD-U (univariate)..."
    wget -q --show-progress -O /tmp/TSB-AD-U.zip "$TSB_U_URL" && \
      unzip -q /tmp/TSB-AD-U.zip -d "$STORAGE_ROOT/datasets/" && \
      rm /tmp/TSB-AD-U.zip && \
      echo "[setup] TSB-AD-U extracted to $STORAGE_ROOT/datasets/TSB-AD-U/" || \
      echo "[setup] WARNING: TSB-AD-U download failed."
  else
    echo "[setup] TSB-AD-U already present — skipping download."
  fi

  # --- Multivariate ---
  if [ -z "$(ls -A "$STORAGE_ROOT/datasets/TSB-AD-M" 2>/dev/null)" ]; then
    echo "[setup] Downloading TSB-AD-M (multivariate)..."
    wget -q --show-progress -O /tmp/TSB-AD-M.zip "$TSB_M_URL" && \
      unzip -q /tmp/TSB-AD-M.zip -d "$STORAGE_ROOT/datasets/" && \
      rm /tmp/TSB-AD-M.zip && \
      echo "[setup] TSB-AD-M extracted to $STORAGE_ROOT/datasets/TSB-AD-M/" || \
      echo "[setup] WARNING: TSB-AD-M download failed."
  else
    echo "[setup] TSB-AD-M already present — skipping download."
  fi

  # --- Symlink datasets into the repo so the code finds them ---
  REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
  for ds in TSB-AD-U TSB-AD-M; do
    SRC="$STORAGE_ROOT/datasets/$ds"
    DST="$REPO_DIR/Datasets/$ds"
    if [ -d "$SRC" ] && [ ! -e "$DST" ]; then
      ln -s "$SRC" "$DST"
      echo "[setup] Symlinked $DST -> $SRC"
    elif [ -e "$DST" ] && [ ! -L "$DST" ]; then
      echo "[setup] $DST already exists as a real directory — leaving it."
    fi
  done

else
  echo "[setup] Skipping dataset download (--no-datasets)."
fi

# ---------------------------------------------------------------------------
# 3. Foundation model weights (HuggingFace — must be done with internet access)
# ---------------------------------------------------------------------------
if [ "$DOWNLOAD_WEIGHTS" = true ]; then
  echo ""
  echo "[setup] === Pre-downloading foundation model weights ==="
  echo "        Weights stored in $STORAGE_ROOT/hf_cache"
  echo "        This prevents timeouts when jobs run on nodes without internet."

  export HF_HOME="$STORAGE_ROOT/hf_cache"
  export TORCH_HOME="$STORAGE_ROOT/torch_cache"

  # Write the storage root path to a file the slurm scripts can source
  echo "$STORAGE_ROOT" > "$(cd "$(dirname "$0")" && pwd)/.storage_root"
  echo "[setup] Saved storage root to .storage_root"

  python - <<PYEOF
import os, sys

models = [
    ("MOMENT_FT / MOMENT_ZS", "AutonLab/MOMENT-1-large",                    "momentfm",  "MOMENTPipeline"),
    ("TimesFM",                "google/timesfm-1.0-200m",                    "huggingface_hub", None),
    ("Chronos",                "amazon/chronos-t5-small",                    "chronos",   "ChronosPipeline"),
    ("Lag-Llama",              "time-series-foundation-models/Lag-Llama",    "huggingface_hub", None),
    ("OFA / TimesNet / USAD",  None,                                         None,        None),
]

for name, model_id, pkg, cls in models:
    if model_id is None:
        print(f"  {name}: trained from scratch — no pre-download needed")
        continue
    try:
        if pkg == "momentfm":
            from momentfm import MOMENTPipeline
            MOMENTPipeline.from_pretrained(model_id, task="anomaly_detection")
            print(f"  [OK] {name} weights cached")
        elif pkg == "chronos":
            from chronos import ChronosPipeline
            ChronosPipeline.from_pretrained(model_id, device_map="cpu")
            print(f"  [OK] {name} weights cached")
        else:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id)
            print(f"  [OK] {name} weights cached")
    except ImportError:
        print(f"  [SKIP] {name}: package '{pkg}' not installed yet — will download on first job run")
    except Exception as e:
        print(f"  [WARN] {name}: {e}")
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

  # Load cmake via module if not already in PATH
  if ! command -v cmake &>/dev/null; then
    module load cmake 2>/dev/null && echo "[setup] Loaded cmake via module" || \
      { echo "[setup] WARNING: cmake not found — skipping SZ3. Run: module load cmake"; BUILD_SZ3=false; }
  fi

  SZ3_DIR="external/SZ3"
  if [ "$BUILD_SZ3" = true ] && [ -d "$SZ3_DIR" ]; then
    mkdir -p "$SZ3_DIR/build" "$SZ3_DIR/install"
    cmake -S "$SZ3_DIR" -B "$SZ3_DIR/build" \
          -DCMAKE_INSTALL_PREFIX="$(pwd)/$SZ3_DIR/install" \
          -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_SHARED_LIBS=ON 2>&1 | tail -5
    cmake --build "$SZ3_DIR/build" --parallel "$(nproc)" 2>&1 | tail -5
    cmake --install "$SZ3_DIR/build" 2>&1 | tail -5
    echo "[setup] SZ3 built → $SZ3_DIR/install/lib/"
  elif [ "$BUILD_SZ3" = true ]; then
    echo "[setup] external/SZ3 not found — skipping."
    echo "        Clone with: git clone https://github.com/szcompressor/SZ3 external/SZ3"
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
    cmake -S "$SERF_DIR" -B "$SERF_DIR/build" -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -5
    cmake --build "$SERF_DIR/build" --parallel "$(nproc)" 2>&1 | tail -5
    PYWRAPPER=$(find "$SERF_DIR/build" -name "pyserf*.so" -o -name "pyserf*.pyd" 2>/dev/null | head -1)
    if [ -n "$PYWRAPPER" ]; then
      echo "[setup] SERF built at: $(dirname "$PYWRAPPER")"
      echo "        export SERF_PYWRAPPER_PATH=$(pwd)/$(dirname "$PYWRAPPER")"
    else
      echo "[setup] WARNING: pyserf shared library not found after build."
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
echo "  Storage root: $STORAGE_ROOT"
echo "  Datasets:     $STORAGE_ROOT/datasets/"
echo "  HF cache:     $STORAGE_ROOT/hf_cache/"
echo ""
echo "Next steps:"
echo "  1. Smoke test:   python main.py experiment --dataset-list Datasets/File_List/TSB-AD-U-Smoke.csv --detectors IForest"
echo "  2. Submit jobs:  bash slurm/submit.sh"
