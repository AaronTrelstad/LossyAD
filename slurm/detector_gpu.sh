#!/bin/bash
# =============================================================================
# detector_gpu.sh — Run one deep-learning detector (GPU) on the full Eva set.
# Called by submit.sh with --export=DETECTOR=<name>
#
# Example:
#   sbatch --export=DETECTOR=LSTMAD slurm/detector_gpu.sh
# =============================================================================

#SBATCH --job-name=lossyad_%x
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=2-0:0:0
#SBATCH --partition=nova
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trelstad@iastate.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge

source ~/.bashrc
conda activate lossyad

cd $SLURM_SUBMIT_DIR

export CUDA_VISIBLE_DEVICES=0

# Redirect HuggingFace and Torch caches to HPC storage (set by setup.sh)
STORAGE_ROOT=$(cat "$SLURM_SUBMIT_DIR/.storage_root" 2>/dev/null || echo "/work/classtmp/$USER/lossyad")
export HF_HOME="$STORAGE_ROOT/hf_cache"
export TORCH_HOME="$STORAGE_ROOT/torch_cache"
mkdir -p "$HF_HOME" "$TORCH_HOME"

echo "=== Detector: $DETECTOR (GPU) ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

python main.py experiment --detectors "$DETECTOR"

echo "End: $(date)"
echo "Exit: $?"
