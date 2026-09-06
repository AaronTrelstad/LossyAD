#!/bin/bash
# =============================================================================
# detector_fm.sh — Run one foundation-model detector (GPU) on the FM subset.
# Uses TSB-AD-U-FM.csv (50 datasets) to keep walltime feasible.
# The full 350-dataset Eva run would take 5-15 days each.
#
# Called by submit.sh with --export=DETECTOR=<name>
#
# Example:
#   sbatch --export=DETECTOR=MOMENT_FT slurm/detector_fm.sh
# =============================================================================

#SBATCH --job-name=lossyad_%x
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=3-0:0:0
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

echo "=== Detector: $DETECTOR (Foundation Model, FM subset) ==="
echo "Node: $SLURMD_NODENAME"
echo "GPU:  $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Start: $(date)"

python main.py experiment --detectors "$DETECTOR" --dataset-list Datasets/File_List/TSB-AD-U-FM.csv

echo "End: $(date)"
echo "Exit: $?"
