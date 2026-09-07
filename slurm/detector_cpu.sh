#!/bin/bash
# =============================================================================
# detector_cpu.sh — Run one classical (CPU-only) detector on the full Eva set.
# Called by submit.sh with --export=DETECTOR=<name>
#
# Example:
#   sbatch --export=DETECTOR=IForest slurm/detector_cpu.sh
# =============================================================================

#SBATCH --job-name=lossyad_%x
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --error=slurm/logs/%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=1-0:0:0
#SBATCH --partition=instruction
#SBATCH --account=f2026.coms.5790.01
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trelstad@iastate.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge

source ~/.bashrc
conda activate lossyad

cd $SLURM_SUBMIT_DIR

echo "=== Detector: $DETECTOR (CPU) ==="
echo "Node: $SLURMD_NODENAME"
echo "Start: $(date)"

python main.py experiment --detectors "$DETECTOR"

echo "End: $(date)"
echo "Exit: $?"
