#!/bin/bash
# =============================================================================
# bounds.sh — Build CR→error-bound calibration maps for all compressors.
# Must complete before any detector jobs are submitted.
#
# Submit:  sbatch slurm/bounds.sh
# =============================================================================

#SBATCH --job-name=lossyad_bounds
#SBATCH --output=slurm/logs/bounds_%j.out
#SBATCH --error=slurm/logs/bounds_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --time=0-4:0:0
#SBATCH --partition=nova
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=trelstad@iastate.edu

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
module purge

source ~/.bashrc
conda activate lossyad

cd $SLURM_SUBMIT_DIR

mkdir -p slurm/logs

echo "=== Building CR→bound maps ==="
echo "Node: $SLURMD_NODENAME"
echo "Tasks: $SLURM_NTASKS_PER_NODE"
echo "Start: $(date)"

python main.py bounds

echo "End: $(date)"
echo "=== Done ==="
