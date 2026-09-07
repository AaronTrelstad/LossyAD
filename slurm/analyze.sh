#!/bin/bash
# =============================================================================
# analyze.sh — Generate all charts and statistical tests after experiments complete.
#
# Submit after all detector jobs finish:
#   sbatch --dependency=afterok:<job1>:<job2>:... slurm/analyze.sh
# Or run manually once results are in:
#   sbatch slurm/analyze.sh
# =============================================================================

#SBATCH --job-name=lossyad_analyze
#SBATCH --output=slurm/logs/analyze_%j.out
#SBATCH --error=slurm/logs/analyze_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --time=0-2:0:0
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

echo "=== Running analysis ==="
echo "Start: $(date)"

python main.py analyze

echo "End: $(date)"
