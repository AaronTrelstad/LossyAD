#!/bin/bash
# =============================================================================
# submit.sh — Master submission script for the LossyAD univariate experiment.
#
# Submits:
#   1. bounds job (builds CR→error-bound maps)
#   2. All 35 univariate detector jobs (depends on bounds completing)
#   3. analyze job (depends on all detector jobs completing)
#
# Usage:
#   cd /path/to/LossyAD
#   bash slurm/submit.sh
# =============================================================================

set -euo pipefail

mkdir -p slurm/logs

# ---------------------------------------------------------------------------
# 1. Bounds job — must finish before detectors start
# ---------------------------------------------------------------------------
echo "Submitting bounds job..."
BOUNDS_JID=$(sbatch --parsable slurm/bounds.sh)
echo "  bounds job ID: $BOUNDS_JID"

# ---------------------------------------------------------------------------
# 2. Univariate detector jobs — all depend on bounds completing
# ---------------------------------------------------------------------------
# Classical / CPU-only detectors  (24h walltime, no GPU)
CPU_DETECTORS=(
    KShapeAD
    Series2Graph
    MatrixProfile
    NORMA
    SAND
    Left_STAMPi
    KMeansAD_U
    POLY
    SR
    FFT
    FITS
    Sub_PCA
    Sub_KNN
    Sub_LOF
    Sub_IForest
    Sub_HBOS
    Sub_MCD
    Sub_OCSVM
    LOF
    IForest
)

# Deep-learning detectors — full Eva set (350 datasets), GPU required (48h walltime)
GPU_DETECTORS=(
    CNN
    AutoEncoder
    USAD
    Donut
    LSTMAD
    OmniAnomaly
    TranAD
    AnomalyTransformer
)

# Foundation-model detectors — FM subset (50 datasets), GPU required (72h walltime)
FM_DETECTORS=(
    MOMENT_FT
    MOMENT_ZS
    TimesNet
    TimesFM
    Chronos
    Lag_Llama
    OFA
)

DETECTOR_JIDS=()

echo ""
echo "Submitting CPU detector jobs (dependency: afterok:$BOUNDS_JID)..."
for DET in "${CPU_DETECTORS[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="lossyad_${DET}" \
        --dependency=afterok:$BOUNDS_JID \
        --export=DETECTOR=$DET \
        slurm/detector_cpu.sh)
    DETECTOR_JIDS+=($JID)
    echo "  $DET -> $JID"
done

echo ""
echo "Submitting GPU detector jobs (dependency: afterok:$BOUNDS_JID)..."
for DET in "${GPU_DETECTORS[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="lossyad_${DET}" \
        --dependency=afterok:$BOUNDS_JID \
        --export=DETECTOR=$DET \
        slurm/detector_gpu.sh)
    DETECTOR_JIDS+=($JID)
    echo "  $DET -> $JID"
done

echo ""
echo "Submitting FM detector jobs (dependency: afterok:$BOUNDS_JID)..."
for DET in "${FM_DETECTORS[@]}"; do
    JID=$(sbatch --parsable \
        --job-name="lossyad_${DET}" \
        --dependency=afterok:$BOUNDS_JID \
        --export=DETECTOR=$DET \
        slurm/detector_fm.sh)
    DETECTOR_JIDS+=($JID)
    echo "  $DET -> $JID"
done

# ---------------------------------------------------------------------------
# 3. Analyze job — runs after ALL detector jobs complete
# ---------------------------------------------------------------------------
# Build the dependency string: afterok:JID1:JID2:...:JIDn
DEP_STR=$(IFS=:; echo "afterok:${DETECTOR_JIDS[*]}")

echo ""
echo "Submitting analyze job (dependency: $DEP_STR)..."
ANALYZE_JID=$(sbatch --parsable --dependency=$DEP_STR slurm/analyze.sh)
echo "  analyze job ID: $ANALYZE_JID"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "============================================================"
echo "  All jobs submitted."
echo ""
echo "  bounds:   $BOUNDS_JID"
echo "  detectors: ${DETECTOR_JIDS[*]}"
echo "  analyze:  $ANALYZE_JID"
echo ""
echo "  Monitor with:  squeue -u \$USER"
echo "  Logs in:       slurm/logs/"
echo "============================================================"
