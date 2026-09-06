# LossyAD

A benchmarking framework that evaluates how **lossy time series compression** affects **anomaly detection (AD)** performance. Built on [TSB-AD](https://github.com/TheDatumOrg/TSB-AD) and [TerseTS](https://github.com/cmcuza/TerseTS).

## Use Case

In many real-world deployments, time series data is compressed for storage or transmission before being analyzed. This framework quantifies the degradation in AD quality as a function of compression ratio — answering the question: *how much can you compress before anomaly detection breaks?*

Pipeline:
```
Raw TS → Normalize → Compress → Store → Decompress → Run AD detector → Evaluate
```

---

## Quick Start

```bash
# Full experiment (all compressors, all detectors)
python main.py

# Specific detectors and compressors
python main.py experiment --detectors KShapeAD LSTMAD IForest --compressors DWT ZFP SZ3

# Multivariate datasets
python main.py --multivariate

# Build compression ratio maps only
python main.py bounds

# Analyze results and generate charts
python main.py analyze

# Run statistical significance tests
python main.py analyze --stats
```

---

## Installation

### 1. Clone and set up environment

```bash
git clone https://github.com/AaronTrelstad/LossyAD.git
cd LossyAD
conda create -n LossyAD python=3.10
conda activate LossyAD
pip install -r requirements.txt
```

### 2. Install TerseTS (PLR compressors)

```bash
git clone https://github.com/cmcuza/TerseTS.git
cd TerseTS/bindings/python && pip install . && cd ../../..
```

### 3. Build native compressors (optional — needed for SZ3 and SERF)

```bash
bash setup.sh
```

This script builds SZ3 and SERF from source and prints the environment variables to set. The framework runs without them — they are gracefully skipped if unavailable.

### 4. Download datasets

| Dataset | URL |
|---------|-----|
| TSB-AD-U (univariate) | https://www.thedatum.org/datasets/TSB-AD-U.zip |
| TSB-AD-M (multivariate) | https://www.thedatum.org/datasets/TSB-AD-M.zip |

Extract into `Datasets/`:
```
Datasets/
  TSB-AD-U/
  TSB-AD-M/
  File_List/
    TSB-AD-U-Test.csv
    TSB-AD-M-Test.csv
```

---

## Running Experiments

### CLI reference

```
python main.py [subcommand] [options]

Subcommands:
  all         Run full pipeline: baseline + bounds + experiment + analysis (default)
  bounds      Build CR -> error-bound calibration maps only
  experiment  Run compressed experiments only (requires bounds)
  analyze     Generate charts and statistics from existing results

Options:
  --multivariate          Use TSB-AD-M (multivariate) dataset
  --detectors D [D ...]   Subset of detectors to run
  --compressors C [C ...] Subset of compressors to run
  --crs N [N ...]         Target compression ratios (default: 1 3 5 7 10 15 20 30 40 50)
  --workers N             Number of parallel detector workers (default: 1)
  --seed N                Random seed (default: 2024)
  --chart                 Save waveform comparison plots (slow)
  --stats                 Run statistical significance tests
```

### GPU usage

TSB-AD deep-learning detectors (CNN, LSTMAD, USAD, OmniAnomaly, TranAD, AnomalyTransformer, MOMENT, TimesNet, etc.) automatically detect and use CUDA.

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python main.py

# Multiple parallel workers (one per GPU)
LOSSYAD_WORKERS=4 python main.py
```

### Runtime estimate (TSB-AD-U-Test, 10 datasets)

| Detector family | Approx. time (single GPU) |
|----------------|--------------------------|
| Classical (IForest, LOF, Sub_*, KShapeAD, MatrixProfile, etc.) | 2-6 hours |
| Deep learning (CNN, LSTMAD, USAD, OmniAnomaly, TranAD, AnomalyTransformer) | 8-20 hours |
| Foundation models (MOMENT, TimesNet, TimesFM, Chronos, Lag_Llama, OFA) | 12-30 hours |
| **Total (35 detectors, 12 compressors, 10 CRs)** | **~2-4 days** |

Each result CSV is written as soon as its (detector, compressor, CR) combination finishes. If a run is interrupted, it resumes from where it left off — completed files are never re-run.

For the full TSB-AD-U evaluation set (350 datasets), multiply times by ~35.

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOSSYAD_WORKERS` | `1` | Parallel detector workers |
| `SZ3_LIB_PATH` | auto-detected | Path to `libSZ3c.dylib` / `.so` |
| `SERF_PYWRAPPER_PATH` | auto-detected | Path to `pyserf*.so` |

---

## Compressors

All 13 compressors are organized by algorithmic family following the TerseTS taxonomy (EDBT 2026):

| Name | Family | Method | Source |
|------|--------|--------|--------|
| `NONE` | Baseline | Passthrough (no compression) | — |
| `ZFP` | Error-bounded | ZFP floating-point compression | `zfpy` |
| `SZ3` | Error-bounded | SZ3 prediction+quantization | native build |
| `DWT` | Transform | Discrete Wavelet Transform (db4), detail thresholding | `PyWavelets` |
| `PIP` | Perceptual/Shape | Perceptually Important Points + linear interpolation | `src/pip_helpers.py` |
| `VW` | Perceptual/Shape | Visvalingam-Whyatt simplification | `TerseTS` |
| `QUANT` | Quantization | Float64 → {4..16}-bit uniform quantization | built-in |
| `GORILLA` | XOR/Streaming | Gorilla XOR delta encoding (Pelkonen et al., VLDB 2015) | built-in (pure Python) |
| `SERF` | XOR/Streaming | SERF — lossless-quality XOR float compressor | native build |
| `MP` | PLR | MixPiece — mixed piecewise linear (Xenofontos et al., VLDB 2023) | `TerseTS` |
| `SIMPIE` | PLR | SimPiece — simplified piecewise linear | `TerseTS` |
| `SWING` | PLR | Swing Filter — one-pass PLR | `TerseTS` |
| `SLIDE` | PLR | Slide Filter — sliding-window PLR | `TerseTS` |

All compressors support **multivariate** data via per-channel compression.

Compressors with optional native dependencies (SZ3, SERF, ZFP) are gracefully excluded at startup if their libraries are not found — no code changes needed.

---

## Anomaly Detectors

### Univariate (35 methods, TSB-AD-U)

| Family | Methods |
|--------|---------|
| Shape / Pattern | KShapeAD, Series2Graph, MatrixProfile, NORMA, SAND, Left_STAMPi, KMeansAD_U |
| Statistical / Frequency | POLY, SR, FFT, FITS |
| Subspace / Distance | Sub_PCA, Sub_KNN, Sub_LOF, Sub_IForest, Sub_HBOS, Sub_MCD, Sub_OCSVM |
| Density | LOF |
| Ensemble / Tree | IForest |
| Deep Reconstruction | CNN, AutoEncoder, USAD, Donut |
| Deep Prediction / VAE | LSTMAD, OmniAnomaly, TranAD |
| Transformer | AnomalyTransformer |
| Foundation Models | MOMENT_FT, MOMENT_ZS, TimesNet, TimesFM, Chronos, Lag_Llama, OFA |

### Multivariate (24 methods, TSB-AD-M)

| Family | Methods |
|--------|---------|
| Subspace / Linear | PCA, RobustPCA |
| Density / Proximity | LOF, KNN, MCD, OCSVM |
| Statistical / Histogram | HBOS, CBLOF, COPOD |
| Ensemble / Tree | IForest, EIF |
| Shape / Pattern | KShapeAD, KMeansAD |
| Deep Reconstruction | CNN, AutoEncoder, USAD, Donut, FITS |
| Deep Prediction / VAE | LSTMAD, OmniAnomaly, TranAD |
| Transformer | AnomalyTransformer |
| Foundation Models | TimesNet, OFA |

---

## Output Structure

```
results/
  original/
    <detector>.csv          # Baseline: raw uncompressed data
  <COMPRESSOR>/
    <detector>/
      <cr>.csv              # Compressed: CR=1,3,5,...,50

cr_bound_maps/
  <COMPRESSOR>.json         # Per-dataset CR -> error-bound calibration

charts/
  <detector>/
    <detector>_<compressor>.png          # F1 vs CR (single compressor)
    <detector>_cross_method.png          # All compressors overlaid
    <detector>_normalized_degradation.png # F1/baseline vs CR
    <detector>_all_metrics.png           # AUC-PR, AUC-ROC, VUS-PR, VUS-ROC
  affinity_heatmap.png                   # Compressor x detector mean delta-F1
  reconstruction_correlation.png         # RMSE vs delta-F1 Spearman scatter
  safe_cr_budget.png                     # Max CR within 5% F1 degradation

results/statistical/
  wilcoxon_results.csv                   # Per (compressor, CR, detector): signed-rank test
  wilcoxon_results.tex                   # LaTeX table
  friedman_results.csv                   # Per (CR, detector): Friedman test across compressors
  significance_heatmap.png               # -log10(p) heatmap
```

Each result CSV has columns:
```
Dataset, Time, RMSE, MAE, MaxAE, AUC_PR, AUC_ROC, VUS_PR, VUS_ROC, Standard_F1, ...
```

---

## Extending the Framework

### Add a new compressor

1. Create a class with `compress(data)` and `decompress(compressed, shape, dtype)` methods.
2. Add it to `_COMPRESSOR_REGISTRY` in `src/compression_methods.py`.
3. If it has an optional native dependency, raise `RuntimeError` on import failure — it will be auto-excluded.

### Add a new detector

Detectors are pulled directly from TSB-AD. If a detector appears in `Optimal_Uni_algo_HP_dict` or `Optimal_Multi_algo_HP_dict`, add its name to `_DESIRED_UNI_AD_METHODS` or `_DESIRED_MULTI_AD_METHODS` in `src/configs.py`.

### Change the dataset

Update `dataset_dir` and `dataset_list` in `ExperimentConfig` (or pass `--multivariate`). Any TSB-AD-format CSV with a `Label` column works.

---

## Smoke Tests

```bash
# Test all compressors (1-D and 2-D round-trip)
python test.py --compressors

# Test all classical detectors
python test.py --detectors --fast

# Full test (includes deep learning — requires GPU)
python test.py --detectors
```

---

## Citation

If you use this framework, please cite:

```bibtex
@software{lossyad2025,
  title  = {LossyAD: Benchmarking Lossy Compression for Time Series Anomaly Detection},
  author = {Trelstad, Aaron},
  year   = {2025},
  url    = {https://github.com/AaronTrelstad/LossyAD}
}
```

And the underlying frameworks:

```bibtex
@article{paparrizos2022tsb,
  title   = {TSB-UAD: An End-to-End Benchmark Suite for Univariate Time-Series Anomaly Detection},
  author  = {Paparrizos, John and others},
  journal = {PVLDB},
  year    = {2022}
}

@inproceedings{xenofontos2026tersetts,
  title     = {TerseTS: A Library for Compressing Time Series with Error Guarantees},
  author    = {Xenofontos, Christos and others},
  booktitle = {EDBT},
  year      = {2026}
}
```
