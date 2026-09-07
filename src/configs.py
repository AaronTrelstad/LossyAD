"""
configs.py — central configuration for the LossyAD benchmark.
"""

import os
import random

import numpy as np
import torch
from enum import Enum

from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

from .compression_methods import get_available_compressors


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        names  = [torch.cuda.get_device_name(i) for i in range(n_gpus)]
        print(f"CUDA: {n_gpus} GPU(s) available — {names}")
        print(f"      Active device: cuda:{torch.cuda.current_device()}")
        # TSB-AD deep-learning detectors (CNN, LSTMAD, USAD, MOMENT) detect CUDA
        # internally and move models to GPU automatically.
        # To pin a specific GPU:  CUDA_VISIBLE_DEVICES=0 python main.py
    else:
        print("CUDA: not available — running on CPU")
        print("      GPU is required for CNN / LSTMAD / USAD / MOMENT detectors.")
    print(f"cuDNN: {torch.backends.cudnn.version()}")


# ---------------------------------------------------------------------------
# MethodType — dynamically built from available compressors
# ---------------------------------------------------------------------------

_available = get_available_compressors()

# Build the enum members: name → compressor class
MethodType = Enum("MethodType", {name: cls for name, cls in _available.items()})
"""
Enum of all compressor methods whose native dependencies are satisfied.
Members are added/removed automatically — no code change needed when
installing or removing an optional compressor's native library.
"""


# ---------------------------------------------------------------------------
# AD method registry — Univariate (TSB-AD-U)
# ---------------------------------------------------------------------------
# Representative selection spanning the major algorithmic families in the
# time-series anomaly detection literature.  All methods are sourced from
# TSB-AD (Paparrizos et al., PVLDB 2022 / 2024).
#
# Family                  | Principle                                 | Methods
# ----------------------- | ----------------------------------------- | -------
# Subspace / distance     | Anomalies are distant in projected space   | Sub_PCA, Sub_KNN, Sub_LOF
# Density                 | Low-density regions are anomalous          | LOF
# Ensemble / tree         | Anomalies are isolated quickly             | IForest, Sub_IForest
# Shape / pattern         | Anomalies deviate from recurring patterns  | KShapeAD, MatrixProfile, NORMA
# Frequency / spectral    | Anomalies cause spectral residuals         | SR, FFT
# Deep reconstruction     | High reconstruction error = anomaly        | CNN, USAD
# Deep prediction         | High prediction error = anomaly            | LSTMAD, OmniAnomaly, TranAD
# Transformer             | Attention-based reconstruction             | AnomalyTransformer
# Foundation model        | Pre-trained on large TS corpora            | MOMENT_FT, MOMENT_ZS, TimesNet

_DESIRED_UNI_AD_METHODS = [
    # ── Shape / pattern ──────────────────────────────────────────────────────
    # Leaderboard rank: KShapeAD 1st (0.40), Series2Graph 4th (0.39), MatrixProfile 6th
    "KShapeAD",      # k-Shape clustering + shape-based distance (Paparrizos & Gravano, SIGMOD 2015)
    "Series2Graph",  # Graph of transition probabilities between discretised values (Boniol et al., PVLDB 2020)
    "MatrixProfile", # Nearest-neighbour distance in subsequence space (Yeh et al., ICDM 2016)
    "NORMA",         # Normal pattern extraction via matrix profile (Boniol et al., SIGMOD 2021)
    "SAND",          # Streaming anomaly detection via normal pattern update (Boniol et al., PVLDB 2021)
    "Left_STAMPi",   # Streaming matrix profile (left-side) for online anomaly detection
    "KMeansAD_U",    # k-Means clustering; distance to nearest centroid (univariate variant)
    # ── Statistical / frequency ──────────────────────────────────────────────
    # Leaderboard rank: POLY 4th (0.39), SR 8th
    "POLY",          # Polynomial fitting residual — simple parametric baseline
    "SR",            # Spectral Residual — Fourier magnitude residual (Ren et al., KDD 2019)
    "FFT",           # FFT reconstruction error
    "FITS",          # Frequency Interpolation TS (Zhou et al., ICLR 2024)
    # ── Subspace / distance ──────────────────────────────────────────────────
    "Sub_PCA",       # PCA on subsequences; distance to subspace
    "Sub_KNN",       # KNN on subsequences; distance to k-th neighbour
    "Sub_LOF",       # LOF on subsequences; local density ratio
    "Sub_IForest",   # IForest on subsequences (leaderboard rank: 12th)
    "Sub_HBOS",      # Histogram-Based Outlier Score on subsequences
    "Sub_MCD",       # Minimum Covariance Determinant on subsequences
    "Sub_OCSVM",     # One-Class SVM on subsequences
    # ── Density ──────────────────────────────────────────────────────────────
    "LOF",           # Local Outlier Factor (Breunig et al., SIGMOD 2000)
    # ── Ensemble / tree ──────────────────────────────────────────────────────
    "IForest",       # Isolation Forest (Liu et al., ICDM 2008)
    # ── Deep reconstruction ───────────────────────────────────────────────────
    "CNN",           # 1-D CNN autoencoder
    "AutoEncoder",   # Fully-connected autoencoder (reconstruction error)
    "USAD",          # Adversarial encoder-decoder (Audibert et al., KDD 2020)
    "Donut",         # VAE with Donut latent space (Xu et al., WWW 2018)
    # ── Deep prediction / VAE ─────────────────────────────────────────────────
    "LSTMAD",        # LSTM prediction error (Malhotra et al., ESANN 2015)
    "OmniAnomaly",   # Stochastic LSTM-VAE (Su et al., KDD 2019)
    "TranAD",        # Transformer reconstruction + prediction (Tuli et al., VLDB 2022)
    # ── Transformer ───────────────────────────────────────────────────────────
    "AnomalyTransformer",  # Anomaly-attention mechanism (Xu et al., ICLR 2022)
    # ── Foundation / pre-trained ──────────────────────────────────────────────
    # Leaderboard rank: MOMENT_FT 2nd (0.39), MOMENT_ZS 5th (0.38), TimesFM 9th, Chronos 10th, Lag_Llama 11th
    "MOMENT_FT",     # MOMENT — fine-tuned (Goswami et al., ICML 2024)
    "MOMENT_ZS",     # MOMENT — zero-shot
    "TimesNet",      # 2-D temporal variation (Wu et al., ICLR 2023)
    "TimesFM",       # TimesFM — Google foundation model for TS
    "Chronos",       # Chronos — Amazon pre-trained TS model (Ansari et al., 2024)
    "Lag_Llama",     # Lag-Llama — LLaMA adapted for TS forecasting
    "OFA",           # One-Fits-All — GPT-2 adapted for TS (Zhou et al., NeurIPS 2023)
]

# ---------------------------------------------------------------------------
# AD method registry — Multivariate (TSB-AD-M)
# ---------------------------------------------------------------------------
# Family                  | Principle                                  | Methods
# ----------------------- | ------------------------------------------ | -------
# Subspace / linear       | PCA / robust PCA of feature matrix         | PCA, RobustPCA
# Density                 | Low-density neighbours are anomalous       | LOF, KNN
# Ensemble / tree         | Isolation depth                            | IForest
# Shape / pattern         | Shape-based clustering                     | KShapeAD
# Deep reconstruction     | High reconstruction error                  | CNN, USAD
# Deep prediction / VAE   | Prediction / stochastic model error        | LSTMAD, OmniAnomaly, TranAD
# Transformer             | Attention-based reconstruction             | AnomalyTransformer
# Foundation model        | Pre-trained on large TS corpora            | TimesNet, OFA

_DESIRED_MULTI_AD_METHODS = [
    # ── Subspace / linear ────────────────────────────────────────────────────
    "PCA",                # Standard PCA reconstruction error
    "RobustPCA",          # Robust PCA via RPCA (Candès et al., JACM 2011)
    # ── Density / proximity ──────────────────────────────────────────────────
    "LOF",                # Local Outlier Factor (Breunig et al., SIGMOD 2000)
    "KNN",                # k-Nearest Neighbour distance
    "MCD",                # Minimum Covariance Determinant
    "OCSVM",              # One-Class SVM
    # ── Statistical / histogram ──────────────────────────────────────────────
    "HBOS",               # Histogram-Based Outlier Score (Goldstein & Dengel, 2012)
    "CBLOF",              # Cluster-Based Local Outlier Factor
    "COPOD",              # Copula-Based Outlier Detection (Li et al., ICDM 2020)
    # ── Ensemble / tree ──────────────────────────────────────────────────────
    "IForest",            # Isolation Forest (Liu et al., ICDM 2008)
    "EIF",                # Extended Isolation Forest (Hariri et al., TKDE 2019)
    # ── Shape / pattern ──────────────────────────────────────────────────────
    "KShapeAD",           # k-Shape on multivariate subsequences
    "KMeansAD",           # k-Means clustering; distance to nearest centroid
    # ── Deep reconstruction ───────────────────────────────────────────────────
    "CNN",                # 1-D CNN autoencoder per channel
    "AutoEncoder",        # Fully-connected autoencoder
    "USAD",               # Adversarial encoder-decoder (Audibert et al., KDD 2020)
    "Donut",              # VAE with Donut latent space (Xu et al., WWW 2018)
    "FITS",               # Frequency Interpolation TS (Zhou et al., ICLR 2024)
    # ── Deep prediction / VAE ─────────────────────────────────────────────────
    "LSTMAD",             # LSTM prediction error (Malhotra et al., ESANN 2015)
    "OmniAnomaly",        # Stochastic LSTM-VAE (Su et al., KDD 2019) — multivariate-native
    "TranAD",             # Transformer reconstruction + prediction (Tuli et al., VLDB 2022)
    # ── Transformer ───────────────────────────────────────────────────────────
    "AnomalyTransformer", # Anomaly-attention mechanism (Xu et al., ICLR 2022)
    # ── Foundation models ─────────────────────────────────────────────────────
    "TimesNet",           # 2-D temporal variation (Wu et al., ICLR 2023)
    "OFA",                # One-Fits-All — GPT-2 adapted for TS (Zhou et al., NeurIPS 2023)
]

# Filter to methods actually installed in the current TSB-AD version
AVAILABLE_UNI_AD_METHODS: list[str] = [
    m for m in _DESIRED_UNI_AD_METHODS if m in Optimal_Uni_algo_HP_dict
]
AVAILABLE_MULTI_AD_METHODS: list[str] = [
    m for m in _DESIRED_MULTI_AD_METHODS if m in Optimal_Multi_algo_HP_dict
]

# Backwards-compatible alias (used by experiment.py before multivariate flag)
AVAILABLE_AD_METHODS = AVAILABLE_UNI_AD_METHODS

# ---------------------------------------------------------------------------
# Reduced method sets — one best-performing method per algorithmic family.
# Selected from the TSB-AD public leaderboard (VLDBj 2025 version).
# Full lists above are kept for completeness; use these for the main experiment
# to keep wall time feasible.
# ---------------------------------------------------------------------------

# Univariate: 13 methods covering 13 distinct families
# Family               | Method          | Leaderboard (TSB-AD-U)
# -------------------- | --------------- | ----------------------
# Shape / pattern      | KShapeAD        | 0.40  (rank 1)
# Statistical          | POLY            | 0.39  (rank 4)
# Foundation FT        | MOMENT_FT       | 0.39  (rank 5)
# Foundation ZS        | MOMENT_ZS       | 0.38  (rank 6)
# Clustering           | KMeansAD_U      | 0.37
# Deep reconstruction  | USAD            | 0.36
# Deep prediction      | LSTMAD          | 0.33
# Spectral residual    | SR              | 0.32
# Ensemble / tree      | IForest         | 0.30
# Forecasting FM       | TimesFM         | 0.30
# Density              | LOF             | 0.25
# Subspace             | Sub_IForest     | 0.22
# Transformer          | AnomalyTransformer | 0.12
_REDUCED_UNI_AD_METHODS = [
    "KShapeAD",
    "POLY",
    "MOMENT_FT",
    "MOMENT_ZS",
    "KMeansAD_U",
    "USAD",
    "LSTMAD",
    "SR",
    "IForest",
    "TimesFM",
    "LOF",
    "Sub_IForest",
    "AnomalyTransformer",
]

# Multivariate: 9 methods covering 9 distinct families
# Family               | Method           | Leaderboard (TSB-AD-M)
# -------------------- | ---------------- | ----------------------
# Subspace / linear    | PCA              | 0.31
# Deep reconstruction  | CNN              | 0.31
# Deep prediction      | OmniAnomaly      | 0.31
# Shape / clustering   | KMeansAD         | 0.29
# Statistical          | CBLOF            | 0.27
# Density              | MCD              | 0.27
# Ensemble / tree      | EIF              | 0.21
# Foundation           | OFA              | 0.21
# Transformer          | AnomalyTransformer | 0.12
_REDUCED_MULTI_AD_METHODS = [
    "PCA",
    "CNN",
    "OmniAnomaly",
    "KMeansAD",
    "CBLOF",
    "MCD",
    "EIF",
    "OFA",
    "AnomalyTransformer",
]

# Compressors: one best-in-class per compression paradigm
# Family                      | Compressor | Rationale
# --------------------------- | ---------- | ---------
# Baseline (no compression)   | NONE       | Reference point
# Error-bounded (prediction)  | SZ3        | State-of-the-art scientific lossy compression
# Error-bounded (transform)   | ZFP        | Widely used in HPC; well-studied
# Wavelet / transform         | DWT        | Classic transform coding baseline
# Scalar quantization         | QUANT      | Simple uniform quantization
# Point selection             | PIP        | Perceptually important points
# Encoding                    | GORILLA    | Streaming float compression (Facebook)
REDUCED_COMPRESSOR_NAMES: list[str] = ["NONE", "SZ3", "ZFP", "DWT", "QUANT", "GORILLA"]

REDUCED_UNI_AD_METHODS: list[str] = [
    m for m in _REDUCED_UNI_AD_METHODS if m in Optimal_Uni_algo_HP_dict
]
REDUCED_MULTI_AD_METHODS: list[str] = [
    m for m in _REDUCED_MULTI_AD_METHODS if m in Optimal_Multi_algo_HP_dict
]

_missing_uni   = set(_DESIRED_UNI_AD_METHODS)   - set(AVAILABLE_UNI_AD_METHODS)
_missing_multi = set(_DESIRED_MULTI_AD_METHODS) - set(AVAILABLE_MULTI_AD_METHODS)
if _missing_uni:
    print(f"[configs] Univariate   detectors not in TSB-AD (skipped): {sorted(_missing_uni)}")
if _missing_multi:
    print(f"[configs] Multivariate detectors not in TSB-AD (skipped): {sorted(_missing_multi)}")


# ---------------------------------------------------------------------------
# Experiment configuration
# ---------------------------------------------------------------------------

class ExperimentConfig:
    """
    Central configuration for the experiment pipeline.

    Parameters
    ----------
    multivariate : bool
        If True, use the TSB-AD-M (multivariate) dataset and file list.
        If False (default), use TSB-AD-U (univariate).

    All compressors handle multivariate data by compressing each channel
    independently and packing the results into a single byte string.

    Environment variables
    ---------------------
    LOSSYAD_WORKERS : int
        Parallelism for the detector loop.  Defaults to 1.  Set higher on
        multi-GPU nodes (one worker per GPU is typical).
    """

    @staticmethod
    def _storage_root() -> str | None:
        """
        Return the storage root for large output files, or None to use repo-relative paths.

        Priority:
          1. LOSSYAD_STORAGE_ROOT environment variable
          2. .storage_root file in the repo root (written by setup.sh)
        """
        env = os.environ.get("LOSSYAD_STORAGE_ROOT")
        if env:
            return env
        dot = os.path.join(os.path.dirname(__file__), "..", ".storage_root")
        dot = os.path.abspath(dot)
        if os.path.exists(dot):
            with open(dot) as f:
                val = f.read().strip()
            if val:
                return val
        return None

    def __init__(self, multivariate: bool = False):
        self.seed         = 2024
        self.multivariate = multivariate

        # Results and bound-map directories.
        # On HPC: set LOSSYAD_STORAGE_ROOT (or write path to .storage_root) to
        # redirect output to /work/classtmp or similar large-storage location.
        _storage = self._storage_root()
        self.results_dir = os.path.join(_storage, "results") if _storage else "results"
        self.cr_map_dir  = os.path.join(_storage, "cr_bound_maps") if _storage else "cr_bound_maps"

        if multivariate:
            self.dataset_dir  = "Datasets/TSB-AD-M"
            self.dataset_list = "Datasets/File_List/TSB-AD-M-Eva.csv"
        else:
            self.dataset_dir  = "Datasets/TSB-AD-U"
            self.dataset_list = "Datasets/File_List/TSB-AD-U-Eva.csv"

        # Target compression ratios to sweep
        self.compression_ratios = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50]

        # Error-bound grid for CR→bound calibration.
        # Range extended to 0.99 so DWT / QUANT can reach very high CRs.
        self.error_bounds = np.linspace(0, 0.99, 200)

        # AD methods — reduced set (one per family) by default.
        # Pass full=True or override via CLI --detectors to use all methods.
        if multivariate:
            self.ad_methods = list(REDUCED_MULTI_AD_METHODS)
        else:
            self.ad_methods = list(REDUCED_UNI_AD_METHODS)

        # Compressors — reduced set (one per family) by default.
        # Pass full=True or override via CLI --compressors to use all.
        name_map = {m.name: m for m in MethodType}
        self.compressors = [name_map[n] for n in REDUCED_COMPRESSOR_NAMES if n in name_map]

        # Worker count for the detector parallel loop.
        self.n_workers = int(os.environ.get("LOSSYAD_WORKERS", 1))

        # Save original-vs-decompressed waveform plots (slow, off by default)
        self.chart = False


# ---------------------------------------------------------------------------
# Analysis configuration
# ---------------------------------------------------------------------------

class AnalysisConfig:
    """Configuration for the analysis / plotting pipeline."""

    def __init__(self, multivariate: bool = False):
        self.multivariate = multivariate
        self.results_dir  = "results/"
        self.ad_methods   = list(AVAILABLE_AD_METHODS)
