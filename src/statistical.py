"""
statistical.py — significance tests for the LossyAD benchmark.

Functions
---------
run_statistical_tests(args)
    Full pipeline: load results, run Wilcoxon and Friedman tests, save outputs.

Output files
------------
  results/statistical/wilcoxon_results.csv   — per-(compressor, CR, detector) Wilcoxon test
  results/statistical/friedman_results.csv   — per-(CR, detector) Friedman test
  results/statistical/wilcoxon_results.tex   — LaTeX table
  charts/significance_heatmap.png            — p-value heatmap
"""

import os
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAVE_SEABORN = True
except ImportError:
    _HAVE_SEABORN = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_f1_col(df: pd.DataFrame) -> str | None:
    for candidate in ["F1", "F", "f1", "f"]:
        if candidate in df.columns:
            return candidate
    return None


def _load_per_dataset_f1(results_dir: str, compressor: str, detector: str, cr: float) -> pd.Series | None:
    """
    Load per-dataset F1 scores from a single per-CR CSV.
    Returns a Series indexed by Dataset, or None if not found.
    """
    path = os.path.join(results_dir, compressor, detector, f"{cr}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        col = _get_f1_col(df)
        if col is None or "Dataset" not in df.columns:
            return None
        return df.set_index("Dataset")[col]
    except Exception:
        return None


def _load_baseline_per_dataset(results_dir: str, detector: str) -> pd.Series | None:
    path = os.path.join(results_dir, "original", f"{detector}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        col = _get_f1_col(df)
        if col is None or "Dataset" not in df.columns:
            return None
        return df.set_index("Dataset")[col]
    except Exception:
        return None


def _rank_biserial(x, y) -> float:
    """Rank-biserial correlation as effect size for Wilcoxon signed-rank test."""
    n = len(x)
    if n == 0:
        return float("nan")
    diffs = np.array(x) - np.array(y)
    nonzero = diffs[diffs != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    r_plus  = np.sum(ranks[nonzero > 0])
    r_minus = np.sum(ranks[nonzero < 0])
    return (r_plus - r_minus) / (r_plus + r_minus)


# ---------------------------------------------------------------------------
# Wilcoxon signed-rank tests
# ---------------------------------------------------------------------------

def wilcoxon_vs_baseline(args) -> pd.DataFrame:
    """
    For each (compressor, CR, detector), run a Wilcoxon signed-rank test
    comparing per-dataset F1 scores against the uncompressed baseline.

    Returns a DataFrame with columns:
      compressor, cr, detector, n, statistic, p_value, effect_size,
      mean_f1_baseline, mean_f1_compressed, mean_delta_f1
    """
    rows = []

    for compressor in os.listdir(args.results_dir):
        comp_path = os.path.join(args.results_dir, compressor)
        if not os.path.isdir(comp_path) or compressor in ("original", "statistical"):
            continue

        for detector in os.listdir(comp_path):
            det_path = os.path.join(comp_path, detector)
            if not os.path.isdir(det_path):
                continue

            baseline = _load_baseline_per_dataset(args.results_dir, detector)
            if baseline is None:
                continue

            for fname in sorted(os.listdir(det_path)):
                if not fname.endswith(".csv") or fname == "summary.csv":
                    continue
                try:
                    cr = float(os.path.splitext(fname)[0])
                except ValueError:
                    continue

                compressed = _load_per_dataset_f1(args.results_dir, compressor, detector, cr)
                if compressed is None:
                    continue

                # Align on shared datasets
                shared = baseline.index.intersection(compressed.index)
                if len(shared) < 5:
                    continue

                b = baseline[shared].values.astype(float)
                c = compressed[shared].values.astype(float)

                try:
                    stat, pval = stats.wilcoxon(c, b, alternative="two-sided", zero_method="wilcox")
                    eff = _rank_biserial(c, b)
                except Exception:
                    stat, pval, eff = float("nan"), float("nan"), float("nan")

                rows.append({
                    "compressor":          compressor,
                    "cr":                  cr,
                    "detector":            detector,
                    "n":                   len(shared),
                    "statistic":           round(stat, 4),
                    "p_value":             round(pval, 6),
                    "effect_size":         round(eff, 4),
                    "mean_f1_baseline":    round(float(b.mean()), 4),
                    "mean_f1_compressed":  round(float(c.mean()), 4),
                    "mean_delta_f1":       round(float((c - b).mean()), 4),
                })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Friedman test (multi-compressor comparison at each CR)
# ---------------------------------------------------------------------------

def friedman_across_compressors(args) -> pd.DataFrame:
    """
    At each (CR, detector), run a Friedman test across all compressors using
    per-dataset F1 as the repeated measure (datasets = blocks).

    Returns a DataFrame with columns:
      cr, detector, n_compressors, n_datasets, statistic, p_value
    """
    # Collect {(cr, detector): {compressor: Series(dataset→f1)}}
    data: dict[tuple, dict[str, pd.Series]] = defaultdict(dict)

    for compressor in os.listdir(args.results_dir):
        comp_path = os.path.join(args.results_dir, compressor)
        if not os.path.isdir(comp_path) or compressor in ("original", "statistical"):
            continue
        for detector in os.listdir(comp_path):
            det_path = os.path.join(comp_path, detector)
            if not os.path.isdir(det_path):
                continue
            for fname in sorted(os.listdir(det_path)):
                if not fname.endswith(".csv") or fname == "summary.csv":
                    continue
                try:
                    cr = float(os.path.splitext(fname)[0])
                except ValueError:
                    continue
                s = _load_per_dataset_f1(args.results_dir, compressor, detector, cr)
                if s is not None:
                    data[(cr, detector)][compressor] = s

    rows = []
    for (cr, detector), comp_map in data.items():
        if len(comp_map) < 3:
            continue
        # Align on shared datasets across all compressors
        shared = None
        for s in comp_map.values():
            shared = s.index if shared is None else shared.intersection(s.index)
        if shared is None or len(shared) < 5:
            continue

        groups = [comp_map[c][shared].values.astype(float) for c in sorted(comp_map)]
        try:
            stat, pval = stats.friedmanchisquare(*groups)
        except Exception:
            stat, pval = float("nan"), float("nan")

        rows.append({
            "cr":            cr,
            "detector":      detector,
            "n_compressors": len(comp_map),
            "n_datasets":    len(shared),
            "statistic":     round(stat, 4),
            "p_value":       round(pval, 6),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Significance heatmap
# ---------------------------------------------------------------------------

def plot_significance_heatmap(wilcoxon_df: pd.DataFrame):
    """
    Heatmap of −log10(p_value) for the Wilcoxon test at each
    (compressor × detector) pair, averaged across CRs.
    """
    if wilcoxon_df.empty:
        print("[stats] No Wilcoxon results to plot.")
        return

    pivot = (
        wilcoxon_df.groupby(["compressor", "detector"])["p_value"]
        .mean()
        .unstack("compressor")
    )

    log_p = -np.log10(pivot.clip(lower=1e-10))

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5),
                                    max(5, len(pivot.index)  * 0.8)))

    if _HAVE_SEABORN:
        sns.heatmap(
            log_p, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "−log₁₀(p-value)   [higher = more significant]"},
        )
    else:
        im = ax.imshow(log_p.values, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(log_p.columns)))
        ax.set_xticklabels(log_p.columns, rotation=45, ha="right")
        ax.set_yticks(range(len(log_p.index)))
        ax.set_yticklabels(log_p.index)
        for i in range(len(log_p.index)):
            for j in range(len(log_p.columns)):
                val = log_p.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="−log₁₀(p-value)")

    # Draw significance threshold line (p=0.05 → −log10 = 1.3)
    ax.set_title(
        "Wilcoxon Significance: Compressed vs Uncompressed F1\n"
        "(−log₁₀ p-value averaged across CRs; >1.3 significant at p<0.05)"
    )
    ax.set_xlabel("Compressor")
    ax.set_ylabel("Detector")
    fig.tight_layout()

    os.makedirs("charts", exist_ok=True)
    fig.savefig("charts/significance_heatmap.png", dpi=150)
    plt.close(fig)
    print("[stats] Saved charts/significance_heatmap.png")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_statistical_tests(args):
    """Run all significance tests and save outputs."""
    out_dir = os.path.join(args.results_dir, "statistical")
    os.makedirs(out_dir, exist_ok=True)

    print("[stats] Running Wilcoxon signed-rank tests …")
    wilcoxon_df = wilcoxon_vs_baseline(args)
    if not wilcoxon_df.empty:
        wilcoxon_df.to_csv(os.path.join(out_dir, "wilcoxon_results.csv"), index=False)
        # LaTeX table: pivot mean p-value across CRs
        pivot = (
            wilcoxon_df.groupby(["compressor", "detector"])["p_value"]
            .mean()
            .unstack("compressor")
        )
        with open(os.path.join(out_dir, "wilcoxon_results.tex"), "w") as f:
            f.write(pivot.to_latex(float_format="%.4f", na_rep="—"))
        print(f"[stats] Wilcoxon results → {out_dir}/wilcoxon_results.{{csv,tex}}")
        plot_significance_heatmap(wilcoxon_df)
    else:
        print("[stats] No Wilcoxon results (missing per-dataset result files).")

    print("[stats] Running Friedman tests …")
    friedman_df = friedman_across_compressors(args)
    if not friedman_df.empty:
        friedman_df.to_csv(os.path.join(out_dir, "friedman_results.csv"), index=False)
        print(f"[stats] Friedman results → {out_dir}/friedman_results.csv")
    else:
        print("[stats] No Friedman results.")

    print("[stats] Done.")
    return wilcoxon_df, friedman_df
