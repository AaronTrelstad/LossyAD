"""
analysis.py — aggregation and visualisation for the LossyAD benchmark.

Functions
---------
calc_aggregate          : Build per-(compressor, detector) summary CSVs from per-CR files.
plot_per_method         : Individual F1-vs-CR chart for each (compressor, detector) pair.
plot_cross_method       : All compressors overlaid on one F1-vs-CR chart per detector.
plot_normalized_degradation : Relative F1 vs CR for all compressors.
plot_all_metrics        : One chart per metric (AUC-PR, AUC-ROC, VUS-PR, VUS-ROC, F1).
plot_affinity_heatmap   : Compressor × detector sensitivity heatmap.
plot_reconstruction_correlation : Scatter of RMSE vs ΔF1 with Spearman r.
plot_safe_cr_budget     : Table + bar chart of the maximum "safe" CR per (compressor, detector).
run_analysis            : Run the full analysis pipeline.
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

from concurrent.futures import ThreadPoolExecutor, as_completed


# ---------------------------------------------------------------------------
# Known metric column names (in order of preference).
# TSB-AD typically returns: AUC_PR, AUC_ROC, VUS_PR, VUS_ROC, Precision,
# Recall, F, Rprecision, PRAUC  (exact spelling varies by version).
# ---------------------------------------------------------------------------
_METRIC_ALIASES = {
    "F1":      ["F1", "F", "f1", "f"],
    "AUC_PR":  ["AUC_PR", "AUC-PR", "AUCPR", "auc_pr"],
    "AUC_ROC": ["AUC_ROC", "AUC-ROC", "AUCROC", "auc_roc"],
    "VUS_PR":  ["VUS_PR", "VUS-PR", "VUSPR", "vus_pr"],
    "VUS_ROC": ["VUS_ROC", "VUS-ROC", "VUSROC", "vus_roc"],
    "Precision": ["Precision", "precision", "P"],
    "Recall":    ["Recall", "recall", "R"],
}

_RECON_COLS = ["RMSE", "MAE", "MaxAE"]

def _find_column(df: pd.DataFrame, aliases: list[str]):
    """Return the first column name in *df* that matches any alias, else None."""
    for alias in aliases:
        if alias in df.columns:
            return alias
    return None


def _get_metric_col(df: pd.DataFrame, metric: str) -> str | None:
    """Return the actual column name in *df* for the logical *metric* name."""
    return _find_column(df, _METRIC_ALIASES.get(metric, [metric]))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def calc_aggregate(args):
    """
    Walk ``results/<compressor>/<detector>/`` and collapse per-CR CSV files
    into a single ``summary.csv`` per (compressor, detector) pair.

    Summary CSV columns: compression_ratio, Time, RMSE, MAE, MaxAE, <metrics…>
    All numeric columns are averaged across datasets.
    """
    for compressor in os.listdir(args.results_dir):
        compressor_path = os.path.join(args.results_dir, compressor)
        if not os.path.isdir(compressor_path) or compressor == "original":
            continue

        for detector in os.listdir(compressor_path):
            cr_dir = os.path.join(compressor_path, detector)
            if not os.path.isdir(cr_dir):
                continue

            print(f"[aggregate] {compressor}/{detector}")
            summary_rows = []

            for fname in sorted(os.listdir(cr_dir)):
                if not fname.endswith(".csv") or fname == "summary.csv":
                    continue
                try:
                    cr_val = float(os.path.splitext(fname)[0])
                except ValueError:
                    continue

                fpath = os.path.join(cr_dir, fname)
                try:
                    df = pd.read_csv(fpath)
                    if df.empty:
                        continue
                    avg = df.drop(columns=["Dataset"], errors="ignore").mean(numeric_only=True)
                    avg["compression_ratio"] = cr_val
                    summary_rows.append(avg)
                except Exception as exc:
                    print(f"[aggregate] Error reading {fpath}: {exc}")

            if summary_rows:
                summary = pd.DataFrame(summary_rows).sort_values("compression_ratio")
                cols = ["compression_ratio"] + [c for c in summary.columns if c != "compression_ratio"]
                summary[cols].to_csv(os.path.join(cr_dir, "summary.csv"), index=False)


def _load_summary(path: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
        df["compression_ratio"] = pd.to_numeric(df["compression_ratio"], errors="coerce")
        df = df.dropna(subset=["compression_ratio"]).sort_values("compression_ratio")
        return df
    except Exception:
        return None


def _load_baseline_metric(results_dir: str, detector: str, metric: str = "F1") -> float | None:
    """Return the mean *metric* for *detector* from the uncompressed baseline."""
    path = os.path.join(results_dir, "original", f"{detector}.csv")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
        col = _get_metric_col(df, metric)
        return float(df[col].mean()) if col else None
    except Exception:
        return None


def _discover_summaries(results_dir: str) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Return {detector: {compressor: summary_df}} for all available summaries.
    Excludes the "original" directory.
    """
    ad_to_methods: dict[str, dict[str, pd.DataFrame]] = defaultdict(dict)
    for compressor in os.listdir(results_dir):
        if compressor == "original":
            continue
        comp_path = os.path.join(results_dir, compressor)
        if not os.path.isdir(comp_path):
            continue
        for detector in os.listdir(comp_path):
            spath = os.path.join(comp_path, detector, "summary.csv")
            if not os.path.exists(spath):
                continue
            df = _load_summary(spath)
            if df is not None:
                ad_to_methods[detector][compressor] = df
    return ad_to_methods


# ---------------------------------------------------------------------------
# Per-method F1-vs-CR plots
# ---------------------------------------------------------------------------

def plot_per_method(args):
    """Individual F1-vs-CR chart for each (compressor, detector) pair."""
    for compressor in os.listdir(args.results_dir):
        comp_path = os.path.join(args.results_dir, compressor)
        if not os.path.isdir(comp_path) or compressor == "original":
            continue

        for detector in os.listdir(comp_path):
            spath = os.path.join(comp_path, detector, "summary.csv")
            if not os.path.exists(spath):
                continue

            df = _load_summary(spath)
            if df is None:
                continue
            col = _get_metric_col(df, "F1")
            if col is None:
                continue

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df["compression_ratio"], df[col], marker="o", linewidth=1.5, markersize=4)
            baseline = _load_baseline_metric(args.results_dir, detector, "F1")
            if baseline is not None:
                ax.axhline(baseline, color="gray", linestyle="--", linewidth=1, label="Uncompressed")
            ax.set_xlabel("Compression Ratio")
            ax.set_ylabel("F1")
            ax.set_title(f"{detector} — {compressor}")
            ax.legend()
            fig.tight_layout()

            out_dir = os.path.join("charts", detector, compressor)
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"{detector}_{compressor}.png"), dpi=150)
            plt.close(fig)
            print(f"[plot] per_method {detector}/{compressor}")


# ---------------------------------------------------------------------------
# Cross-method overlay
# ---------------------------------------------------------------------------

def plot_cross_method(args):
    """All compressors overlaid on one F1-vs-CR chart per detector."""
    summaries = _discover_summaries(args.results_dir)

    for detector, method_map in summaries.items():
        fig, ax = plt.subplots(figsize=(12, 5))
        for compressor, df in sorted(method_map.items()):
            col = _get_metric_col(df, "F1")
            if col is None:
                continue
            ax.plot(df["compression_ratio"], df[col],
                    marker="o", linewidth=1.5, markersize=4, label=compressor)

        baseline = _load_baseline_metric(args.results_dir, detector, "F1")
        if baseline is not None:
            ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2, label="Uncompressed")

        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("F1")
        ax.set_title(f"{detector} — F1 vs CR (all compressors)")
        ax.legend(loc="lower left")
        fig.tight_layout()

        out_dir = os.path.join("charts", detector)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{detector}_cross_method.png"), dpi=150)
        plt.close(fig)
        print(f"[plot] cross_method {detector}")


# ---------------------------------------------------------------------------
# Normalised degradation
# ---------------------------------------------------------------------------

def plot_normalized_degradation(args):
    """F1 / baseline_F1 vs CR for all compressors, per detector."""
    summaries = _discover_summaries(args.results_dir)

    for detector, method_map in summaries.items():
        baseline = _load_baseline_metric(args.results_dir, detector, "F1")
        if not baseline:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        for compressor, df in sorted(method_map.items()):
            col = _get_metric_col(df, "F1")
            if col is None:
                continue
            ax.plot(df["compression_ratio"], df[col] / baseline,
                    marker="o", linewidth=1.5, markersize=4, label=compressor)

        ax.axhline(1.0, color="black", linestyle="--", linewidth=1.2, label="Uncompressed (1.0)")
        ax.set_xlabel("Compression Ratio")
        ax.set_ylabel("Relative F1 (vs uncompressed)")
        ax.set_title(f"{detector} — Normalised F1 Degradation")
        ax.legend(loc="lower left")
        fig.tight_layout()

        out_dir = os.path.join("charts", detector)
        os.makedirs(out_dir, exist_ok=True)
        fig.savefig(os.path.join(out_dir, f"{detector}_normalized_degradation.png"), dpi=150)
        plt.close(fig)
        print(f"[plot] normalized_degradation {detector}")


# ---------------------------------------------------------------------------
# All metrics
# ---------------------------------------------------------------------------

def plot_all_metrics(args):
    """
    For each detector and each metric in _METRIC_ALIASES, plot all compressors
    overlaid on a single chart.  Saves to charts/<detector>/<metric>_cross.png.
    """
    summaries = _discover_summaries(args.results_dir)

    for metric in _METRIC_ALIASES:
        for detector, method_map in summaries.items():
            baseline = _load_baseline_metric(args.results_dir, detector, metric)
            fig, ax = plt.subplots(figsize=(12, 5))
            plotted = False

            for compressor, df in sorted(method_map.items()):
                col = _get_metric_col(df, metric)
                if col is None:
                    continue
                ax.plot(df["compression_ratio"], df[col],
                        marker="o", linewidth=1.5, markersize=4, label=compressor)
                plotted = True

            if not plotted:
                plt.close(fig)
                continue

            if baseline is not None:
                ax.axhline(baseline, color="black", linestyle="--", linewidth=1.2, label="Uncompressed")

            ax.set_xlabel("Compression Ratio")
            ax.set_ylabel(metric)
            ax.set_title(f"{detector} — {metric} vs CR")
            ax.legend(loc="lower left")
            fig.tight_layout()

            out_dir = os.path.join("charts", detector)
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"{detector}_{metric}_cross.png"), dpi=150)
            plt.close(fig)
            print(f"[plot] all_metrics {detector}/{metric}")


# ---------------------------------------------------------------------------
# Compressor × detector affinity heatmap
# ---------------------------------------------------------------------------

def plot_affinity_heatmap(args):
    """
    Heatmap of mean normalised ΔF1 = mean(F1_compressed / F1_baseline − 1)
    across all CRs and datasets, for every (compressor, detector) pair.

    Blue = F1 degradation; Red = improvement (rare).
    Saved to charts/affinity_heatmap.png.
    """
    summaries = _discover_summaries(args.results_dir)

    compressors = sorted({c for det in summaries.values() for c in det})
    detectors   = sorted(summaries.keys())

    matrix = pd.DataFrame(index=detectors, columns=compressors, dtype=float)

    for detector in detectors:
        baseline = _load_baseline_metric(args.results_dir, detector, "F1")
        if not baseline:
            continue
        for compressor in compressors:
            df = summaries[detector].get(compressor)
            if df is None:
                continue
            col = _get_metric_col(df, "F1")
            if col is None:
                continue
            mean_delta = float((df[col] / baseline - 1.0).mean())
            matrix.loc[detector, compressor] = mean_delta

    matrix = matrix.dropna(how="all").dropna(axis=1, how="all").astype(float)
    if matrix.empty:
        print("[plot] affinity_heatmap: no data")
        return

    fig, ax = plt.subplots(figsize=(max(8, len(compressors) * 1.5), max(5, len(detectors) * 0.8)))

    if _HAVE_SEABORN:
        sns.heatmap(
            matrix, annot=True, fmt=".2f", cmap="RdBu", center=0,
            linewidths=0.5, ax=ax, cbar_kws={"label": "Mean ΔF1 (relative to baseline)"},
        )
    else:
        im = ax.imshow(matrix.values, cmap="RdBu", aspect="auto",
                       vmin=-max(abs(matrix.values.min()), abs(matrix.values.max())),
                       vmax= max(abs(matrix.values.min()), abs(matrix.values.max())))
        ax.set_xticks(range(len(compressors)))
        ax.set_xticklabels(compressors, rotation=45, ha="right")
        ax.set_yticks(range(len(detectors)))
        ax.set_yticklabels(detectors)
        for i in range(len(detectors)):
            for j in range(len(compressors)):
                val = matrix.iloc[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Mean ΔF1 (relative to baseline)")

    ax.set_title("Compressor × Detector Sensitivity (Mean Relative ΔF1 across all CRs)")
    ax.set_xlabel("Compressor")
    ax.set_ylabel("Detector")
    fig.tight_layout()

    os.makedirs("charts", exist_ok=True)
    fig.savefig("charts/affinity_heatmap.png", dpi=150)
    plt.close(fig)
    print("[plot] affinity_heatmap → charts/affinity_heatmap.png")


# ---------------------------------------------------------------------------
# Reconstruction error vs ΔF1 correlation
# ---------------------------------------------------------------------------

def plot_reconstruction_correlation(args):
    """
    Scatter plot of RMSE vs ΔF1 across all (compressor, CR, dataset) triples,
    with Spearman correlation coefficient.  One plot per detector.

    Requires that reconstruction error columns (RMSE, MAE, MaxAE) were saved
    during the experiment (available in per-CR CSVs, not just summary.csv).
    """
    for compressor in os.listdir(args.results_dir):
        comp_path = os.path.join(args.results_dir, compressor)
        if not os.path.isdir(comp_path) or compressor == "original":
            continue

        for detector in os.listdir(comp_path):
            cr_dir = os.path.join(comp_path, detector)
            if not os.path.isdir(cr_dir):
                continue

            baseline_path = os.path.join(args.results_dir, "original", f"{detector}.csv")
            if not os.path.exists(baseline_path):
                continue

            try:
                base_df = pd.read_csv(baseline_path)
                f1_col_base = _get_metric_col(base_df, "F1")
                if f1_col_base is None:
                    continue
                baseline_by_dataset = base_df.set_index("Dataset")[f1_col_base].to_dict()
            except Exception:
                continue

            rmse_vals, delta_f1_vals = [], []

            for fname in sorted(os.listdir(cr_dir)):
                if not fname.endswith(".csv") or fname == "summary.csv":
                    continue
                try:
                    df = pd.read_csv(os.path.join(cr_dir, fname))
                    if "RMSE" not in df.columns:
                        continue
                    f1_col = _get_metric_col(df, "F1")
                    if f1_col is None:
                        continue
                    for _, row in df.iterrows():
                        base_f1 = baseline_by_dataset.get(row["Dataset"])
                        if base_f1 and base_f1 > 0:
                            rmse_vals.append(row["RMSE"])
                            delta_f1_vals.append(row[f1_col] / base_f1 - 1.0)
                except Exception:
                    continue

            if len(rmse_vals) < 5:
                continue

            r, pval = stats.spearmanr(rmse_vals, delta_f1_vals)
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(rmse_vals, delta_f1_vals, alpha=0.4, s=20, color="#2196F3")
            # Regression line
            z = np.polyfit(rmse_vals, delta_f1_vals, 1)
            x_line = np.linspace(min(rmse_vals), max(rmse_vals), 200)
            ax.plot(x_line, np.polyval(z, x_line), color="#F44336", linewidth=1.5)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
            ax.set_xlabel("Reconstruction RMSE")
            ax.set_ylabel("Relative ΔF1 (vs uncompressed)")
            ax.set_title(
                f"{detector} — {compressor}\n"
                f"Spearman r = {r:.3f}  (p = {pval:.3g}, n = {len(rmse_vals)})"
            )
            fig.tight_layout()

            out_dir = os.path.join("charts", detector, compressor)
            os.makedirs(out_dir, exist_ok=True)
            fig.savefig(os.path.join(out_dir, f"{detector}_{compressor}_recon_corr.png"), dpi=150)
            plt.close(fig)

    print("[plot] reconstruction_correlation done")


# ---------------------------------------------------------------------------
# Safe compression budget
# ---------------------------------------------------------------------------

def plot_safe_cr_budget(args, tolerance: float = 0.05):
    """
    For each (compressor, detector), find the maximum CR where mean F1 stays
    within *tolerance* (default 5 %) of the uncompressed baseline.

    Outputs:
      - charts/safe_cr_budget.csv   — table of safe CRs
      - charts/safe_cr_budget.png   — grouped bar chart
      - charts/safe_cr_budget.tex   — LaTeX table
    """
    summaries = _discover_summaries(args.results_dir)

    compressors = sorted({c for det in summaries.values() for c in det})
    detectors   = sorted(summaries.keys())

    budget = pd.DataFrame(index=detectors, columns=compressors, dtype=float)

    for detector in detectors:
        baseline = _load_baseline_metric(args.results_dir, detector, "F1")
        if not baseline:
            continue
        threshold = baseline * (1.0 - tolerance)

        for compressor in compressors:
            df = summaries[detector].get(compressor)
            if df is None:
                continue
            col = _get_metric_col(df, "F1")
            if col is None:
                continue
            safe = df[df[col] >= threshold]["compression_ratio"]
            budget.loc[detector, compressor] = float(safe.max()) if not safe.empty else 1.0

    budget = budget.dropna(how="all").dropna(axis=1, how="all").astype(float)
    if budget.empty:
        print("[plot] safe_cr_budget: no data")
        return

    os.makedirs("charts", exist_ok=True)

    # CSV
    budget.to_csv("charts/safe_cr_budget.csv")

    # LaTeX
    with open("charts/safe_cr_budget.tex", "w") as f:
        f.write(budget.to_latex(float_format="%.0f", na_rep="—"))

    # Bar chart
    n_det  = len(budget.index)
    n_comp = len(budget.columns)
    fig, ax = plt.subplots(figsize=(max(10, n_comp * 2), 5))
    x = np.arange(n_det)
    width = 0.8 / n_comp

    for i, compressor in enumerate(budget.columns):
        ax.bar(x + i * width, budget[compressor].fillna(0),
               width=width, label=compressor, alpha=0.85)

    ax.set_xlabel("Detector")
    ax.set_ylabel(f"Max Safe CR (F1 within {int(tolerance*100)}% of baseline)")
    ax.set_title(f"Safe Compression Budget (tolerance = {int(tolerance*100)}%)")
    ax.set_xticks(x + width * (n_comp - 1) / 2)
    ax.set_xticklabels(budget.index, rotation=20, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig("charts/safe_cr_budget.png", dpi=150)
    plt.close(fig)
    print("[plot] safe_cr_budget → charts/safe_cr_budget.{csv,png,tex}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_analysis(args):
    """Run the complete analysis and plotting pipeline."""

    print("[analysis] Aggregating results …")
    calc_aggregate(args)

    print("[analysis] Generating plots …")
    plot_per_method(args)
    plot_cross_method(args)
    plot_normalized_degradation(args)
    plot_all_metrics(args)
    plot_affinity_heatmap(args)
    plot_reconstruction_correlation(args)
    plot_safe_cr_budget(args)

    print("[analysis] Done.")
