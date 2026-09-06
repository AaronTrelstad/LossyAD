"""
main.py — entry point for the LossyAD benchmark.

Usage
-----
# Run everything (bounds + experiment + analysis) on univariate data:
    python main.py

# Multivariate:
    python main.py --multivariate

# Only build CR→bound maps:
    python main.py bounds

# Only run experiments (assumes bound maps exist):
    python main.py experiment

# Only run experiments for specific detectors / compressors / CRs:
    python main.py experiment --detectors IForest LSTMAD --compressors DWT ZFP --crs 1 5 10

# Only run analysis / plotting:
    python main.py analyze

# Parallel detectors (e.g. 4 GPUs on a node):
    python main.py experiment --workers 4
    # or: LOSSYAD_WORKERS=4 python main.py experiment
"""

import argparse
import sys

from src.configs import ExperimentConfig, AnalysisConfig, MethodType, set_seed, AVAILABLE_AD_METHODS
from src.experiment import run_experiment, create_bound_map, uncompressed_experiment, compressed_experiment, _preload_datasets
from src.analysis import run_analysis

import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser(
        description="LossyAD: effect of lossy compression on time-series anomaly detection."
    )

    parser.add_argument(
        "mode",
        nargs="?",
        default="all",
        choices=["all", "bounds", "experiment", "analyze"],
        help=(
            "Pipeline stage to run.  "
            "'all' (default) runs bounds → experiment → analyze in sequence."
        ),
    )

    parser.add_argument(
        "--multivariate", "-m",
        action="store_true",
        help="Use TSB-AD-M (multivariate) dataset instead of TSB-AD-U (univariate).",
    )

    parser.add_argument(
        "--detectors", "-d",
        nargs="+",
        metavar="DET",
        default=None,
        help=(
            f"AD methods to run (subset of available: {AVAILABLE_AD_METHODS}). "
            "Default: all available."
        ),
    )

    parser.add_argument(
        "--compressors", "-c",
        nargs="+",
        metavar="COMP",
        default=None,
        help=(
            f"Compressors to run (subset of: {[m.name for m in MethodType]}). "
            "Default: all available."
        ),
    )

    parser.add_argument(
        "--crs",
        nargs="+",
        type=int,
        metavar="CR",
        default=None,
        help="Compression ratios to evaluate. Default: 1 3 5 7 10 15 20 30 40 50.",
    )

    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel detector workers. Default: LOSSYAD_WORKERS env var or 1.",
    )

    parser.add_argument(
        "--chart",
        action="store_true",
        help="Save original-vs-decompressed waveform plots (slow).",
    )

    parser.add_argument(
        "--dataset-list",
        metavar="CSV",
        default=None,
        help="Override the dataset file list CSV (e.g. Datasets/File_List/TSB-AD-U-FM.csv).",
    )

    parser.add_argument(
        "--force-bounds",
        action="store_true",
        help="Rebuild CR→bound maps even if they already exist.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2024,
        help="Random seed (default: 2024).",
    )

    return parser.parse_args()


def _build_config(args) -> ExperimentConfig:
    cfg = ExperimentConfig(multivariate=args.multivariate)
    cfg.seed  = args.seed
    cfg.chart = args.chart

    if args.detectors is not None:
        unknown = set(args.detectors) - set(AVAILABLE_AD_METHODS)
        if unknown:
            print(f"[main] Warning: unknown/unavailable detectors ignored: {sorted(unknown)}")
        cfg.ad_methods = [d for d in args.detectors if d in AVAILABLE_AD_METHODS]

    if args.compressors is not None:
        name_map = {m.name: m for m in MethodType}
        unknown  = set(args.compressors) - set(name_map)
        if unknown:
            print(f"[main] Warning: unknown/unavailable compressors ignored: {sorted(unknown)}")
        cfg.compressors = [name_map[c] for c in args.compressors if c in name_map]

    if args.crs is not None:
        cfg.compression_ratios = sorted(set(args.crs))

    if args.workers is not None:
        cfg.n_workers = args.workers

    if args.dataset_list is not None:
        cfg.dataset_list = args.dataset_list

    return cfg


def main():
    args = _parse_args()
    cfg  = _build_config(args)

    set_seed(cfg.seed)

    print(f"[main] Mode           : {args.mode}")
    print(f"[main] Dataset        : {'multivariate (TSB-AD-M)' if cfg.multivariate else 'univariate (TSB-AD-U)'}")
    print(f"[main] Dataset list   : {cfg.dataset_list}")
    print(f"[main] Detectors      : {cfg.ad_methods}")
    print(f"[main] Compressors    : {[m.name for m in cfg.compressors]}")
    print(f"[main] CRs            : {cfg.compression_ratios}")
    print(f"[main] Workers        : {cfg.n_workers}")

    if args.mode in ("all", "bounds"):
        print("\n[main] === Building CR → bound maps ===")
        force_bounds = getattr(args, "force_bounds", False)
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(6, len(cfg.compressors))) as pool:
            futures = {
                pool.submit(create_bound_map, c, cfg, force_bounds): c
                for c in cfg.compressors
            }
            for f in as_completed(futures):
                c = futures[f]
                try:
                    f.result()
                except Exception as exc:
                    print(f"[main] Bound map failed for {c.name}: {exc}")

    if args.mode in ("all", "experiment"):
        print("\n[main] === Running experiments ===")
        run_experiment(cfg)

    if args.mode in ("all", "analyze"):
        print("\n[main] === Running analysis ===")
        acfg = AnalysisConfig(multivariate=args.multivariate)
        acfg.ad_methods = cfg.ad_methods
        run_analysis(acfg)


if __name__ == "__main__":
    main()
