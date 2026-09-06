"""
experiment.py — core pipeline for the LossyAD benchmark.

Pipeline:
  1. create_bound_map  : for each compressor, build a per-dataset CR → error-bound map.
  2. uncompressed_experiment : run all detectors on raw (normalised) data as baseline.
  3. compressed_experiment   : compress → decompress → run detectors at each target CR.
  4. run_experiment     : orchestrate the above for all compressors and detectors.
"""

import os
import time
import json
import tempfile
import threading

import numpy as np
import pandas as pd
import zstandard as zstd

from concurrent.futures import ThreadPoolExecutor, as_completed
from scipy.interpolate import interp1d

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import (
    run_Semisupervise_AD,
    run_Unsupervise_AD,
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool,
)
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict, Optimal_Multi_algo_HP_dict

from .helpers import normalize_data, gen_chart
from .configs import MethodType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Thread-local ZstdCompressor — ZstdCompressor is NOT thread-safe, so each
# thread gets its own instance.  create_bound_map runs in a ThreadPoolExecutor
# with up to 6 workers, so a shared singleton would corrupt results.
_ZSTD_LOCAL = threading.local()


def _zstd_size(data_bytes: bytes) -> int:
    """Return the zstd-compressed byte count of *data_bytes* (thread-safe)."""
    if not hasattr(_ZSTD_LOCAL, "cctx"):
        _ZSTD_LOCAL.cctx = zstd.ZstdCompressor(level=3)
    return len(_ZSTD_LOCAL.cctx.compress(data_bytes))


def _reconstruction_errors(original: np.ndarray, reconstructed: np.ndarray) -> dict:
    """Compute point-wise reconstruction quality metrics."""
    diff = original.ravel() - reconstructed.ravel()
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    mae  = float(np.mean(np.abs(diff)))
    maxae = float(np.max(np.abs(diff)))
    return {"RMSE": rmse, "MAE": mae, "MaxAE": maxae}


def _train_index_from_filename(filename: str) -> int:
    """
    Extract the training split index encoded in TSB-AD filenames.
    Filename convention: <name>_<train_end>_<...>.csv
    """
    stem = os.path.splitext(os.path.basename(filename))[0]
    return int(stem.split("_")[-3])


# ---------------------------------------------------------------------------
# Step 1 — Build CR → error-bound maps
# ---------------------------------------------------------------------------

def create_bound_map(method: MethodType, args, force: bool = False) -> dict:
    """
    For *method*, scan every dataset in *args.dataset_list* and fit a linear
    interpolant from achieved CR → required error bound.  The result is
    written to ``args.cr_map_dir/<method.name>.json`` and returned as a dict
    {dataset_name: {str(cr): bound}}.

    If the output JSON already exists and *force* is False, it is loaded and
    returned immediately without recomputation.
    """
    os.makedirs(args.cr_map_dir, exist_ok=True)
    output_path = os.path.join(args.cr_map_dir, f"{method.name}.json")

    if not force and os.path.exists(output_path):
        print(f"[bound_map] {method.name} — already exists, skipping (use --force-bounds to rebuild)")
        with open(output_path) as f:
            raw = json.load(f)
        return {item["dataset"]: item["map"] for item in raw["datasets"]}

    try:
        file_list = pd.read_csv(args.dataset_list)["file_name"].values
    except Exception as exc:
        print(f"[bound_map] Cannot read file list: {exc}")
        return {}

    method_data_list = []

    for filename in file_list:
        file_path = os.path.join(args.dataset_dir, filename)
        try:
            df = pd.read_csv(file_path).dropna()
            data = df.iloc[:, :-1].values.astype(float)
        except Exception as exc:
            print(f"[bound_map] Failed to load {filename}: {exc}")
            continue

        norm_data = normalize_data(data).astype(np.float64)

        crs    = [1.0]
        bounds = [0.0]

        for bound in args.error_bounds:
            try:
                compressor = method.value(error_bound=bound)
                compressed = compressor.compress(norm_data)

                if isinstance(compressed, (bytes, bytearray)):
                    compressed_bytes = compressed
                else:
                    compressed_bytes = np.array(compressed, dtype=np.float64).tobytes()

                compressed_size = _zstd_size(compressed_bytes)
                if compressed_size == 0:
                    continue

                cr = norm_data.nbytes / compressed_size
                crs.append(cr)
                bounds.append(float(bound))

            except Exception as exc:
                print(
                    f"[bound_map] Compression failed — {filename}, "
                    f"method={method.name}, bound={bound:.4f}: {exc}"
                )

        # Sort by CR and build interpolant
        paired = sorted(zip(crs, bounds))
        crs_sorted, bounds_sorted = zip(*paired)

        try:
            interp_func = interp1d(
                crs_sorted,
                bounds_sorted,
                kind="linear",
                bounds_error=False,
                fill_value="extrapolate",
            )
        except Exception as exc:
            print(f"[bound_map] Interpolation failed — {filename}: {exc}")
            continue

        cr_bound_mapping = {}
        for cr_target in args.compression_ratios:
            bound_val = float(interp_func(cr_target))
            # Clamp to [0, 1]: extrapolation can go negative or +inf (e.g. NONE
            # where all achieved CRs are ≈1 so the slope is undefined).
            if not np.isfinite(bound_val) or bound_val > 1.0:
                bound_val = 1.0
            elif bound_val < 0.0:
                bound_val = 0.0
            cr_bound_mapping[str(cr_target)] = round(bound_val, 6)

        dataset_name = os.path.splitext(filename)[0]
        method_data_list.append({"dataset": dataset_name, "map": cr_bound_mapping})
        print(f"[bound_map] {method.name} — processed {dataset_name}")

    # Write atomically: temp file then rename so a crash never leaves a partial JSON
    tmp_fd, tmp_path = tempfile.mkstemp(dir=args.cr_map_dir, suffix=".json.tmp")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            json.dump({"datasets": method_data_list}, f, indent=4)
        os.replace(tmp_path, output_path)
    except Exception:
        os.unlink(tmp_path)
        raise

    print(f"[bound_map] Saved {output_path}")
    return {item["dataset"]: item["map"] for item in method_data_list}


def _load_bound_map(method: MethodType, args) -> dict:
    """Load or create the CR → bound map for *method*."""
    bound_map_path = os.path.join(args.cr_map_dir, f"{method.name}.json")
    if not os.path.exists(bound_map_path):
        return create_bound_map(method, args)
    with open(bound_map_path) as f:
        raw = json.load(f)
    return {item["dataset"]: item["map"] for item in raw["datasets"]}


# ---------------------------------------------------------------------------
# Step 2 — Uncompressed baseline
# ---------------------------------------------------------------------------

def _hp_dict(args):
    """Return the appropriate HP dict based on univariate vs multivariate mode."""
    if getattr(args, "multivariate", False):
        return Optimal_Multi_algo_HP_dict
    return Optimal_Uni_algo_HP_dict


def _run_baseline_detector(detector: str, args, file_list, hp_dict):
    """Run one detector on the uncompressed data. Called from uncompressed_experiment."""
    out_root = os.path.join(args.results_dir, "original")
    out_path = os.path.join(out_root, f"{detector}.csv")
    if os.path.exists(out_path):
        print(f"[baseline] Skipping {detector} (exists)")
        return

    Optimal_Det_HP = hp_dict[detector]
    columns_written = False
    rows_written    = 0
    tmp_path        = out_path + ".part"
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    for filename in file_list:
        file_path = os.path.join(args.dataset_dir, filename)
        try:
            df = pd.read_csv(file_path).dropna()
        except Exception as exc:
            print(f"[baseline][{detector}] Load failed {filename}: {exc}")
            continue

        data = normalize_data(df.iloc[:, :-1].values.astype(float))
        data = np.asarray(data)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        labels        = df["Label"].astype(int).to_numpy()
        sliding_window = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index   = _train_index_from_filename(filename)
        data_train    = data[:train_index, :]

        try:
            t0 = time.time()
            if detector in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(detector, data_train, data, **Optimal_Det_HP)
            else:
                output = run_Unsupervise_AD(detector, data, **Optimal_Det_HP)
            duration = time.time() - t0
        except Exception as exc:
            print(f"[baseline][{detector}] Detector failed {filename}: {exc}")
            continue

        try:
            eval_result = get_metrics(output, labels, slidingWindow=sliding_window)
            row_data    = [filename, duration] + list(eval_result.values())
            if not columns_written:
                columns = ["Dataset", "Time"] + list(eval_result.keys())
        except Exception as exc:
            print(f"[baseline][{detector}] Metrics failed {filename}: {exc}")
            continue

        row_df = pd.DataFrame([row_data], columns=columns)
        row_df.to_csv(tmp_path, mode="a", header=not columns_written, index=False)
        columns_written = True
        rows_written   += 1

    if rows_written > 0:
        os.replace(tmp_path, out_path)
        print(f"[baseline] Saved {out_path} ({rows_written} datasets)")
    elif os.path.exists(tmp_path):
        os.remove(tmp_path)


def uncompressed_experiment(args, file_list):
    """
    Run every detector in *args.ad_methods* on the raw normalised data and
    save results to ``results/original/<detector>.csv``.
    Detectors run in parallel using args.n_workers threads.
    """
    out_root = os.path.join(args.results_dir, "original")
    os.makedirs(out_root, exist_ok=True)

    hp_dict = _hp_dict(args)

    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(_run_baseline_detector, detector, args, file_list, hp_dict): detector
            for detector in args.ad_methods
        }
        for future in as_completed(futures):
            detector = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[baseline] Failed for {detector}: {exc}")
        return


# ---------------------------------------------------------------------------
# Step 3 — Compressed experiment
# ---------------------------------------------------------------------------

def _preload_datasets(file_list, dataset_dir):
    """
    Load all dataset CSVs into memory once.  Returns a dict:
      filename → {"norm_data": np.ndarray, "labels": np.ndarray,
                   "train_index": int, "sliding_window": int}
    Files that fail to load are omitted with a warning.
    """
    cache = {}
    for filename in file_list:
        file_path = os.path.join(dataset_dir, filename)
        try:
            df = pd.read_csv(file_path).dropna()
        except Exception as exc:
            print(f"[load] Failed {filename}: {exc}")
            continue
        norm_data = normalize_data(df.iloc[:, :-1].values.astype(float))
        norm_data = np.asarray(norm_data)
        if norm_data.ndim == 1:
            norm_data = norm_data.reshape(-1, 1)
        try:
            train_index   = _train_index_from_filename(filename)
            sliding_window = find_length_rank(norm_data[:, 0].reshape(-1, 1), rank=1)
        except Exception as exc:
            print(f"[load] Metadata extraction failed {filename}: {exc}")
            continue
        cache[filename] = {
            "norm_data":     norm_data,
            "labels":        df["Label"].astype(int).to_numpy(),
            "train_index":   train_index,
            "sliding_window": sliding_window,
        }
    return cache


def compressed_experiment(detector: str, args, dataset_cache: dict):
    """
    For every compressor in *args.compressors* and every target CR in
    *args.compression_ratios*, compress → decompress → run *detector* and save
    results to ``results/<compressor>/<detector>/<cr>.csv``.

    Each result CSV has columns:
      Dataset, Time, RMSE, MAE, MaxAE, <AD metric columns…>

    The NONE compressor is only run at CR=1 (passthrough does not change with CR).
    *dataset_cache* is pre-loaded by the caller (run_experiment) and shared
    read-only across all parallel worker threads to avoid redundant I/O.
    Results are written incrementally (one row per dataset) so a mid-run crash
    loses at most the current dataset's row, not the entire CR's output.
    """
    Optimal_Det_HP = _hp_dict(args)[detector]

    for method in args.compressors:
        bound_map = _load_bound_map(method, args)

        # For the passthrough baseline, only CR=1 is meaningful
        cr_list = (
            [cr for cr in args.compression_ratios if cr == 1]
            if method == MethodType.NONE
            else args.compression_ratios
        )

        for cr in cr_list:
            out_dir  = os.path.join(args.results_dir, method.name, detector)
            out_path = os.path.join(out_dir, f"{cr}.csv")
            if os.path.exists(out_path):
                print(f"[{detector}] Skipping {method.name} CR={cr} (exists)")
                continue

            os.makedirs(out_dir, exist_ok=True)

            columns_written = False
            rows_written    = 0

            # Open in append mode so each row is flushed to disk immediately.
            # If the process dies mid-CR, the partial file still holds all rows
            # written so far.  On the next run the skip check above prevents
            # re-running the whole CR, but partial files are re-run from scratch
            # (the file is removed at the start of a fresh attempt).
            tmp_path = out_path + ".part"
            # If a previous partial file exists, remove it and start fresh for this CR.
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

            for filename, ds in dataset_cache.items():
                dataset_key = os.path.splitext(filename)[0]
                if dataset_key not in bound_map:
                    print(f"[{detector}] No bound map entry for {dataset_key}, skipping")
                    continue

                norm_data    = ds["norm_data"]
                labels       = ds["labels"]
                train_index  = ds["train_index"]
                sliding_window = ds["sliding_window"]
                error_bound  = bound_map[dataset_key][str(cr)]

                try:
                    compressor   = method.value(error_bound=error_bound)
                    compressed   = compressor.compress(norm_data)
                    decompressed = compressor.decompress(compressed, norm_data.shape, norm_data.dtype)
                except Exception as exc:
                    print(f"[{detector}] Compress/decompress failed {filename} CR={cr}: {exc}")
                    continue

                decompressed = np.asarray(decompressed)
                if decompressed.ndim == 1:
                    decompressed = decompressed.reshape(-1, 1)

                recon_errors = _reconstruction_errors(norm_data, decompressed)

                if args.chart:
                    gen_chart(norm_data, decompressed, labels, filename, method, detector, cr)

                data_train = decompressed[:train_index, :]

                try:
                    t0 = time.time()
                    if detector in Semisupervise_AD_Pool:
                        output = run_Semisupervise_AD(detector, data_train, decompressed, **Optimal_Det_HP)
                    elif detector in Unsupervise_AD_Pool:
                        output = run_Unsupervise_AD(detector, decompressed, **Optimal_Det_HP)
                    else:
                        raise ValueError(f"Unknown detector pool for: {detector}")
                    duration = time.time() - t0
                except Exception as exc:
                    print(f"[{detector}] Detector failed {filename} CR={cr}: {exc}")
                    continue

                try:
                    eval_result = get_metrics(output, labels, slidingWindow=sliding_window)
                    recon_vals  = list(recon_errors.values())
                    row_data    = [filename, duration] + recon_vals + list(eval_result.values())
                    if not columns_written:
                        columns = (
                            ["Dataset", "Time"]
                            + list(recon_errors.keys())
                            + list(eval_result.keys())
                        )
                except Exception as exc:
                    print(f"[{detector}] Metrics failed {filename} CR={cr}: {exc}")
                    continue

                # Append row to partial file immediately
                row_df = pd.DataFrame([row_data], columns=columns)
                row_df.to_csv(tmp_path, mode="a", header=not columns_written, index=False)
                columns_written = True
                rows_written   += 1

            if rows_written > 0:
                os.replace(tmp_path, out_path)
                print(f"[{detector}] Saved {out_path} ({rows_written} datasets)")
            elif os.path.exists(tmp_path):
                os.remove(tmp_path)


# ---------------------------------------------------------------------------
# Step 4 — Orchestrator
# ---------------------------------------------------------------------------

def run_experiment(args):
    """
    Full experiment pipeline:
      1. Pre-load all dataset CSVs into memory once (shared across all workers).
      2. Run uncompressed baseline for all detectors (n_workers in parallel).
      3. Run compressed experiments for all detectors (n_workers in parallel).

    Bound maps must already exist (built by create_bound_map / main.py bounds stage).
    If a map is missing for a compressor, _load_bound_map will build it on demand.
    """
    try:
        file_list = pd.read_csv(args.dataset_list)["file_name"].values
    except Exception as exc:
        print(f"[run_experiment] Cannot read file list: {exc}")
        return

    # Pre-load datasets once — shared read-only across all worker threads.
    # This avoids each of the n_workers threads independently reading all CSVs.
    print(f"[run_experiment] Pre-loading {len(file_list)} datasets …")
    dataset_cache = _preload_datasets(file_list, args.dataset_dir)
    print(f"[run_experiment] Loaded {len(dataset_cache)} datasets.")

    # --- Baseline ---
    uncompressed_experiment(args, file_list)

    # --- Compressed experiments (one detector per worker) ---
    print(f"[run_experiment] Running compressed experiments (n_workers={args.n_workers}) …")
    with ThreadPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {
            pool.submit(compressed_experiment, detector, args, dataset_cache): detector
            for detector in args.ad_methods
        }
        for future in as_completed(futures):
            detector = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(f"[run_experiment] Experiment failed for {detector}: {exc}")

    print("[run_experiment] Done.")
