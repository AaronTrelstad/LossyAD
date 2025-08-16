import os
import time
import pandas as pd
import numpy as np
from tersets import compress, decompress, Method
import json
import zstandard as zstd
import io 

from concurrent.futures import ThreadPoolExecutor
from scipy.interpolate import interp1d

from TSB_AD.evaluation.metrics import get_metrics
from TSB_AD.utils.slidingWindows import find_length_rank
from TSB_AD.model_wrapper import (
    run_Semisupervise_AD,
    run_Unsupervise_AD,
    Semisupervise_AD_Pool,
    Unsupervise_AD_Pool
)
from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict

from .helpers import normalize_data, gen_chart
from .configs import MethodType

'''
Creates a mapping between compression ratio and error bounds, for every compression
method and every dataset
'''
def create_bound_map(method, args):
    try:
        file_list = pd.read_csv(args.dataset_list)['file_name'].values
    except Exception as e:
        print(f"Failed to read file list: {e}")
        return

    os.makedirs(args.cr_map_dir, exist_ok=True)
    output_path = os.path.join(args.cr_map_dir, f"{method.name}.json")

    method_data_list = []

    for filename in file_list:
        file_path = os.path.join(args.dataset_dir, filename)
        try:
            df = pd.read_csv(file_path).dropna()
            data = df.iloc[:, :-1].values.astype(float)

        except Exception as e:
            print(f"Failed to load or normalize {filename}: {e}")
            continue

        crs = []
        for bound in args.error_bounds:
            try:
                def zstd_compress(data_bytes, level = 3):
                    cctx = zstd.ZstdCompressor(level=level)
                    compressed = cctx.compress(data_bytes)
                    return len(compressed)

                norm_data = normalize_data(data) 
                norm_data = norm_data.astype(np.float64)

                compressed_values = compress(norm_data, method.value, bound)
                compressed_values = np.array(compressed_values, dtype=np.uint8)
                compressed_bytes = compressed_values.tobytes()

                compressed_size = zstd_compress(compressed_bytes)

                cr = norm_data.nbytes / compressed_size
                crs.append(cr)

            except Exception as e:
                print(f"Compression failed for {filename}, method={method.name}, bound={bound}: {e}")
 
        bounds_with_zero = list(args.error_bounds)
        crs_array, bounds_array = zip(*sorted(zip(crs, bounds_with_zero)))

        try:
            interp_func = interp1d(crs_array, bounds_array, kind='linear', bounds_error=False, fill_value="extrapolate")
        except Exception as e:
            print(f"Interpolation failed for {filename}, method={method.name}: {e}")
            continue

        cr_bound_mapping = {}
        for cr_target in args.compression_ratios:
            try:
                bound_val = float(interp_func(cr_target))
                cr_bound_mapping[str(cr_target)] = round(bound_val, 6)
            except Exception:
                continue

        dataset_name = os.path.splitext(filename)[0]
        method_data_list.append({
            "dataset": dataset_name,
            "map": cr_bound_mapping
        })

        print(f"Processed dataset {dataset_name} for method {method.name}")

    with open(output_path, "w") as f:
        json.dump({"datasets": method_data_list}, f, indent=4)

    print(f"Saved combined CR-bound map for method {method.name} at {output_path}")
    return {item["dataset"]: item["map"] for item in method_data_list}


def compressed_experiment(detector, args, file_list):
    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[detector]

    for method in MethodType:
        bound_map_path = os.path.join(args.cr_map_dir, f"{method.name}.json")
        if not os.path.exists(bound_map_path):
            bound_map = create_bound_map(method, args)  
        else:
            with open(bound_map_path, "r") as f:
                raw = json.load(f)
                bound_map = {
                    item["dataset"]: item["map"]
                    for item in raw["datasets"]
                }

        for cr in args.compression_ratios:
            results_rows = []
            columns = None

            for dataset in file_list:
                print(f'[{detector}] Processing {dataset}')
                file_path = os.path.join(args.dataset_dir, dataset)

                try:
                    df = pd.read_csv(file_path).dropna()
                except Exception as e:
                    print(e)
                    continue

                data = df.iloc[:, :-1].values.astype(float)
                norm_data = normalize_data(data) 
                error_bound = bound_map[dataset.split(".")[0]][str(cr)]

                compressed_values = compress(norm_data, method.value, error_bound)
                decompressed_data = decompress(compressed_values)

                decompressed_data = np.array(decompressed_data)
                if decompressed_data.ndim == 1:
                    decompressed_data = decompressed_data.reshape(-1, 1)

                labels = df['Label'].astype(int).to_numpy()

                if args.chart:
                    gen_chart(norm_data, decompressed_data, labels, dataset, method, detector, cr)

                slidingWindow = find_length_rank(decompressed_data[:, 0].reshape(-1, 1), rank=1)
                train_index = int(dataset.split('.')[0].split('_')[-3])
                data_train = decompressed_data[:train_index, :]

                try:
                    start = time.time()
                    if detector in Semisupervise_AD_Pool:
                        output = run_Semisupervise_AD(detector, data_train, decompressed_data, **Optimal_Det_HP)
                    elif detector in Unsupervise_AD_Pool:
                        output = run_Unsupervise_AD(detector, decompressed_data, **Optimal_Det_HP)
                    else:
                        raise ValueError(f"Unknown detector: {detector}")
                    duration = time.time() - start
                except Exception as e:
                    print(e)
                    continue

                try:
                    eval_result = get_metrics(output, labels, slidingWindow=slidingWindow)
                    row = [dataset, duration] + list(eval_result.values())
                    if columns is None:
                        columns = ['Dataset', 'Time'] + list(eval_result.keys())
                except Exception as e:
                    print(e)
                    row = [dataset, duration] + [0]*9  

                results_rows.append(row)

            if results_rows:
                out_dir = os.path.join(args.results_dir, method.name, detector)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{cr}.csv")

                df_all = pd.DataFrame(results_rows, columns=columns)
                df_all.to_csv(out_path, index=False)

def uncompressed_experiment(args, file_list):
    for detector in args.ad_methods:
        Optimal_Det_HP = Optimal_Uni_algo_HP_dict[detector]

        results_rows = []
        columns = None

        for dataset in file_list:
            print(f'[{detector}] Processing {dataset}')
            file_path = os.path.join(args.dataset_dir, dataset)

            try:
                df = pd.read_csv(file_path).dropna()
            except Exception as e:
                print(e)
                continue

            data = df.iloc[:, :-1].values.astype(float)
            norm_data = normalize_data(data) 

            data = np.array(data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
            labels = df['Label'].astype(int).to_numpy()
            slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
            train_index = int(dataset.split('.')[0].split('_')[-3])
            data_train = data[:train_index, :]

            try:
                start = time.time()
                if detector in Semisupervise_AD_Pool:
                    output = run_Semisupervise_AD(detector, data_train, data, **Optimal_Det_HP)
                elif detector in Unsupervise_AD_Pool:
                    output = run_Unsupervise_AD(detector, data, **Optimal_Det_HP)
                else:
                    raise ValueError(f"Unknown detector: {detector}")
                duration = time.time() - start
            except Exception as e:
                print(e)
                continue

            try:
                eval_result = get_metrics(output, labels, slidingWindow=slidingWindow)
                row = [dataset, duration] + list(eval_result.values())
                if columns is None:
                    columns = ['Dataset', 'Time'] + list(eval_result.keys())
            except Exception as e:
                print(e)
                row = [dataset, duration] + [0]*9  

            results_rows.append(row)

        if results_rows:
            out_dir = os.path.join(args.results_dir, "original")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, f"{detector}.csv")

            df_all = pd.DataFrame(results_rows, columns=columns)
            df_all.to_csv(out_path, index=False)

def run_experiment(args):
    try:
        file_list = pd.read_csv(args.dataset_list)['file_name'].values
    except Exception as e:
        print(e)

    uncompressed_experiment(args, file_list)

    def bound_worker(compressor):
        create_bound_map(compressor, args)

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(
            bound_worker,
            compressor
        )
        for compressor in MethodType]

        for future in futures:
            future.result()

    def experiment_worker(method):
        compressed_experiment(method, args, file_list)

    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = [executor.submit(
            experiment_worker,
            detector
        )
        for detector in args.ad_methods]

        for future in futures:
            future.result()
