import os
import sys
import time
import random
import argparse
import logging
import pandas as pd
import numpy as np
import torch
from tersets import compress, decompress, Method
import json

from enum import Enum
from sklearn.preprocessing import MinMaxScaler
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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

seed = 2024
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print("CUDA Available: ", torch.cuda.is_available())
print("cuDNN Version: ", torch.backends.cudnn.version())

'''
TODO: 
1. Add the rest of the TerseTS compression methods

For time complexity I think we should limit this to only the essential compression methods
PMC, SWING, SLIDE, VW?, Piecewise?
'''
class Methods(Enum):
    PMC_M = Method.PoorMansCompressionMean
    PMC_MR = Method.PoorMansCompressionMidrange
    SWING = Method.SwingFilter
    #SLIDE = Method.SlideFilter

class Args:
    def __init__(self):
        self.results_dir = 'results/'
        self.cr_map_dir = 'cr_bound_maps/'
        self.dataset_dir = 'Datasets/TSB-AD-U'
        self.file_list = 'Datasets/File_List/TSB-AD-U-Test.csv'
        self.compression_ratios = [0, 3, 5, 7, 10, 15, 20, 30, 40, 50]
        self.error_bounds = np.linspace(0, 0.8, 100)
        self.ad_methods = list(Optimal_Uni_algo_HP_dict.keys())

'''
Normalize all of the values in the dataset between [0, 1]
'''
def normalize_data(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    denom = np.where(max_val - min_val == 0, 1e-8, max_val - min_val)
    return (data - min_val) / denom

'''
Creates a mapping between compression ratio and error bounds, for every compression
method and every dataset/
'''
def create_bound_map(method, args):
    try:
        file_list = pd.read_csv(args.file_list)['file_name'].values
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
            norm_data = normalize_data(data) 
        except Exception as e:
            print(f"Failed to load or normalize {filename}: {e}")
            continue

        crs = []
        for bound in args.error_bounds:
            try:
                '''
                This is still not working as I intended,

                TerseTS explaination,
                "TerseTS takes a list of f64 values as input and returns a list of u8 values (bytes). 
                So for each input element, it outputs 8 u8 values. Once you store the elements, you’ll see the compression effect. 
                Otherwise, to compare lengths using len(X) (in Python), you need to divide the output length by 8. In the example you mentioned, 
                the output array contains 24 bytes for the compressed representation of 3 elements, plus 1 byte for storing the compression model. 
                In turn the input contains, 40 bytes"

                For several of the datasets the compressed data set is still larger than the original
                '''
                compressed_values = compress(data, method.value, bound)
                compressed_size = (len(compressed_values) - 1) / 8 
                cr = data.size / compressed_size
                crs.append(cr)
            except Exception as e:
                print(f"Compression failed for {filename}, method={method.name}, bound={bound}: {e}")
                crs.append(np.nan)

        crs = np.array(crs)

        try:
            '''
            The interpolation errors stem from errors realted to compression ratio, also at times the first 
            compression ratio of 0 gives a negative error bound after interpolation?
            '''
            interp_func = interp1d(crs, args.error_bounds, bounds_error=False, fill_value="extrapolate")
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


def experiment(method, args):
    bound_map_path = os.path.join(args.cr_map_dir, method.name, ".json")
    if not os.path.exists(bound_map_path):
        bound_map = create_bound_map(method, args)  
    else:
        with open(bound_map_path, "r") as f:
            bound_map = json.load(f)

    # Is there a way to further parallize this?
    for detector in args.ad_methods:
        Optimal_Det_HP = Optimal_Uni_algo_HP_dict[detector]
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
                error_bound = bound_map[dataset.split(".")[0]][str(cr)] # There is an error here because sometimes the error bounds are negative
                # Temporary fix
                if error_bound < 0:
                    error_bound = 0
                compressed_values = compress(data, method.value, error_bound)
                data = decompress(compressed_values)

                data = np.array(data)
                if data.ndim == 1:
                    data = data.reshape(-1, 1)
                label = df['Label'].astype(int).to_numpy()
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
                    eval_result = get_metrics(output, label, slidingWindow=slidingWindow)
                    row = [dataset, duration] + list(eval_result.values())
                    if columns is None:
                        columns = ['Dataset', 'Time'] + list(eval_result.keys())
                except Exception as e:
                    print(e)
                    row = [dataset, duration] + [0]*9  

                results_rows.append(row)

            if results_rows:
                out_dir = os.path.join("results", method.name, detector)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"{cr}.csv")

                df_all = pd.DataFrame(results_rows, columns=columns)
                df_all.to_csv(out_path, index=False)

if __name__ == '__main__':
    args = Args()

    try:
        file_list = pd.read_csv(args.file_list)['file_name'].values
    except Exception as e:
        print(e)

    def experiment_worker(method):
        experiment(method, args)

    '''
    I believe the Python GIL, prevents true parallelism on CPU-bound tasks because it only allows one thread to execute Python
    bytecode at a time, thus we need to move this to use multiple processes, this way each process has its own interpreter and GIL.
    Alternatively, we can distribute the workload across multiple machines.
    '''
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(
            experiment_worker,
            method
        )
        for method in Methods]

        for future in futures:
            future.result()

    print("All detectors finished.")
