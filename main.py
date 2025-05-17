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
'''
class Method(Enum):
    PMC_M = Method.PoorMansCompressionMean
    PMC_MR = Method.PoorMansCompressionMidrange
    SWING = Method.SwingFilter
    #SLIDE = Method.SlideFilter

'''
TODO:
1. Have each detector also go through each compression ratio, load the compression mapping and use related error bound.
'''
def run_AD(detector_name, file_list, args):
    target_dir = os.path.join(args.score_dir, detector_name)
    os.makedirs(target_dir, exist_ok=True)

    logger = logging.getLogger(detector_name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        log_file = os.path.join(target_dir, f'{detector_name}.log')
        fh = logging.FileHandler(log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    Optimal_Det_HP = Optimal_Uni_algo_HP_dict[detector_name]
    print(f'[{detector_name}] Optimal HP: {Optimal_Det_HP}')

    columns = [
        'file', 'Time',
        'AUC-PR', 'AUC-ROC', 'VUS-PR', 'VUS-ROC',
        'Standard-F1', 'PA-F1', 'Event-based-F1', 'R-based-F1', 'Affiliation-F'
    ]
    out_path = os.path.join(args.save_dir, f'{detector_name}.csv')
    os.makedirs(args.save_dir, exist_ok=True)

    write_header = not os.path.exists(out_path)
    if write_header:
        pd.DataFrame(columns=columns).to_csv(out_path, index=False)

    for filename in file_list:
        print(f'[{detector_name}] Processing {filename}')
        file_path = os.path.join(args.dataset_dir, filename)

        try:
            df = pd.read_csv(file_path).dropna()
        except Exception as e:
            logger.error(f"Could not read {filename}: {e}")
            continue

        data = df.iloc[:, :-1].values.astype(float)
        label = df['Label'].astype(int).to_numpy()
        slidingWindow = find_length_rank(data[:, 0].reshape(-1, 1), rank=1)
        train_index = int(filename.split('.')[0].split('_')[-3])
        data_train = data[:train_index, :]

        try:
            start = time.time()
            if detector_name in Semisupervise_AD_Pool:
                output = run_Semisupervise_AD(detector_name, data_train, data, **Optimal_Det_HP)
            elif detector_name in Unsupervise_AD_Pool:
                output = run_Unsupervise_AD(detector_name, data, **Optimal_Det_HP)
            else:
                raise ValueError(f"Unknown detector: {detector_name}")
            duration = time.time() - start
        except Exception as e:
            logger.error(f"Error running {detector_name} on {filename}: {e}")
            continue

        if args.save:
            try:
                eval_result = get_metrics(output, label, slidingWindow=slidingWindow)
                row = [filename, duration] + list(eval_result.values())
            except Exception as e:
                logger.error(f"Evaluation error on {filename}: {e}")
                row = [filename, duration] + [0]*9

            df_row = pd.DataFrame([row], columns=columns)
            df_row.to_csv(out_path, mode='a', header=False, index=False)
            logger.info(f'Success at {filename} using {detector_name} | Time cost: {duration:.3f}s at length {len(label)}')

'''
Normalize all of the values in the dataset between [0, 1]
'''
def normalize_data(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    denom = np.where(max_val - min_val == 0, 1e-8, max_val - min_val)
    return (data - min_val) / denom

'''
Create the error bound to compression ratio mappings
'''
def generate_error_bound_mapping(args, cr_targets, error_bounds, output_dir="cr_bound_maps"):
    try:
        file_list = pd.read_csv(args.file_list)['file_name'].values
    except Exception as e:
        print(f"Failed to read file list: {e}")
        return {}

    os.makedirs(output_dir, exist_ok=True)

    for method in Method: 
        dataset_mappings = []

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
            for bound in error_bounds:
                try:
                    '''
                    I still believe this is not working as I intended,

                    From Carlos,
                    "TerseTS takes a list of f64 values as input and returns a list of u8 values (bytes). 
                    So for each input element, it outputs 8 u8 values. Once you store the elements, you’ll see the compression effect. 
                    Otherwise, to compare lengths using len(X) (in Python), you need to divide the output length by 8. In the example you mentioned, 
                    the output array contains 24 bytes for the compressed representation of 3 elements, plus 1 byte for storing the compression model. 
                    In turn the input contains, 40 bytes"

                    For most datasets the compressed data set is still larger than the original
                    '''

                    # I think the sizes here are not being calculated correctly
                    compressed_values = compress(data, method.value, bound)
                    compressed_size = len(compressed_values) / 8
                    cr = data.size / compressed_size

                    print(len(data), compressed_size)
                    crs.append(cr)
                except Exception as e:
                    print(f"Compression failed for {filename}, method={method.name}, bound={bound}: {e}")
                    crs.append(np.nan)

            crs = np.array(crs)

            try:
                '''
                In certain datasets the interpolation is not successful, this stems from the error in the compression
                ratio calculations
                '''
                interp_func = interp1d(crs, error_bounds, bounds_error=False, fill_value="extrapolate")
            except Exception as e:
                print(f"Interpolation failed for {filename}, method={method.name}: {e}")
                continue

            cr_bound_mapping = {}
            for cr_target in cr_targets:
                try:
                    bound = float(interp_func(cr_target))
                    cr_bound_mapping[str(cr_target)] = round(bound, 6)
                except Exception:
                    continue

            dataset_mappings.append({
                "dataset": filename,
                "data": cr_bound_mapping
            })

        output_data = {
            "compression_method": method.name,
            "datasets": dataset_mappings
        }

        output_path = os.path.join(output_dir, f"cr_bound_map_{method.name}.json")
        with open(output_path, "w") as f:
            json.dump(output_data, f, indent=4)

        print(f"Mapping for method {method.name} complete. Saved to {output_path}")


if __name__ == '__main__':
    cwd = os.getcwd()
    parser = argparse.ArgumentParser(description='TSB-Benchmark')
    parser.add_argument('--dataset_dir', type=str, default="Datasets/TSB-AD-U")
    parser.add_argument('--file_list', type=str, default="Datasets/File_List/TSB-AD-U-Test.csv") # You can modify this to choose which datasets to use
    parser.add_argument('--score_dir', type=str, default='eval/score/uni/')
    parser.add_argument('--save_dir', type=str, default='eval/metrics/uni/')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--output_dir', type=str, default='cr_bound_maps')
    parser.add_argument('--threads', type=int, default=6) # Need to identify ideal thread amount to prevent resource saturation
    args = parser.parse_args()

    '''
    Find out about available compute resources:
    Idealy we can get enough resources to run half of the detectors at a time, a full run would then take around
    8ish hours, then for 10 compression ratios we could estimate about 4 days of total test time?
    '''

    '''
    Concept: Test with ~100 errors bounds then using the achieved compression ratios
    we can interpolate to find the error bound to reach the desired compression ratio.
    '''
    compression_ratios = [3, 5, 7, 10, 13, 15, 17, 20, 25, 30, 40, 50]
    error_bounds = np.linspace(0, 0.8, 100)

    '''
    Create JSON file that stores the map for each compression method, after interpolation:
    {

        "compression_method": "example",
        "datasets": {
            [
                {
                    "dataset": "dojsfnsjndj",
                    "data": {
                        "compression_ratio_1": "error_bound_1",
                        "compression_ratio_2": "error_bound_2,
                        ...
                    }
                }
            ]       
        }
    }

    TODO: If their already exists a map we don't need to create another one
    '''
    generate_error_bound_mapping(args, compression_ratios, error_bounds, output_dir=args.output_dir)

    detectors = list(Optimal_Uni_algo_HP_dict.keys())

    try:
        file_list = pd.read_csv(args.file_list)['file_name'].values
    except Exception as e:
        logger.error(f"Failed to read file list: {e}")

    def run_detector_wrapper(detector_name):
        print(f"[{detector_name}] Detector started.")
        run_AD(detector_name, file_list, args)
        print(f"[{detector_name}] Detector finished.")

    with ThreadPoolExecutor(max_workers=args.threads) as executor:
        futures = [executor.submit(run_detector_wrapper, det) for det in detectors]
        for future in futures:
            future.result()

    print("All detectors finished. Check logs for progress.")
