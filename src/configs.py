import random
import numpy as np
import torch

from TSB_AD.HP_list import Optimal_Uni_algo_HP_dict
from enum import Enum
from .compression_methods import SZ3Compressor, PIPCompressor, DWTCompressor, PySerfCompressor, MixPieceCompressor, NoneCompressor

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    print("CUDA Available: ", torch.cuda.is_available())
    print("cuDNN Version: ", torch.backends.cudnn.version())

class MethodType(Enum):
    #NONE = NoneCompressor
    # SZ3 = SZ3Compressor
    #PIP = PIPCompressor
    # DWT = DWTCompressor
    SERF = PySerfCompressor
    #MP = MixPieceCompressor # Need to validate this

class ExperimentConfig:
    def __init__(self):
        self.seed = 2024

        self.results_dir = 'results/'
        self.cr_map_dir = 'cr_bound_maps/'
        self.dataset_dir = 'Datasets/TSB-AD-U'

        self.dataset_list = 'Datasets/File_List/TSB-AD-U-Test.csv'

        self.compression_ratios = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50]
        self.error_bounds = np.linspace(0, 0.8, 100)

        # for all AD methods use: list(Optimal_Uni_algo_HP_dict.keys())
        self.ad_methods = ['Sub_PCA', 'KShapeAD', 'POLY', 'Sub_KNN', 'SR', 'CNN', 'LSTMAD', 'USAD'] # 'MOMENT_FT', 'MOMENT_ZS'

        self.chart = False

class AnalysisConfig:
    def __init__(self):
        self.results_dir = 'results/'
        self.ad_methods = ['Sub_PCA', 'KShapeAD', 'POLY', 'Sub_KNN', 'SR', 'CNN', 'LSTMAD', 'USAD', 'MOMENT_FT', 'MOMENT_ZS']

