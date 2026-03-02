import sys
import os

pysz_path = os.path.join(os.path.dirname(__file__), "..", "external", "SZ3", "tools", "pysz")
sys.path.append(pysz_path)
sys.path.append('/Users/aarontrelstad/cameo')
sys.path.append("/Users/aarontrelstad/serf/build/pywrapper")

from compression.line_simplification import LineSimplification
from tersets import compress, decompress, Method
from pysz import SZ
from pyserf import PySerfXORCompressor, PySerfXORDecompressor

import numpy as np
from scipy.fftpack import dct, idct

class SZ3Compressor:
    def __init__(self, error_bound=1e-3):
        lib_path = os.path.join(
            os.path.dirname(__file__),
            "..", "external", "SZ3", "install", "lib", "libSZ3c.dylib"
        )
        self.sz = SZ(lib_path)
        self.error_bound = error_bound

    def compress(self, data):
        data_cmpr, _ = self.sz.compress(data, 0, self.error_bound, 0, 0)
        return data_cmpr

    def decompress(self, data_cmpr, original_shape, original_dtype):
        data_dec = self.sz.decompress(data_cmpr, original_shape, original_dtype)
        return data_dec

    def verify(self, original_data, decompressed_data):
        self.sz.verify(original_data, decompressed_data)

class MixPieceCompressor:
    def __init__(self, error_bound=1e-3):
        self.error_bound = error_bound

    def compress(self, data):
        data_cmpr = compress(data[0], Method.MixPiece, self.error_bound)
        return data_cmpr

    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        data_dec = decompress(data_cmp)
        return data_dec

class DWTCompressor:
    def __init__(self, error_bound=1e-3):
        pass

    def compress(self, data, fraction, _=None, __=None):
        dct_coeffs = dct(data, type=2, norm='ortho')

        num_coefficients = int(fraction * len(dct_coeffs))
        dct_coeffs[-num_coefficients:] = 0

        return dct_coeffs

    def decompress(self, data_cmp, original_shape, original_dtype):
        data_reconstructed = idct(data_cmp, type=2, norm='ortho')

        return data_reconstructed

class PIPCompressor:
    def __init__(self, error_bound=1e-3):
        self.pip = LineSimplification()
        self.pip.set_target(target='pip') 
        self.error_bound = error_bound
        self.nlags = 24

    def compress(self, data):
        #y = np.ascontiguousarray(y, dtype=np.float64)
        y = np.asarray(data).squeeze()
        hops = int(np.log(y.shape[0]) * 10)
        kappa = 1
        data_cmpr = self.pip.compress(y.copy(), self.error_bound, self.nlags, hops, kappa)
        return data_cmpr

    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        data_dec = self.pip.decompress(data_cmp)
        return data_dec

class PySerfCompressor:
    def __init__(self, error_bound=1e-3):
        self.error_bound = error_bound
        self.window_size = 1000
        self.adjust = 0

        self.compressor = PySerfXORCompressor(self.window_size, self.error_bound, self.adjust)
        self.decompressor = PySerfXORDecompressor(self.adjust)

    def compress(self, data):
        for val in data:
            self.compressor.add_value(val)
        
        self.compressor.close()
        pack = self.compressor.get()
        return pack
    
    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        data_dec = self.decompressor.decompress(data_cmp)
        return data_dec
    
class NoneCompressor:
    def __init__(self, error_bound):
        pass

    def compress(self, data):
        return data

    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        return data_cmp
