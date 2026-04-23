import sys
import os

pysz_path = os.path.join(os.path.dirname(__file__), "..", "external", "SZ3", "tools", "pysz")
sys.path.append(pysz_path)
sys.path.append('/Users/aarontrelstad/cameo')
sys.path.append("/Users/aarontrelstad/serf/build/pywrapper")

import numpy as np
from scipy.fftpack import dct, idct

class SZ3Compressor:
    def __init__(self, error_bound=1e-3):
        from pysz import SZ

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
        from tersets import compress, decompress, Method

        self._compress = compress
        self._decompress = decompress
        self._method = Method.MixPiece
        self.error_bound = error_bound

    def compress(self, data):
        data_cmpr = self._compress(data[0], self._method, self.error_bound)
        return data_cmpr

    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        data_dec = self._decompress(data_cmp)
        return data_dec

class DWTCompressor:
    def __init__(self, error_bound=1e-3):
        self.error_bound = float(error_bound)

    def _keep_fraction(self):
        # Map error bound to retained coefficient fraction.
        bounded = min(max(self.error_bound, 0.0), 0.99)
        return max(0.01, 1.0 - bounded)

    def compress(self, data):
        values = np.asarray(data, dtype=np.float64)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        dct_coeffs = dct(values, type=2, norm='ortho', axis=0)
        keep_fraction = self._keep_fraction()
        keep_count = max(1, int(np.ceil(dct_coeffs.shape[0] * keep_fraction)))

        compressed = np.zeros_like(dct_coeffs)

        for col in range(dct_coeffs.shape[1]):
            magnitudes = np.abs(dct_coeffs[:, col])
            keep_idx = np.argpartition(magnitudes, -keep_count)[-keep_count:]
            compressed[keep_idx, col] = dct_coeffs[keep_idx, col]

        return compressed

    def decompress(self, data_cmp, original_shape, original_dtype):
        coeffs = np.asarray(data_cmp, dtype=np.float64)
        if coeffs.ndim == 1:
            coeffs = coeffs.reshape(-1, 1)

        data_reconstructed = idct(coeffs, type=2, norm='ortho', axis=0)
        if original_shape is not None:
            data_reconstructed = data_reconstructed.reshape(original_shape)
        if original_dtype is not None:
            data_reconstructed = data_reconstructed.astype(original_dtype, copy=False)

        return data_reconstructed

class PIPCompressor:
    def __init__(self, error_bound=1e-3):
        from compression.line_simplification import LineSimplification

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
        from pyserf import PySerfXORCompressor, PySerfXORDecompressor

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
