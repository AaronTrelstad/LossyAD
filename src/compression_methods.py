import sys
import os
import struct

pysz_path = os.path.join(os.path.dirname(__file__), "..", "external", "SZ3", "tools", "pysz")
sys.path.append(pysz_path)
sys.path.append('/Users/aarontrelstad/cameo')
serf_pywrapper_path = os.environ.get(
    "SERF_PYWRAPPER_PATH",
    os.path.join(os.path.dirname(__file__), "..", "external", "Serf", "build", "pywrapper"),
)
sys.path.append(serf_pywrapper_path)

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
    _FRAME_MAGIC = b"SRF1"
    _FRAME_LEN_STRUCT = struct.Struct("<I")

    def __init__(self, error_bound=1e-3):
        from pyserf import ArrayOfBytes, PySerfXORCompressor, PySerfXORDecompressor

        self.error_bound = float(error_bound)
        self.window_size = 1000
        # SERF's upstream benchmarks flush small blocks and reuse codec state
        # across those blocks. Sending an entire long series through one close()
        # can overflow the native output buffer and corrupt the heap.
        self.block_size = 50
        self.adjust = 0

        self._array_of_bytes_cls = ArrayOfBytes
        self._compressor_cls = PySerfXORCompressor
        self._decompressor_cls = PySerfXORDecompressor

    def _pack_to_bytes(self, pack):
        if isinstance(pack, (bytes, bytearray, memoryview)):
            return bytes(pack)

        if hasattr(pack, "__getstate__"):
            state = pack.__getstate__()
            if isinstance(state, tuple) and state:
                return bytes(state[0])

        return bytes(pack)

    def compress(self, data):
        values = np.asarray(data, dtype=np.float64).reshape(-1)
        compressor = self._compressor_cls(self.window_size, self.error_bound, self.adjust)
        framed_bytes = bytearray(self._FRAME_MAGIC)

        for start in range(0, values.size, self.block_size):
            block = values[start:start + self.block_size]
            for val in block:
                compressor.add_value(float(val))

            compressor.close()
            pack = compressor.get()
            pack_bytes = self._pack_to_bytes(pack)
            framed_bytes.extend(self._FRAME_LEN_STRUCT.pack(len(pack_bytes)))
            framed_bytes.extend(pack_bytes)

        return bytes(framed_bytes)
    
    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        if isinstance(data_cmp, (bytes, bytearray, memoryview)):
            payload = bytes(data_cmp)
            decompressor = self._decompressor_cls(self.adjust)

            if payload.startswith(self._FRAME_MAGIC):
                offset = len(self._FRAME_MAGIC)
                values = []

                while offset < len(payload):
                    if offset + self._FRAME_LEN_STRUCT.size > len(payload):
                        raise ValueError("Corrupt SERF payload: truncated block header")

                    (block_len,) = self._FRAME_LEN_STRUCT.unpack_from(payload, offset)
                    offset += self._FRAME_LEN_STRUCT.size
                    block_end = offset + block_len

                    if block_end > len(payload):
                        raise ValueError("Corrupt SERF payload: truncated block data")

                    pack = self._array_of_bytes_cls(list(payload[offset:block_end]))
                    values.extend(decompressor.decompress(pack))
                    offset = block_end

                data_dec = np.asarray(values, dtype=np.float64)
            else:
                pack = self._array_of_bytes_cls(list(payload))
                data_dec = np.asarray(decompressor.decompress(pack), dtype=np.float64)
        else:
            pack = data_cmp
            decompressor = self._decompressor_cls(self.adjust)
            data_dec = np.asarray(decompressor.decompress(pack), dtype=np.float64)

        if original_shape is not None:
            target_size = int(np.prod(original_shape))
            if data_dec.size > target_size:
                data_dec = data_dec[:target_size]
            data_dec = data_dec.reshape(original_shape)

        if original_dtype is not None:
            data_dec = data_dec.astype(original_dtype, copy=False)

        return data_dec
    
class NoneCompressor:
    def __init__(self, error_bound):
        pass

    def compress(self, data):
        return data

    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        return data_cmp
