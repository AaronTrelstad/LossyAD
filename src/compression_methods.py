"""
compression_methods.py — lossy compressor classes for the LossyAD benchmark.

All compressors share a common interface::

    compress(data: np.ndarray) -> bytes
    decompress(data_cmp, original_shape, original_dtype) -> np.ndarray

Multivariate data (shape N×D) is handled by compressing each channel (column)
independently and concatenating the results.

Compressor Families
-------------------
The compressors implemented here span the major families identified in the
time-series compression literature (see also Xenophontos & Palpanas, EDBT 2026):

  Family                        | Key idea                              | Implementations
  ----------------------------- | ------------------------------------- | ---------------
  Passthrough / baseline        | No compression                        | NONE
  Error-bounded (sci. compute)  | Prediction + quantisation, ε-bounded  | SZ3, ZFP
  Transform / frequency         | Wavelet coefficients thresholded      | DWT
  Perceptual / shape-preserving | Keep "important" points, interpolate  | PIP, VW
  Piecewise linear (PLR)        | Fit line segments within error bound  | MP, SWING, SIMPIE, SLIDE
  Uniform quantisation          | Reduce bit depth uniformly            | QUANT
  XOR / streaming               | XOR-encode consecutive floats         | GORILLA, SERF

Availability
------------
Always available (pure Python or pip only):
  NONE, PIP, QUANT, GORILLA

Pip-installable (probed at import):
  ZFP   — pip install zfpy
  DWT   — pip install PyWavelets
  MP, SWING, SIMPIE, SLIDE, VW — pip install git+https://github.com/cmcuza/TerseTS.git

Optional native builds (gracefully skipped if absent):
  SZ3  — build from external/SZ3  (cmake)   or set SZ3_LIB_PATH
  SERF — build from external/Serf (cmake)   or set SERF_PYWRAPPER_PATH

If a native library is absent the class raises RuntimeError on instantiation.
The MethodType enum in configs.py automatically excludes unavailable compressors.
"""

import os
import sys
import struct
import platform

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compress_channels(compress_fn, data: np.ndarray) -> list[bytes]:
    """
    Apply *compress_fn* (single 1-D array → bytes) to each column of *data*
    and return the list of compressed byte strings.
    """
    data2d = data.reshape(len(data), -1) if data.ndim == 1 else data
    return [compress_fn(data2d[:, c]) for c in range(data2d.shape[1])]


def _decompress_channels(
    decompress_fn,
    channel_blobs: list[bytes],
    original_shape: tuple,
    original_dtype,
) -> np.ndarray:
    """
    Decompress each channel blob with *decompress_fn* (bytes → 1-D array) and
    stack them back into *original_shape*.
    """
    n_rows = original_shape[0]
    channels = [decompress_fn(blob, (n_rows,), original_dtype) for blob in channel_blobs]
    out = np.stack(channels, axis=1)          # (N, D)
    return out.reshape(original_shape)


def _pack_channels(channel_blobs: list[bytes]) -> bytes:
    """
    Pack a list of per-channel byte strings into a single bytes object.
    Format: [n_channels:u32]  (  [len:u32][data:len×u8]  ) × n_channels
    """
    header = struct.pack("<I", len(channel_blobs))
    parts  = [header]
    for blob in channel_blobs:
        parts.append(struct.pack("<I", len(blob)))
        parts.append(blob)
    return b"".join(parts)


def _unpack_channels(payload: bytes) -> list[bytes]:
    """Inverse of _pack_channels."""
    n_channels, = struct.unpack_from("<I", payload, 0)
    offset = 4
    blobs = []
    for _ in range(n_channels):
        length, = struct.unpack_from("<I", payload, offset)
        offset += 4
        blobs.append(payload[offset : offset + length])
        offset += length
    return blobs


# ---------------------------------------------------------------------------
# Passthrough baseline
# ---------------------------------------------------------------------------

class NoneCompressor:
    """No compression — returns data unchanged."""

    def __init__(self, error_bound=0.0):
        pass

    def compress(self, data: np.ndarray):
        return data

    def decompress(self, data_cmp, original_shape=None, original_dtype=None):
        return data_cmp


# ---------------------------------------------------------------------------
# ZFP — error-bounded, pip-installable  (pip install zfpy)
# ---------------------------------------------------------------------------

class ZFPCompressor:
    """
    Error-bounded lossy compressor using ZFP.

    ZFP uses orthogonal block transforms with rate or precision control.
    In *accuracy* mode the maximum absolute error is bounded by *error_bound*.

    Install: pip install zfpy

    Reference: Lindstrom, P. (2014). Fixed-rate compressed floating-point arrays.
    """

    def __init__(self, error_bound: float = 1e-3):
        try:
            import zfpy
            self._zfpy = zfpy
        except ImportError as exc:
            raise RuntimeError(
                "zfpy is required for ZFPCompressor.  Install with: pip install zfpy"
            ) from exc
        self.error_bound = float(error_bound)

    def _compress_1d(self, arr: np.ndarray) -> bytes:
        arr = np.ascontiguousarray(arr, dtype=np.float64)
        return self._zfpy.compress_numpy(arr, tolerance=self.error_bound)

    def _decompress_1d(self, blob: bytes, shape, dtype) -> np.ndarray:
        arr = self._zfpy.decompress_numpy(blob)
        return arr[:shape[0]].astype(dtype, copy=False)

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        blobs = _compress_channels(self._compress_1d, data)
        return _pack_channels(blobs)

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        blobs = _unpack_channels(bytes(data_cmp))
        dtype = original_dtype if original_dtype is not None else np.float64
        return _decompress_channels(self._decompress_1d, blobs, original_shape, dtype)


# ---------------------------------------------------------------------------
# ZFP2D — ZFP applied natively to full N×D matrix  (pip install zfpy)
# ---------------------------------------------------------------------------

class ZFP2DCompressor:
    """
    Multivariate-native ZFP compressor.

    Unlike the channel-wise ZFPCompressor, this passes the full N×D matrix
    directly to ZFP's 2-D block transform (4×4 blocks), allowing it to exploit
    correlations *across channels* as well as along the time axis.

    For univariate data (D=1) the behaviour is identical to ZFPCompressor.

    error_bound: maximum absolute pointwise error (same semantics as ZFPCompressor).

    Install: pip install zfpy
    Reference: Lindstrom, P. (2014). Fixed-rate compressed floating-point arrays.
    """

    def __init__(self, error_bound: float = 1e-3):
        try:
            import zfpy
            self._zfpy = zfpy
        except ImportError as exc:
            raise RuntimeError(
                "zfpy is required for ZFP2DCompressor.  Install with: pip install zfpy"
            ) from exc
        self.error_bound = float(error_bound)

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        # Ensure C-contiguous so ZFP sees the correct memory layout
        arr = np.ascontiguousarray(data)
        shape = np.array(list(arr.shape), dtype=np.int32)
        shape_header = struct.pack("<I", arr.ndim) + shape.tobytes()
        compressed = self._zfpy.compress_numpy(arr, tolerance=self.error_bound)
        return shape_header + compressed

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        ndim, = struct.unpack_from("<I", payload, 0)
        stored_shape = tuple(
            int(x) for x in np.frombuffer(payload, dtype=np.int32, count=ndim, offset=4)
        )
        offset = 4 + ndim * 4
        arr = self._zfpy.decompress_numpy(payload[offset:])
        shape = original_shape if original_shape is not None else stored_shape
        dtype = original_dtype if original_dtype is not None else np.float64
        return arr.reshape(shape).astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# DWT — Discrete Wavelet Transform  (pip install PyWavelets)
# ---------------------------------------------------------------------------

class DWTCompressor:
    """
    Lossy compressor using the Discrete Wavelet Transform (Daubechies-4).

    Detail coefficients are hard-thresholded; the approximation sub-band is
    always kept to preserve coarse shape.

    error_bound in [0, 1):
        0.0  → keep 100 % of detail coefficients (near-lossless)
        0.99 → keep   1 % of detail coefficients (maximum compression)

    Install: pip install PyWavelets
    """

    WAVELET = "db4"
    MODE    = "periodization"

    def __init__(self, error_bound: float = 1e-3):
        try:
            import pywt
            self._pywt = pywt
        except ImportError as exc:
            raise RuntimeError(
                "PyWavelets is required for DWTCompressor.  Install with: pip install PyWavelets"
            ) from exc
        self.error_bound = float(error_bound)

    def _keep_fraction(self) -> float:
        return max(0.01, 1.0 - min(max(self.error_bound, 0.0), 0.99))

    def _compress_1d(self, arr: np.ndarray) -> bytes:
        arr = np.asarray(arr, dtype=np.float64)
        coeffs = self._pywt.wavedec(arr, self.WAVELET, mode=self.MODE)
        kf = self._keep_fraction()

        thresholded = [coeffs[0]]
        for detail in coeffs[1:]:
            n_keep = max(1, int(np.ceil(len(detail) * kf)))
            if n_keep >= len(detail):
                thresholded.append(detail.copy())
            else:
                idx = np.argpartition(np.abs(detail), -n_keep)[-n_keep:]
                tc = np.zeros_like(detail)
                tc[idx] = detail[idx]
                thresholded.append(tc)

        flat       = np.concatenate([c.ravel() for c in thresholded])
        level_lens = np.array([len(c) for c in thresholded], dtype=np.int32)
        n          = np.int32(len(arr))
        n_levels   = np.int32(len(thresholded))

        # Header: n (int32) | n_levels (int32) | level_lens (n_levels × int32) | coeffs (float64)
        return (
            n.tobytes()
            + n_levels.tobytes()
            + level_lens.tobytes()
            + flat.astype(np.float64).tobytes()
        )

    def _decompress_1d(self, blob: bytes, shape, dtype) -> np.ndarray:
        n        = int(np.frombuffer(blob, dtype=np.int32, count=1, offset=0)[0])
        n_levels = int(np.frombuffer(blob, dtype=np.int32, count=1, offset=4)[0])
        offset   = 8  # past n + n_levels

        level_lens = np.frombuffer(blob, dtype=np.int32, count=n_levels, offset=offset)
        offset    += n_levels * 4

        flat = np.frombuffer(blob, dtype=np.float64, offset=offset)
        coeffs, idx = [], 0
        for length in level_lens:
            coeffs.append(flat[idx : idx + length].copy())
            idx += length

        reconstructed = self._pywt.waverec(coeffs, self.WAVELET, mode=self.MODE)
        return reconstructed[:n].astype(dtype, copy=False)

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        original_shape = np.array(list(data.shape), dtype=np.int32)
        blobs = _compress_channels(self._compress_1d, data)
        # prepend ndim + shape so decompress can reconstruct exactly
        shape_header = struct.pack("<I", len(data.shape)) + original_shape.tobytes()
        return shape_header + _pack_channels(blobs)

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        ndim, = struct.unpack_from("<I", payload, 0)
        stored_shape = tuple(
            int(x) for x in np.frombuffer(payload, dtype=np.int32, count=ndim, offset=4)
        )
        offset = 4 + ndim * 4

        blobs = _unpack_channels(payload[offset:])
        shape = original_shape if original_shape is not None else stored_shape
        dtype = original_dtype if original_dtype is not None else np.float64
        return _decompress_channels(self._decompress_1d, blobs, shape, dtype)


# ---------------------------------------------------------------------------
# DWT2D — 2-D Discrete Wavelet Transform on full N×D matrix (pip install PyWavelets)
# ---------------------------------------------------------------------------

class DWT2DCompressor:
    """
    Multivariate-native DWT compressor using 2-D wavelet decomposition.

    Unlike the channel-wise DWTCompressor, this applies ``pywt.wavedec2`` to
    the full N×D matrix, decomposing along *both* the time axis and the channel
    axis simultaneously.  This allows coefficients that capture cross-channel
    correlation (cH, cV subbands) to be kept or discarded together, yielding
    better compression on datasets whose channels are correlated.

    For univariate data (D=1) the full matrix is a (N,1) array and the 2-D
    transform degenerates to a 1-D transform — identical to DWTCompressor.

    Requires D ≥ 1 and N ≥ 2^level.  Level is chosen automatically as
    ``floor(log2(min(N, D)))`` capped at 6.

    error_bound in [0, 1): fraction of detail coefficients *discarded* at each
    subband, same semantics as DWTCompressor.

    Install: pip install PyWavelets
    """

    WAVELET = "db4"
    MODE    = "periodization"

    def __init__(self, error_bound: float = 1e-3):
        try:
            import pywt
            self._pywt = pywt
        except ImportError as exc:
            raise RuntimeError(
                "PyWavelets is required for DWT2DCompressor.  Install with: pip install PyWavelets"
            ) from exc
        self.error_bound = float(error_bound)

    def _keep_fraction(self) -> float:
        return max(0.01, 1.0 - min(max(self.error_bound, 0.0), 0.99))

    def _n_levels(self, n_rows: int, n_cols: int) -> int:
        return min(6, int(np.floor(np.log2(max(min(n_rows, n_cols), 1)))))

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        n_rows, n_cols = data.shape
        level = self._n_levels(n_rows, n_cols)
        kf    = self._keep_fraction()

        coeffs = self._pywt.wavedec2(data, self.WAVELET, mode=self.MODE, level=level)
        # coeffs[0]: approximation (cA), shape varies
        # coeffs[1..level]: tuples (cH, cV, cD) at each detail level

        thresholded = [coeffs[0]]   # always keep full approximation
        for detail_tuple in coeffs[1:]:
            new_tuple = []
            for subband in detail_tuple:
                flat   = subband.ravel()
                n_keep = max(1, int(np.ceil(len(flat) * kf)))
                if n_keep >= len(flat):
                    new_tuple.append(subband.copy())
                else:
                    idx = np.argpartition(np.abs(flat), -n_keep)[-n_keep:]
                    tc  = np.zeros_like(flat)
                    tc[idx] = flat[idx]
                    new_tuple.append(tc.reshape(subband.shape))
            thresholded.append(tuple(new_tuple))

        # Serialise: store original shape + level, then each band's shape + data
        header = struct.pack("<IIII", n_rows, n_cols, level, len(thresholded))
        parts  = [header]
        # approximation
        cA = thresholded[0]
        parts.append(struct.pack("<II", *cA.shape))
        parts.append(cA.astype(np.float64).tobytes())
        # detail bands
        for detail_tuple in thresholded[1:]:
            for subband in detail_tuple:   # always 3 subbands: cH, cV, cD
                parts.append(struct.pack("<II", *subband.shape))
                parts.append(subband.astype(np.float64).tobytes())

        return b"".join(parts)

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        n_rows, n_cols, level, n_bands = struct.unpack_from("<IIII", payload, 0)
        offset = 16

        def _read_array(off):
            r, c = struct.unpack_from("<II", payload, off)
            off  += 8
            arr   = np.frombuffer(payload, dtype=np.float64, count=r * c, offset=off).copy()
            return arr.reshape(r, c), off + r * c * 8

        cA, offset = _read_array(offset)
        coeffs     = [cA]
        for _ in range(level):
            subbands = []
            for _ in range(3):   # cH, cV, cD
                sb, offset = _read_array(offset)
                subbands.append(sb)
            coeffs.append(tuple(subbands))

        reconstructed = self._pywt.waverec2(coeffs, self.WAVELET, mode=self.MODE)
        # waverec2 may add 1 extra row/col due to periodization; trim to original
        result = reconstructed[:n_rows, :n_cols]

        shape = original_shape if original_shape is not None else (n_rows, n_cols)
        dtype = original_dtype if original_dtype is not None else np.float64
        return result.reshape(shape).astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# PIP — Perceptually Important Points  (pure Python, built-in)
# ---------------------------------------------------------------------------

class PIPCompressor:
    """
    Lossy compressor using Perceptually Important Points (PIP) with ACF preservation.

    Points are removed in order of increasing perpendicular distance until the
    mean absolute ACF deviation exceeds *error_bound*.  Removed points are
    reconstructed by linear interpolation.

    error_bound: maximum allowable mean absolute ACF deviation (0 = lossless).
    Pure Python — no external dependencies beyond NumPy.
    """

    def __init__(self, error_bound: float = 1e-3):
        self.error_bound = float(error_bound)
        self.nlags = 24

    def _compress_1d(self, arr: np.ndarray) -> bytes:
        from .pip_helpers import simplify_by_pip

        y = np.asarray(arr, dtype=np.float64)
        original_len = len(y)
        y_work = y.copy()
        mask = simplify_by_pip(y_work, nlags=self.nlags, acf_threshold=self.error_bound)

        kept_indices = np.where(mask)[0].astype(np.int32)
        kept_values  = y[kept_indices]

        header = struct.pack("<II", original_len, len(kept_indices))
        return header + kept_indices.tobytes() + kept_values.astype(np.float64).tobytes()

    def _decompress_1d(self, blob: bytes, shape, dtype) -> np.ndarray:
        payload = bytes(blob)
        original_len, n_kept = struct.unpack_from("<II", payload, 0)
        offset = 8
        indices = np.frombuffer(payload, dtype=np.int32,  count=n_kept, offset=offset)
        offset += n_kept * 4
        values  = np.frombuffer(payload, dtype=np.float64, count=n_kept, offset=offset)
        out = np.interp(np.arange(original_len), indices, values)
        return out[:shape[0]].astype(dtype, copy=False)

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        original_shape = np.array(list(data.shape), dtype=np.int32)
        blobs = _compress_channels(self._compress_1d, data)
        shape_header = struct.pack("<I", len(data.shape)) + original_shape.tobytes()
        return shape_header + _pack_channels(blobs)

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        ndim, = struct.unpack_from("<I", payload, 0)
        stored_shape = tuple(
            int(x) for x in np.frombuffer(payload, dtype=np.int32, count=ndim, offset=4)
        )
        offset = 4 + ndim * 4
        blobs = _unpack_channels(payload[offset:])
        shape = original_shape if original_shape is not None else stored_shape
        dtype = original_dtype if original_dtype is not None else np.float64
        return _decompress_channels(self._decompress_1d, blobs, shape, dtype)


# ---------------------------------------------------------------------------
# Quantization  (pure Python)
# ---------------------------------------------------------------------------

class QuantizationCompressor:
    """
    Uniform quantization compressor via bit-depth reduction.

    Assumes data is normalised to [0, 1].

    error_bound controls the effective bit depth:
        0.0  → 16-bit (near-lossless for normalised float data)
        0.25 → 12-bit
        0.5  →  8-bit
        0.75 →  6-bit
        1.0  →  4-bit  (very coarse)

    Pure Python — no external dependencies beyond NumPy.
    """

    _MIN_BITS = 4
    _MAX_BITS = 16

    def __init__(self, error_bound: float = 0.0):
        self.error_bound = float(error_bound)

    def _n_bits(self) -> int:
        frac = min(max(self.error_bound, 0.0), 1.0)
        return max(self._MIN_BITS,
                   min(self._MAX_BITS,
                       int(round(self._MAX_BITS - frac * (self._MAX_BITS - self._MIN_BITS)))))

    def compress(self, data: np.ndarray) -> bytes:
        n_bits   = self._n_bits()
        n_levels = (1 << n_bits) - 1
        values   = np.asarray(data, dtype=np.float64)
        shape    = np.array(list(values.shape), dtype=np.int32)
        flat     = values.ravel()

        quantized = np.clip(np.round(flat * n_levels), 0, n_levels)
        storage   = quantized.astype(np.uint8 if n_bits <= 8 else np.uint16).tobytes()

        header = struct.pack("<BI", n_bits, len(flat))
        shape_header = struct.pack("<I", len(values.shape)) + shape.tobytes()
        return shape_header + header + storage

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        ndim, = struct.unpack_from("<I", payload, 0)
        stored_shape = tuple(
            int(x) for x in np.frombuffer(payload, dtype=np.int32, count=ndim, offset=4)
        )
        offset = 4 + ndim * 4

        n_bits, n = struct.unpack_from("<BI", payload, offset)
        offset += 5

        n_levels  = (1 << n_bits) - 1
        dtype_q   = np.uint8 if n_bits <= 8 else np.uint16
        quantized = np.frombuffer(payload, dtype=dtype_q, count=n, offset=offset).astype(np.float64)
        reconstructed = quantized / n_levels

        shape = original_shape if original_shape is not None else stored_shape
        dtype = original_dtype if original_dtype is not None else np.float64
        return reconstructed.reshape(shape).astype(dtype, copy=False)


# ---------------------------------------------------------------------------
# MixPiece — piecewise linear  (pip install git+https://github.com/cmcuza/TerseTS)
# ---------------------------------------------------------------------------

class _TerseTSCompressor:
    """
    Generic wrapper for any TerseTS compression method.

    Subclasses set ``_METHOD_NAME`` to the string name of the TerseTS Method
    enum member (e.g. "MixPiece", "SwingFilter", "SimPiece").

    error_bound is an absolute error bound on the normalised signal.

    Install: pip install git+https://github.com/cmcuza/TerseTS.git
    """

    _METHOD_NAME: str = ""

    def __init__(self, error_bound: float = 1e-3):
        try:
            from tersets import compress, decompress, Method
            self._compress_fn   = compress
            self._decompress_fn = decompress
            self._method        = Method[self._METHOD_NAME]
        except ImportError as exc:
            raise RuntimeError(
                f"TerseTS is required for {self.__class__.__name__}.\n"
                "Install with: pip install git+https://github.com/cmcuza/TerseTS.git"
            ) from exc
        except KeyError:
            raise RuntimeError(
                f"TerseTS method '{self._METHOD_NAME}' not found in installed TerseTS version."
            )
        self.error_bound = float(error_bound)

    def _compress_1d(self, arr: np.ndarray) -> bytes:
        y      = np.asarray(arr, dtype=np.float64).squeeze()
        result = self._compress_fn(y, self._method, self.error_bound)
        return bytes(result) if not isinstance(result, bytes) else result

    def _decompress_1d(self, blob: bytes, shape, dtype) -> np.ndarray:
        arr = np.asarray(self._decompress_fn(blob), dtype=np.float64)
        return arr[: shape[0]].astype(dtype, copy=False)

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        original_shape = np.array(list(data.shape), dtype=np.int32)
        blobs = _compress_channels(self._compress_1d, data)
        shape_header = struct.pack("<I", len(data.shape)) + original_shape.tobytes()
        return shape_header + _pack_channels(blobs)

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        ndim,  = struct.unpack_from("<I", payload, 0)
        stored_shape = tuple(
            int(x) for x in np.frombuffer(payload, dtype=np.int32, count=ndim, offset=4)
        )
        offset = 4 + ndim * 4
        blobs  = _unpack_channels(payload[offset:])
        shape  = original_shape if original_shape is not None else stored_shape
        dtype  = original_dtype if original_dtype is not None else np.float64
        return _decompress_channels(self._decompress_1d, blobs, shape, dtype)


class MixPieceCompressor(_TerseTSCompressor):
    """
    MixPiece piecewise-linear compressor (best-in-class PLR via TerseTS).

    Jointly fits multi-segment piecewise linear approximations to minimise
    the number of segments at a given error bound.

    Reference: Xenophontos & Palpanas, VLDB 2023.
    """
    _METHOD_NAME = "MixPiece"


class SwingFilterCompressor(_TerseTSCompressor):
    """
    Swing Filter piecewise-linear compressor (classic PLR via TerseTS).

    The original error-bounded PLR algorithm; widely used as a baseline
    in time-series compression literature.

    Reference: Elmeleegy et al., VLDB 2009.
    """
    _METHOD_NAME = "SwingFilter"


class SimPieceCompressor(_TerseTSCompressor):
    """
    SimPiece piecewise-linear compressor (simplified MixPiece via TerseTS).

    Achieves near-MixPiece compression ratios with lower computational cost.

    Reference: Xenophontos & Palpanas, VLDB 2022.
    """
    _METHOD_NAME = "SimPiece"


class SlideFilterCompressor(_TerseTSCompressor):
    """
    Slide Filter piecewise-linear compressor (sliding-window PLR via TerseTS).

    Maintains a sliding error cone and emits a new segment whenever the
    current reading falls outside the cone.  Lower memory than SwingFilter
    for streaming settings.

    Reference: Elmeleegy et al., VLDB 2009.
    """
    _METHOD_NAME = "SlideFilter"


class VisvalingamWhyattCompressor(_TerseTSCompressor):
    """
    Visvalingam–Whyatt line simplification compressor (via TerseTS).

    Iteratively removes the point that forms the triangle of smallest area
    with its two neighbours.  Related to PIP but uses area rather than
    perpendicular distance as the importance metric.

    Reference: Visvalingam & Whyatt, The Cartographic Journal, 1993.
    """
    _METHOD_NAME = "VisvalingamWhyatt"


# ---------------------------------------------------------------------------
# SZ3 — optional, requires compiled native library
# ---------------------------------------------------------------------------

def _sz3_lib_path() -> str:
    base = os.path.join(os.path.dirname(__file__), "..", "external", "SZ3", "install", "lib")
    env  = os.environ.get("SZ3_LIB_PATH")
    if env:
        return env
    suffix = ".dylib" if platform.system() == "Darwin" else ".so"
    for name in [f"libSZ3c{suffix}", "libSZ3c.so.3", "libSZ3c.so"]:
        full = os.path.join(base, name)
        if os.path.exists(full):
            return full
    return os.path.join(base, f"libSZ3c{suffix}")


class SZ3Compressor:
    """
    Error-bounded lossy compressor using SZ3 (prediction + quantization).
    *Optional* — requires the SZ3 shared library built from source.

    Build instructions: see external/SZ3/README.md
    Set SZ3_LIB_PATH env var to override the default search path.

    The underlying SZ ctypes wrapper is cached at class level so the shared
    library is loaded only once regardless of how many SZ3Compressor instances
    are created (one per dataset per CR in the experiment loop).
    """

    _sz_cache: dict = {}       # lib_path → SZ instance
    _sz_lock = __import__("threading").Lock()

    def __init__(self, error_bound: float = 1e-3):
        pysz_path = os.path.join(os.path.dirname(__file__), "..", "external", "SZ3", "tools", "pysz")
        if pysz_path not in sys.path:
            sys.path.insert(0, pysz_path)
        try:
            from pysz import SZ
            self._SZ_cls = SZ
        except ImportError as exc:
            raise RuntimeError(
                "pysz not found.  Build SZ3 and ensure external/SZ3/tools/pysz is accessible.\n"
                "See external/SZ3/README.md for build instructions."
            ) from exc

        lib = _sz3_lib_path()
        if not os.path.exists(lib):
            raise RuntimeError(
                f"SZ3 library not found at {lib}.\n"
                "Build SZ3 (see external/SZ3/README.md) or set SZ3_LIB_PATH."
            )

        with self._sz_lock:
            if lib not in self._sz_cache:
                self._sz_cache[lib] = SZ(lib)
        self.sz          = self._sz_cache[lib]
        self.error_bound = float(error_bound)

    def compress(self, data: np.ndarray):
        data_cmpr, _ = self.sz.compress(data, 0, self.error_bound, 0, 0)
        return data_cmpr

    def decompress(self, data_cmpr, original_shape, original_dtype):
        return self.sz.decompress(data_cmpr, original_shape, original_dtype)


# ---------------------------------------------------------------------------
# Gorilla — pure-Python XOR streaming compressor (Gorilla/SERF family)
# ---------------------------------------------------------------------------

class GorillaCompressor:
    """
    Pure-Python XOR-based streaming float compressor in the Gorilla/SERF family.

    Algorithm
    ---------
    1. Values are first rounded to the nearest multiple of *error_bound*,
       reducing entropy in smooth signals.
    2. The rounded float64 array is reinterpreted as uint64 and XOR-encoded
       (each element XOR'd with its predecessor), exploiting the fact that
       consecutive similar float values share many high-order bits.
    3. The resulting XOR delta stream is stored as raw bytes; a downstream
       entropy coder (zstd, used for CR measurement) exploits the zero-byte
       runs that dominate smooth signals.

    This captures the same compression principle as SERF/Gorilla without
    requiring any native extension.  It is always available.

    error_bound: absolute error bound on the normalised [0,1] signal.
    """

    def __init__(self, error_bound: float = 1e-3):
        self.error_bound = float(error_bound)

    def _compress_1d(self, arr: np.ndarray) -> bytes:
        arr = np.asarray(arr, dtype=np.float64)
        n   = len(arr)

        # Round to error_bound precision
        if self.error_bound > 0:
            quantized = np.round(arr / self.error_bound) * self.error_bound
        else:
            quantized = arr.copy()

        # Reinterpret as uint64 and XOR-encode
        bits        = quantized.view(np.uint64).copy()
        xors        = np.empty_like(bits)
        xors[0]     = bits[0]
        xors[1:]    = bits[1:] ^ bits[:-1]

        return struct.pack("<I", n) + xors.tobytes()

    def _decompress_1d(self, blob: bytes, shape, dtype) -> np.ndarray:
        n,     = struct.unpack_from("<I", blob, 0)
        xors   = np.frombuffer(blob, dtype=np.uint64, count=n, offset=4).copy()

        # Cumulative XOR to recover bit patterns
        np.bitwise_xor.accumulate(xors, out=xors)
        result = xors.view(np.float64)
        return result[:shape[0]].astype(dtype, copy=False)

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        original_shape = np.array(list(data.shape), dtype=np.int32)
        blobs = _compress_channels(self._compress_1d, data)
        shape_header = struct.pack("<I", len(data.shape)) + original_shape.tobytes()
        return shape_header + _pack_channels(blobs)

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        ndim,  = struct.unpack_from("<I", payload, 0)
        stored_shape = tuple(
            int(x) for x in np.frombuffer(payload, dtype=np.int32, count=ndim, offset=4)
        )
        offset = 4 + ndim * 4
        blobs  = _unpack_channels(payload[offset:])
        shape  = original_shape if original_shape is not None else stored_shape
        dtype  = original_dtype if original_dtype is not None else np.float64
        return _decompress_channels(self._decompress_1d, blobs, shape, dtype)


# ---------------------------------------------------------------------------
# SERF — optional, requires compiled C++ extension
# ---------------------------------------------------------------------------

class PySerfCompressor:
    """
    Streaming XOR-based float compressor (Gorilla/SERF family).
    *Optional* — requires the pyserf C++ extension built from source.

    Build instructions: see external/Serf/README.md
    Set SERF_PYWRAPPER_PATH env var to override the default build path.
    """

    _FRAME_MAGIC      = b"SRF1"
    _FRAME_LEN_STRUCT = struct.Struct("<I")

    def __init__(self, error_bound: float = 1e-3):
        serf_path = os.environ.get(
            "SERF_PYWRAPPER_PATH",
            os.path.join(os.path.dirname(__file__), "..", "external", "Serf", "build", "pywrapper"),
        )
        if serf_path not in sys.path:
            sys.path.insert(0, serf_path)
        try:
            from pyserf import ArrayOfBytes, PySerfXORCompressor, PySerfXORDecompressor
            self._array_of_bytes_cls = ArrayOfBytes
            self._compressor_cls     = PySerfXORCompressor
            self._decompressor_cls   = PySerfXORDecompressor
        except ImportError as exc:
            raise RuntimeError(
                "pyserf not found.  Build the SERF C++ extension.\n"
                "See external/Serf/README.md or set SERF_PYWRAPPER_PATH."
            ) from exc

        self.error_bound = float(error_bound)
        self.window_size = 1000
        self.block_size  = 500
        self.adjust      = 0

    def _pack_to_bytes(self, pack) -> bytes:
        if isinstance(pack, (bytes, bytearray, memoryview)):
            return bytes(pack)
        if hasattr(pack, "__getstate__"):
            state = pack.__getstate__()
            if isinstance(state, tuple) and state:
                return bytes(state[0])
        return bytes(pack)

    def _compress_1d(self, arr: np.ndarray) -> bytes:
        values     = np.asarray(arr, dtype=np.float64).ravel()
        compressor = self._compressor_cls(self.window_size, self.error_bound, self.adjust)
        framed     = bytearray(self._FRAME_MAGIC)
        for start in range(0, values.size, self.block_size):
            for val in values[start : start + self.block_size]:
                compressor.add_value(float(val))
            compressor.close()
            pack_bytes = self._pack_to_bytes(compressor.get())
            framed.extend(self._FRAME_LEN_STRUCT.pack(len(pack_bytes)))
            framed.extend(pack_bytes)
        return bytes(framed)

    def _decompress_1d(self, blob: bytes, shape, dtype) -> np.ndarray:
        payload      = bytes(blob)
        decompressor = self._decompressor_cls(self.adjust)
        if payload.startswith(self._FRAME_MAGIC):
            offset, values = len(self._FRAME_MAGIC), []
            while offset < len(payload):
                (block_len,) = self._FRAME_LEN_STRUCT.unpack_from(payload, offset)
                offset += self._FRAME_LEN_STRUCT.size
                pack = self._array_of_bytes_cls(list(payload[offset : offset + block_len]))
                values.extend(decompressor.decompress(pack))
                offset += block_len
            data_dec = np.asarray(values, dtype=np.float64)
        else:
            pack     = self._array_of_bytes_cls(list(payload))
            data_dec = np.asarray(decompressor.decompress(pack), dtype=np.float64)
        return data_dec[: shape[0]].astype(dtype, copy=False)

    def compress(self, data: np.ndarray) -> bytes:
        data = np.asarray(data, dtype=np.float64)
        original_shape = np.array(list(data.shape), dtype=np.int32)
        blobs = _compress_channels(self._compress_1d, data)
        shape_header = struct.pack("<I", len(data.shape)) + original_shape.tobytes()
        return shape_header + _pack_channels(blobs)

    def decompress(self, data_cmp: bytes, original_shape=None, original_dtype=None):
        payload = bytes(data_cmp)
        ndim, = struct.unpack_from("<I", payload, 0)
        stored_shape = tuple(
            int(x) for x in np.frombuffer(payload, dtype=np.int32, count=ndim, offset=4)
        )
        offset = 4 + ndim * 4
        blobs  = _unpack_channels(payload[offset:])
        shape  = original_shape if original_shape is not None else stored_shape
        dtype  = original_dtype if original_dtype is not None else np.float64
        return _decompress_channels(self._decompress_1d, blobs, shape, dtype)


# ---------------------------------------------------------------------------
# Availability probe
# ---------------------------------------------------------------------------

def _probe(cls, name: str) -> bool:
    """Return True if *cls* can be instantiated (native libs present)."""
    try:
        cls(error_bound=1e-3)
        return True
    except RuntimeError:
        return False
    except Exception:
        return False


# Map name → (class, always_available)
# always_available=True  → included unconditionally
# always_available=None  → probed at import time (native dep may be absent)
_COMPRESSOR_REGISTRY: dict[str, tuple[type, bool]] = {
    # ── Passthrough baseline ────────────────────────────────────────────────
    "NONE":    (NoneCompressor,              True),

    # ── Perceptual / shape-preserving ───────────────────────────────────────
    # Retain "important" points and reconstruct removed ones by interpolation.
    "PIP":     (PIPCompressor,               True),   # Perpendicular-distance importance + ACF
    "VW":      (VisvalingamWhyattCompressor, None),   # Triangle-area importance (TerseTS)

    # ── Uniform quantisation ────────────────────────────────────────────────
    # Reduce the effective bit depth of each sample.
    "QUANT":   (QuantizationCompressor,      True),   # Pure-Python bit-depth reduction

    # ── XOR / streaming ─────────────────────────────────────────────────────
    # Exploit temporal correlation by XOR-encoding consecutive float bit patterns.
    "GORILLA": (GorillaCompressor,           True),   # Pure-Python (Gorilla/SERF family)
    "SERF":    (PySerfCompressor,            None),   # Native C++ (build from external/Serf)

    # ── Transform / frequency ───────────────────────────────────────────────
    # Decompose into a frequency basis and discard small coefficients.
    "DWT":     (DWTCompressor,               None),   # Daubechies-4 wavelet (PyWavelets)

    # ── Piecewise linear (PLR) ──────────────────────────────────────────────
    # Approximate each segment by a line within an error bound.
    # All PLR methods via TerseTS (Xenophontos & Palpanas, EDBT 2026).
    "MP":      (MixPieceCompressor,          None),   # MixPiece    — joint multi-seg fitting (VLDB'23)
    "SIMPIE":  (SimPieceCompressor,          None),   # SimPiece    — simplified joint fitting (VLDB'22)
    "SWING":   (SwingFilterCompressor,       None),   # SwingFilter — classic error-cone PLR  (VLDB'09)
    "SLIDE":   (SlideFilterCompressor,       None),   # SlideFilter — sliding-window PLR      (VLDB'09)

    # ── Error-bounded (scientific computing) ────────────────────────────────
    # Bound the pointwise absolute error to ε while maximising compression.
    "ZFP":     (ZFPCompressor,               None),   # ZFP 1-D per channel (pip install zfpy)
    "ZFP2D":   (ZFP2DCompressor,             None),   # ZFP native 2-D block transform — multivariate-native
    "SZ3":     (SZ3Compressor,               None),   # SZ3 prediction+quant — multivariate-native (native build)

    # ── Transform / frequency (multivariate-native) ─────────────────────────
    "DWT2D":   (DWT2DCompressor,             None),   # 2-D DWT across time+channel axes (PyWavelets)
}


def get_available_compressors() -> dict[str, type]:
    """
    Return {name: class} for every compressor whose dependencies are present.
    Called once at import time by configs.py to build MethodType.
    """
    available = {}
    for name, (cls, always) in _COMPRESSOR_REGISTRY.items():
        if always is True or _probe(cls, name):
            available[name] = cls
        else:
            print(f"[compression] {name} unavailable — skipping (missing native library or package)")
    return available
