"""
PIP-based Time Series Simplification with ACF Preservation

This module provides a pure Python implementation of the Perceptually Important Points (PIP)
algorithm for time series simplification while preserving autocorrelation function (ACF) properties.
"""

import numpy as np
import math
from typing import Optional, List
from dataclasses import dataclass


# ============================================================================
# PIP Heap Implementation
# ============================================================================

@dataclass
class PIPNode:
    """Node in the PIP (Perceptually Important Points) linked list and heap"""
    left: Optional['PIPNode'] = None
    right: Optional['PIPNode'] = None
    parent: Optional['PIPHeap'] = None
    ts: int = 0  # timestamp/index
    index: int = 0  # index in heap array
    order: int = 0  # insertion order for tie-breaking
    value: float = 0.0
    cache: float = np.inf  # cached vertical distance


class PIPHeap:
    """Heap for managing perceptually important points"""
    
    def __init__(self, size: int):
        self.head: Optional[PIPNode] = None
        self.tail: Optional[PIPNode] = None
        self.values: List[PIPNode] = []
        self.size: int = 0
        self.m_size: int = size
        self.global_order: int = 0
        self._init_heap(size)
    
    def _init_heap(self, size: int):
        """Initialize the heap with pre-allocated nodes"""
        self.values = []
        for i in range(size):
            node = PIPNode()
            node.index = i
            node.parent = self
            node.cache = np.inf
            self.values.append(node)
    
    def add(self, ts: int, value: float):
        """Add a new point to the heap"""
        node = self._acquire_item(ts, value)
        self.tail = self._put_after(node, self.tail)
        if self.head is None:
            self.head = self.tail
    
    def _acquire_item(self, ts: int, value: float) -> PIPNode:
        """Get a node from the pool and initialize it"""
        node = self.values[self.size]
        node.ts = ts
        node.value = value
        self.size += 1
        self.global_order += 1
        node.order = self.global_order
        return node
    
    def remove_at(self, index: int) -> PIPNode:
        """Remove and return the node at the given heap index"""
        self.size -= 1
        self._swap(index, self.size)
        self._bubble_down(index)
        
        node = self.values[self.size]
        return self._recycle(node)
    
    def _recycle(self, node: PIPNode) -> PIPNode:
        """Remove node from linked list and return it"""
        if node.left is None:
            self.head = node.right
        else:
            node.left.right = node.right
            self._update_cache(node.left)
        
        if node.right is None:
            self.tail = node.left
        else:
            node.right.left = node.left
            self._update_cache(node.right)
        
        return self._clear(node)
    
    def _clear(self, node: PIPNode) -> PIPNode:
        """Clear node connections and return a copy"""
        returned_node = PIPNode(
            left=node.left,
            right=node.right,
            value=node.value,
            ts=node.ts
        )
        
        node.left = None
        node.right = None
        node.parent = None
        node.cache = np.inf
        
        return returned_node
    
    def _put_after(self, node: PIPNode, tail: Optional[PIPNode]) -> PIPNode:
        """Insert node after tail in linked list"""
        if tail is not None:
            tail.right = node
            self._update_cache(tail)
        node.left = tail
        self._update_cache(node)
        return node
    
    def _update_cache(self, node: PIPNode):
        """Update the cached vertical distance for a node"""
        if node.left is not None and node.right is not None:
            node.cache = self._vertical_distance(node.left, node, node.right)
        else:
            node.cache = np.inf
        
        self._notify_change(node.index)
    
    @staticmethod
    def _vertical_distance(left: PIPNode, node: PIPNode, right: PIPNode) -> float:
        """Calculate vertical distance from point to line segment"""
        EPSILON = 1e-6
        
        a_x, b_x, c_x = left.ts, node.ts, right.ts
        a_y, b_y, c_y = left.value, node.value, right.value
        
        if abs(a_x - b_x) < EPSILON or abs(b_x - c_x) < EPSILON:
            return 0.0
        elif (c_x - a_x) == 0:
            return np.inf
        else:
            return abs(a_y + (c_y - a_y) * (b_x - a_x) / (c_x - a_x) - b_y)
    
    def _notify_change(self, index: int) -> int:
        """Notify heap of change and restore heap property"""
        return self._bubble_down(self._bubble_up(index))
    
    def _bubble_up(self, n: int) -> int:
        """Move node up the heap until heap property is satisfied"""
        while n != 0 and self._less(n, (n - 1) // 2):
            n = self._swap(n, (n - 1) // 2)
        return n
    
    def _bubble_down(self, n: int) -> int:
        """Move node down the heap until heap property is satisfied"""
        k = self._min(n, n * 2 + 1, n * 2 + 2)
        while k != n and k < self.size:
            n = self._swap(n, k)
            k = self._min(n, n * 2 + 1, n * 2 + 2)
        return n
    
    def _swap(self, i: int, j: int) -> int:
        """Swap two nodes in the heap"""
        self.values[i].index, self.values[j].index = j, i
        self.values[i], self.values[j] = self.values[j], self.values[i]
        return j
    
    def _min(self, i: int, j: int, k: int) -> int:
        """Return index of minimum among three nodes"""
        if k != -1:
            return self._min(i, self._min(j, k, -1), -1)
        else:
            return i if self._less(i, j) else j
    
    def _less(self, i: int, j: int) -> bool:
        """Check if node i is less than node j"""
        return (i < self.size) and (j >= self.size or self._i_smaller_than_j(i, j))
    
    def _i_smaller_than_j(self, i: int, j: int) -> bool:
        """Compare two nodes by cache value, then order"""
        if self.values[i].cache != self.values[j].cache:
            return self.values[i].cache < self.values[j].cache
        else:
            return self.values[i].order < self.values[j].order
    
    def deinit(self):
        """Cleanup the heap (Python handles memory automatically)"""
        self.values.clear()
        self.head = None
        self.tail = None


# ============================================================================
# Math Utilities
# ============================================================================

def cumsum_cumsum(x: np.ndarray) -> tuple:
    """Calculate cumulative sum and cumulative sum of squares"""
    x_cum_sum = np.cumsum(x)
    power_cum_sum = np.cumsum(x ** 2)
    return x_cum_sum, power_cum_sum


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate dot product of two arrays"""
    return np.dot(x, y)


# ============================================================================
# ACF Aggregation Implementation
# ============================================================================

class AcfAgg:
    """Autocorrelation Function aggregation for incremental updates"""
    
    def __init__(self, nlags: int):
        self.nlags: int = nlags
        self.n: int = 0
        self.sxy: np.ndarray = np.zeros(nlags, dtype=np.float64)
        self.xs: np.ndarray = np.zeros(nlags, dtype=np.float64)
        self.ys: np.ndarray = np.zeros(nlags, dtype=np.float64)
        self.xss: np.ndarray = np.zeros(nlags, dtype=np.float64)
        self.yss: np.ndarray = np.zeros(nlags, dtype=np.float64)
    
    def initialize(self):
        """Initialize the aggregation arrays"""
        # Already initialized in __init__
        pass
    
    def fit(self, x: np.ndarray):
        """Fit the ACF aggregates to the data"""
        n = len(x)
        x_cum_sum, power_cum_sum = cumsum_cumsum(x)
        self.n = n
        
        for lag in range(self.nlags):
            self.xs[lag] = x_cum_sum[n - lag - 2]
            self.ys[lag] = x_cum_sum[n - 1] - x_cum_sum[lag]
            self.xss[lag] = power_cum_sum[n - lag - 2]
            self.yss[lag] = power_cum_sum[n - 1] - power_cum_sum[lag]
            self.sxy[lag] = dot_product(x[:n - lag - 1], x[lag + 1:])
    
    def get_acf(self) -> np.ndarray:
        """Calculate and return the autocorrelation function"""
        result = np.zeros(self.nlags, dtype=np.float64)
        n = self.n
        
        for lag in range(self.nlags):
            n -= 1
            numerator = n * self.sxy[lag] - self.xs[lag] * self.ys[lag]
            denominator = math.sqrt(
                (n * self.xss[lag] - self.xs[lag] * self.xs[lag]) *
                (n * self.yss[lag] - self.ys[lag] * self.ys[lag])
            )
            result[lag] = numerator / denominator
        
        return result
    
    def update(self, x: np.ndarray, x_a: float, index: int):
        """Update aggregates after changing x[index] to x_a"""
        delta = x_a - x[index]
        delta_ss = delta * (2 * x[index] + delta)
        
        if delta != 0:
            if index <= self.nlags or index >= self.n - self.nlags:
                self._update_inside_lags(x, delta, delta_ss, index)
            else:
                self._update_outside_lags(x, delta, delta_ss, index)
            
            x[index] = x_a
    
    def _update_inside_lags(self, x: np.ndarray, delta: float, delta_ss: float, index: int):
        """Update when index is within nlags of boundaries"""
        n = self.n - 1
        for lag in range(self.nlags):
            if index >= lag + 1:
                self.ys[lag] += delta
                self.yss[lag] += delta_ss
                self.sxy[lag] += delta * x[index - lag - 1]
            if index < n - lag:
                self.xs[lag] += delta
                self.xss[lag] += delta_ss
                self.sxy[lag] += delta * x[index + lag + 1]
    
    def _update_outside_lags(self, x: np.ndarray, delta: float, delta_ss: float, index: int):
        """Update when index is outside nlags boundary region"""
        for lag in range(self.nlags):
            self.ys[lag] += delta
            self.yss[lag] += delta_ss
            self.xs[lag] += delta
            self.xss[lag] += delta_ss
            self.sxy[lag] += delta * (x[index + lag + 1] + x[index - lag - 1])
    
    def interpolate_update(self, x: np.ndarray, start: int, end: int):
        """Update aggregates when interpolating between start and end"""
        if start <= self.nlags or end >= self.n - self.nlags:
            self._interpolate_update_inside_lags(x, start, end)
        else:
            self._interpolate_update_outside_lags(x, start, end)
    
    def _interpolate_update_outside_lags(self, x: np.ndarray, start: int, end: int):
        """Interpolate update outside lag boundaries"""
        num_deltas = end - start - 1
        slope = (x[end] - x[start]) / (end - start)
        
        deltas = np.zeros(num_deltas, dtype=np.float64)
        x_as = np.zeros(num_deltas, dtype=np.float64)
        
        i = 0
        for index in range(start + 1, end):
            x_as[i] = slope * (index - start) + x[start]
            delta = x_as[i] - x[index]
            deltas[i] = delta
            
            delta_ss = delta * (delta + 2 * x[index])
            for lag in range(self.nlags):
                self.xs[lag] += delta
                self.ys[lag] += delta
                self.yss[lag] += delta_ss
                self.xss[lag] += delta_ss
                self.sxy[lag] += delta * (x[index - lag - 1] + x[index + lag + 1])
            
            i += 1
        
        # Update cross-products between interpolated points
        num_lags = min(num_deltas, self.nlags)
        for lag in range(num_lags):
            for i in range(num_deltas - lag - 1):
                self.sxy[lag] += deltas[i] * deltas[i + lag + 1]
        
        # Update x with interpolated values
        x[start + 1:end] = x_as
    
    def _interpolate_update_inside_lags(self, x: np.ndarray, start: int, end: int):
        """Interpolate update inside lag boundaries"""
        num_deltas = end - start - 1
        slope = (x[end] - x[start]) / (end - start)
        
        deltas = np.zeros(num_deltas, dtype=np.float64)
        x_as = np.zeros(num_deltas, dtype=np.float64)
        
        i = 0
        n = self.n - 1
        for index in range(start + 1, end):
            x_as[i] = slope * (index - start) + x[start]
            delta = x_as[i] - x[index]
            deltas[i] = delta
            
            delta_ss = delta * (delta + 2 * x[index])
            for lag in range(self.nlags):
                if index >= lag + 1:
                    self.ys[lag] += delta
                    self.yss[lag] += delta_ss
                    self.sxy[lag] += delta * x[index - lag - 1]
                if index < n - lag:
                    self.xs[lag] += delta
                    self.xss[lag] += delta_ss
                    self.sxy[lag] += delta * x[index + lag + 1]
            
            i += 1
        
        # Update cross-products between interpolated points
        num_lags = min(num_deltas, self.nlags)
        for lag in range(num_lags):
            for i in range(num_deltas - lag - 1):
                self.sxy[lag] += deltas[i] * deltas[i + lag + 1]
        
        # Update x with interpolated values
        x[start + 1:end] = x_as
    
    def release_memory(self):
        """Release memory (Python handles this automatically)"""
        pass


# ============================================================================
# Main Simplification Function
# ============================================================================

def simplify_by_pip(y: np.ndarray, nlags: int, acf_threshold: float) -> np.ndarray:
    """
    Simplify time series using PIP (Perceptually Important Points) while 
    preserving autocorrelation function (ACF) properties.
    
    This algorithm iteratively removes the least perceptually important points
    from the time series until removing additional points would cause the ACF
    to deviate beyond a specified threshold from the original ACF.
    
    Parameters
    ----------
    y : np.ndarray
        Input time series data (1D array). This array will be modified in place
        during the simplification process (interpolated values replace removed points).
    nlags : int
        Number of lags to use for ACF calculation. Controls how many lag values
        are considered when measuring autocorrelation preservation.
    acf_threshold : float
        Maximum allowable average absolute deviation of the simplified series' ACF
        from the original ACF. Higher values allow more aggressive simplification.
        Typical values range from 0.01 to 0.1.
        
    Returns
    -------
    np.ndarray
        Boolean mask array of shape (N,) where True indicates the point was kept
        and False indicates it was removed during simplification.
        
    Notes
    -----
    - The algorithm uses PIP (Perceptually Important Points) to measure the 
      importance of each point based on its perpendicular distance from the 
      line connecting its neighbors.
    - Points are removed in order of increasing importance until the ACF 
      constraint is violated.
    - The input array `y` is modified in place - removed points are replaced
      with linearly interpolated values.
    - Time complexity: O(N log N) where N is the length of the series.
    
    Examples
    --------
    >>> import numpy as np
    >>> # Generate a time series with autocorrelation
    >>> np.random.seed(42)
    >>> y = np.cumsum(np.random.randn(1000))
    >>> 
    >>> # Simplify while preserving ACF structure
    >>> mask = simplify_by_pip(y, nlags=10, acf_threshold=0.05)
    >>> 
    >>> # Get the indices and values of retained points
    >>> kept_indices = np.where(mask)[0]
    >>> kept_values = y[mask]
    >>> 
    >>> print(f"Retained {mask.sum()} of {len(y)} points ({100*mask.mean():.1f}%)")
    """
    N = len(y)
    
    # Initialize data structures
    acf_agg = AcfAgg(nlags)
    pip_importance_heap = PIPHeap(N)
    non_removed_points = np.ones(N, dtype=bool)
    
    # Initialize ACF aggregates and compute baseline ACF
    acf_agg.initialize()
    acf_agg.fit(y)
    raw_acf = acf_agg.get_acf()
    
    # Add all points to the PIP heap
    for i in range(N):
        pip_importance_heap.add(i, y[i])
    
    # Main loop: iteratively remove least important points
    while pip_importance_heap.values[0].cache < np.inf:
        # Remove the least important point
        min_node = pip_importance_heap.remove_at(0)
        start = min_node.left.ts
        end = min_node.right.ts
        
        # Update ACF aggregates with interpolated or single point update
        if start + 2 < end:
            # Multiple points between neighbors - use interpolation
            acf_agg.interpolate_update(y, start, end)
        else:
            # Single point - direct update
            x_a = (y[end] - y[start]) / (end - start) + y[start]
            acf_agg.update(y, x_a, start + 1)
        
        # Calculate ACF error (average absolute deviation across all lags)
        ace = 0.0
        n = len(y)
        
        for lag in range(acf_agg.nlags):
            n -= 1
            
            # Compute current ACF for this lag
            numerator = n * acf_agg.sxy[lag] - acf_agg.xs[lag] * acf_agg.ys[lag]
            denominator = math.sqrt(
                (n * acf_agg.xss[lag] - acf_agg.xs[lag] * acf_agg.xs[lag]) *
                (n * acf_agg.yss[lag] - acf_agg.ys[lag] * acf_agg.ys[lag])
            )
            c_acf = numerator / denominator
            
            # Accumulate absolute error from original ACF
            ace += abs(raw_acf[lag] - c_acf)
        
        # Average error across all lags
        ace /= acf_agg.nlags
        
        # Stop if ACF deviation exceeds threshold
        if ace >= acf_threshold:
            break
        
        # Mark point as removed
        non_removed_points[min_node.ts] = False
    
    # Cleanup
    pip_importance_heap.deinit()
    acf_agg.release_memory()
    
    return non_removed_points


if __name__ == "__main__":
    # Example usage and demonstration
    import matplotlib.pyplot as plt
    
    # Generate example time series
    np.random.seed(42)
    t = np.linspace(0, 4*np.pi, 500)
    y = np.sin(t) + 0.5 * np.sin(3*t) + 0.2 * np.random.randn(len(t))
    y_original = y.copy()
    
    # Simplify the series
    mask = simplify_by_pip(y, nlags=20, acf_threshold=0.05)
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot original and simplified series
    ax1.plot(t, y_original, 'b-', alpha=0.3, label='Original', linewidth=1)
    ax1.plot(t[mask], y_original[mask], 'ro', markersize=3, label='Kept Points')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Value')
    ax1.set_title(f'PIP Simplification (Kept {mask.sum()}/{len(y)} points = {100*mask.mean():.1f}%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot ACF comparison
    from numpy import correlate
    
    def compute_acf(x, nlags):
        x = x - x.mean()
        c0 = np.dot(x, x) / len(x)
        acf = [correlate(x[:-i], x[i:])[0] / len(x) / c0 if i > 0 else 1.0 
               for i in range(nlags)]
        return np.array(acf)
    
    original_acf = compute_acf(y_original, 50)
    simplified_acf = compute_acf(y_original[mask], 50)
    
    ax2.plot(original_acf, 'b-', label='Original ACF', linewidth=2)
    ax2.plot(simplified_acf, 'r--', label='Simplified ACF', linewidth=2)
    ax2.set_xlabel('Lag')
    ax2.set_ylabel('Autocorrelation')
    ax2.set_title('ACF Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pip_simplification_example.png', dpi=150, bbox_inches='tight')
    print(f"Saved example plot to pip_simplification_example.png")
    print(f"Compression ratio: {100 * (1 - mask.mean()):.1f}%")
