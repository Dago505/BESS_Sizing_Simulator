from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import math
from numba import njit
from typing import Callable, Tuple, Dict, Any



# ===========================================
# --- CDF AND P99 HELPERS --- 
# ===========================================

# --- CDF CALCULATOR ---
@njit
def cdf_trapz_calculator(b_grid, g, p0=None):
    n = b_grid.shape[0]

    # continuous part (integral of g)
    G_cont = np.zeros(n, dtype=np.float64)

    if n > 1:
        G_cont[1] = 0.5 * (g[0] + g[1]) * (b_grid[1] - b_grid[0])
        for i in range(2, n):
            G_cont[i] = G_cont[i-1] + 0.5 * (g[i-1] + g[i]) * (b_grid[i] - b_grid[i-1])

    if p0 is not None:
        G = np.empty(n, dtype=np.float64)
        for i in range(n):
            val = p0 + G_cont[i]
            if val < 0.0:
                val = 0.0
            elif val > 1.0:
                val = 1.0
            G[i] = val
        return G
    else:
        return G_cont



# --- quantile helper for numba optimization ---
@njit
def quantile(arr, q):
    sorted_arr = np.sort(arr)
    n = sorted_arr.shape[0]
    
    pos = q * (n - 1)
    lower = int(np.floor(pos))
    upper = int(np.ceil(pos))
    
    if lower == upper:
        return sorted_arr[lower]
    else:
        return sorted_arr[lower] + (sorted_arr[upper] - sorted_arr[lower]) * (pos - lower)
    

# --- histogram helper for numba optimization ---
@njit
def histogram(values: np.ndarray, n_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    vmin = np.min(values)
    vmax = np.max(values)

    # Create edges: length n_bins + 1
    if vmax == vmin:
        edges = linspace(vmin, vmax + 1.0, n_bins + 1)
    else:
        edges = linspace(vmin, vmax, n_bins + 1)

    counts = np.zeros(n_bins, dtype=np.float64)

    # Fill counts
    for v in values:
        if vmax == vmin:
            idx = 0
        else:
            t = (v - vmin) / (vmax - vmin)
            idx = int(t * n_bins)
            if idx == n_bins:
                idx = n_bins - 1
        counts[idx] += 1.0

    bin_width = edges[1] - edges[0]

    density = counts / (values.size * bin_width)

    # centers: length n_bins
    centers = 0.5 * (edges[:-1] + edges[1:])

    return density, centers

@njit
def histogram_cdf(values, n_bins, vmin=0.0):
    n = values.size

    # find vmax
    vmax = values[0]
    for i in range(1, n):
        if values[i] > vmax:
            vmax = values[i]

    if vmax <= vmin:
        vmax = vmin + 1.0

    dE = (vmax - vmin) / n_bins

    counts = np.zeros(n_bins)

    # fill histogram
    inv = 1.0 / (vmax - vmin)
    for i in range(n):
        v = values[i]
        if v < vmin:
            continue

        idx = int((v - vmin) * inv * n_bins)

        if idx >= n_bins:
            idx = n_bins - 1
        if idx < 0:
            continue

        counts[idx] += 1.0

    # build centers + CDF
    E_sim = np.empty(n_bins)
    F_sim = np.empty(n_bins)

    s = 0.0
    for i in range(n_bins):
        E_sim[i] = vmin + (i + 0.5) * dE

        # convert count → probability mass and accumulate
        s += counts[i] / n
        F_sim[i] = s

    # normalize (removes tiny numerical drift)
    if F_sim[n_bins - 1] > 0.0:
        inv_last = 1.0 / F_sim[n_bins - 1]
        for i in range(n_bins):
            F_sim[i] *= inv_last

    return E_sim, F_sim

def L1_norm(grid, f1, f2):
    delta = grid[1] - grid[0]
    g_diff = np.abs(f1 - f2)
    L1 = trapezoid_integral(g_diff, delta)
    return L1

# ---- cumsum helper for numba optimization ---
@njit
def cumsum(arr):
    n = len(arr)
    out = np.empty(n, dtype=arr.dtype)
    total = 0.0
    for i in range(n):
        total += arr[i]
        out[i] = total
    return out


# --- linspace function for numba optimization ---
import numpy as np
from numba import njit

@njit
def linspace(bmin, bmax, n):
    out = np.empty(n, dtype=np.float64)

    if n == 1:
        out[0] = bmin
        return out

    step = (bmax - bmin) / (n - 1)

    for i in range(n):
        out[i] = bmin + step * i

    return out


# --- boolean indexing numba friendly ---
@njit
def bool_mask(arr, mask):
    n = arr.shape[0]

    # 1) Count how many elements pass the mask
    count = 0
    for i in range(n):
        if mask[i]:
            count += 1

    # 2) Allocate output array
    out = np.empty(count, dtype=arr.dtype)

    # 3) Fill with selected values
    idx = 0
    for i in range(n):
        if mask[i]:
            out[idx] = arr[i]
            idx += 1

    return out


# --- zero counter for p0 calculation numba friendly ---
@njit
def zero_counts(x, threshold):
    # First pass: count how many matches
    count = 0
    for i in range(x.shape[0]):
        if x[i] < threshold:
            count += 1

    # Allocate output array
    out = np.empty(count, dtype=np.int64)

    # Second pass: store indices
    idx = 0
    for i in range(x.shape[0]):
        if x[i] < threshold:
            out[idx] = i
            idx += 1
    return out



# --- Concatenation function numba friendly ---

@njit
def concat(arrays):
    # ----- Step 1: compute total length -----
    total = 0
    for arr in arrays:
        total += arr.shape[0]

    # ----- Step 2: allocate output -----
    out = np.empty(total, dtype=arrays[0].dtype)

    # ----- Step 3: copy arrays sequentially -----
    idx = 0
    for arr in arrays:
        n = arr.shape[0]
        for i in range(n):
            out[idx] = arr[i]
            idx += 1

    return out


# --- Analytical quantile ---
@njit
def cdf_from_pdf(b, g):
    """Cumulative integral of g over b using trapezoids; starts at 0."""
    n = b.shape[0]
    diffs = b[1:] - b[:-1]
    traps = 0.5 * (g[1:] + g[:-1]) * diffs  # length n-1

    c = np.empty(n, dtype=np.float64)
    c[0] = 0.0
    c[1:] = cumsum(traps)  # your numba-friendly cumsum

    return c

@njit
def analytical_quantile(b, g, p0, q=0.99):
    # continuous CDF (without point mass at 0)
    c = cdf_from_pdf(b, g)
    total_cont = c[-1]

    # if quantile falls in the atom at 0
    if q <= p0:
        return 0.0

    # target CDF level in the continuous part
    target = q - p0

    # clip target to [0, total_cont]
    if target < 0.0:
        target = 0.0
    elif target > total_cont:
        target = total_cont

    # --- manual 1D interpolation: np.interp(target, c, b) ---

    n = c.shape[0]

    # if target is below first CDF value
    if target <= c[0]:
        return b[0]

    # search for interval [c[i-1], c[i]] containing target
    for i in range(1, n):
        if target <= c[i]:
            c0 = c[i-1]
            c1 = c[i]
            b0 = b[i-1]
            b1 = b[i]

            dc = c1 - c0
            if dc == 0.0:
                # flat CDF segment; fall back to midpoint in b
                return 0.5 * (b0 + b1)

            t = (target - c0) / dc
            return b0 + t * (b1 - b0)

    # if for numerical reasons we never found it, return last b
    return b[-1]

@njit
def Fitted_quantile(b, g, q=0.99):
    # continuous CDF from PDF (starts at 0)
    c = cdf_from_pdf(b, g)
    total = c[-1]

    # handle degenerate / empty mass
    if total <= 0.0:
        return b[0]

    # target CDF level (in same units as c)
    if q <= 0.0:
        target = 0.0
    elif q >= 1.0:
        target = total
    else:
        target = q * total

    n = c.shape[0]

    # below first
    if target <= c[0]:
        return b[0]

    # find interval and interpolate
    for i in range(1, n):
        if target <= c[i]:
            c0 = c[i - 1]
            c1 = c[i]
            b0 = b[i - 1]
            b1 = b[i]

            dc = c1 - c0
            if dc == 0.0:
                return 0.5 * (b0 + b1)

            t = (target - c0) / dc
            return b0 + t * (b1 - b0)

    return b[-1]



@njit
def trapezoid_integral(y, dx):
    s = 0.0
    n = y.size
    for k in range(n - 1):
        s += 0.5 * (y[k] + y[k+1]) * dx
    return s


#  --- Trapezoide integrator for numba optimization ---
@njit
def trapezoid(y, dx=1.0):
    n = y.shape[0]
    if n < 2:
        return 0.0

    s = 0.0
    for i in range(n - 1):
        s += 0.5 * (y[i] + y[i + 1]) * dx

    return s


@njit
def count_less_than(arr, thresh):
    c = 0
    for i in range(arr.size):
        if arr[i] < thresh:
            c += 1
    return c


# --- max/min helpers for numba optimization ---
@njit
def n_max(arr):
    n = len(arr)
    if n == 0:
        raise ValueError("Empty array has no maximum")
    m = arr[0]
    for i in range(1, n):
        if arr[i] > m:
            m = arr[i]
    return m

@njit
def n_min(arr):
    n = len(arr)
    if n == 0:
        raise ValueError("Empty array has no minimum")
    m = arr[0]
    for i in range(1, n):
        if arr[i] < m:
            m = arr[i]
    return m

@njit
def binom(n, r):
    if r < 0 or r > n:
        return 0.0
    # symmetry
    if r > n - r:
        r = n - r
    c = 1.0
    for i in range(r):
        c = c * (n - i) / (i + 1)
    return c

@njit
def factorial(n):
    if n < 0:
        return 0.0
    result = 1.0
    for i in range(2, n + 1):
        result *= i
    return result