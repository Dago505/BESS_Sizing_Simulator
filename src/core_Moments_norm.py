from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
from scipy.stats import lognorm
from numpy.linalg import solve
from numba import njit

@njit(cache=True, fastmath=True)
def kernel_pdf(delta_tilde, a_tilde):
    return 0.5 * np.exp(-abs(delta_tilde + a_tilde))

@njit(cache=True, fastmath=True)
def Nystrom(z_max_tilde, N, a_tilde):
    n = N + 1

    # grid
    z_grid_tilde = np.linspace(0.0, z_max_tilde, n)
    h_tilde = z_grid_tilde[1] - z_grid_tilde[0]

    # trapezoidal weights
    w = np.ones(n, dtype=np.float64)
    w[0] = 0.5
    w[n - 1] = 0.5

    # identity
    I = np.eye(n, dtype=np.float64)

    # build K directly: K_ij = h * w_j * f_X(z_j - z_i)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        zi = z_grid_tilde[i]
        for j in range(n):
            delta_tilde = z_grid_tilde[j] - zi
            K[i, j] = h_tilde * w[j] * kernel_pdf(delta_tilde, a_tilde)
    
    return I, K, z_grid_tilde, w, h_tilde

@njit
def GM_mix(m_vec, w, h, pdf_at_grid, Ppos):
    total = 0.0
    for i in range(len(m_vec)):
        total += w[i] * m_vec[i] * pdf_at_grid[i]
    return (h * total) / Ppos