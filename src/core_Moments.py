from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
from scipy.stats import lognorm
from numpy.linalg import solve
from numba import njit

@njit(fastmath=True)
def f_X(x, beta, a):
    return 0.5 * beta * np.exp(-beta * np.abs(x + a))

@njit(fastmath=True)
def F_X(x, beta, a):
    z = x + a
    if z < 0.0:
        return 0.5 * np.exp(beta * z)
    else:
        return 1.0 - 0.5 * np.exp(-beta * z)

# Nystrom discretization for conditional moments
"""def Nystrom(z_max, N, kernel_pdf):
    z_grid = np.linspace(0.0, z_max, N + 1)
    h = z_grid[1] - z_grid[0]

    # trapezoidal weights on [0, z_max]
    w = np.ones_like(z_grid, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5

    # delta_ij = b_j - z_i
    delta = z_grid[None, :] - z_grid[:, None]

    # K_ij = h * w_j * f_X(b_j - z_i)
    K = h * kernel_pdf(delta) * w[None, :]
    I = np.eye(z_grid.size)

    return I, K, z_grid, w, h"""


@njit(cache=True, fastmath=True)
def kernel_pdf(delta, beta, a):
    return 0.5 * beta * np.exp(-beta * abs(delta + a))

@njit(cache=True, fastmath=True)
def Nystrom(z_max, N, beta, a):
    n = N + 1

    # grid
    z_grid = np.linspace(0.0, z_max, n)
    h = z_grid[1] - z_grid[0]

    # trapezoidal weights
    w = np.ones(n, dtype=np.float64)
    w[0] = 0.5
    w[n - 1] = 0.5

    # identity
    I = np.eye(n, dtype=np.float64)

    # build K directly: K_ij = h * w_j * f_X(z_j - z_i)
    K = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        zi = z_grid[i]
        for j in range(n):
            delta = z_grid[j] - zi
            K[i, j] = h * w[j] * kernel_pdf(delta, beta, a)

    return I, K, z_grid, w, h

@njit
def GM_mix(m_vec, w, h, pdf_at_grid, Ppos):
    total = 0.0
    for i in range(len(m_vec)):
        total += w[i] * m_vec[i] * pdf_at_grid[i]
    return (h * total) / Ppos