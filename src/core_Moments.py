from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
from scipy.stats import lognorm
from numpy.linalg import solve

def f_X(x, beta, a):        # PDF of the effective increment x
    x = np.asarray(x)
    return 0.5 * beta * np.exp(-beta * np.abs(x + a))

def F_X(x, beta, a):        # CDF of the effective increment x
    x = np.asarray(x)
    z = x + a
    out = np.empty_like(z, dtype=float)
    m = (z < 0)
    out[m]  = 0.5 * np.exp(beta * z[m])
    out[~m] = 1.0 - 0.5 * np.exp(-beta * z[~m])
    return out

# Nystrom discretization for conditional moments
def Nystrom(z_max, N, kernel_pdf):
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

    return I, K, z_grid, w, h

def GM_mix(m_vec, w, h, pdf_at_grid, Ppos):
    return (h * np.sum(w * m_vec * pdf_at_grid)) / Ppos