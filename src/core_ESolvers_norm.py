from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
from scipy.stats import lognorm
from numpy.linalg import solve
from numba import njit
import math

import importlib
# --- Core code helpers ---
import src.core_helpers
importlib.reload(src.core_helpers)
from src.core_helpers import *

# --- Core Synthetic data ---
import src.core_SynData
importlib.reload(src.core_SynData)
from src.core_SynData import *

# --- Solvers ---
import src.core_Solvers
importlib.reload(src.core_Solvers)
from src.core_Solvers import *


# ========== ENERGY ALGORITHMIC SOLUTION: SIMULATION ============
@njit(cache=True, fastmath=True)
def Energy_ramps(Y_tilde, a_tilde, dt):
    N = Y_tilde.size
    B = np.zeros(N + 1, dtype=np.float64)
    energies = np.empty(N // 2 + 2, dtype=np.float64)
    n_ener = 0

    in_ramp = False
    ramp_energy_acc = 0.0

    for n in range(N):
        b_next = B[n] + (-Y_tilde[n] - a_tilde)
        if b_next < 0.0:
            b_next = 0.0
        B[n + 1] = b_next

        if not in_ramp:
            if b_next > 0.0:
                in_ramp = True
                ramp_energy_acc = b_next
        else:
            ramp_energy_acc += b_next
            if b_next == 0.0:
                energies[n_ener] = dt * ramp_energy_acc
                n_ener += 1
                in_ramp = False

    return energies[:n_ener]

@njit(cache=True, fastmath=True)
def E_algorithmic_solution(a_tilde: float, beta: float, Y: np.ndarray, dt:float, 
                         grid: int, q: float = 0.99) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """ Computes the pdf "f(e)" of the BESS simulation an returns the 99 percentile "p99". 

    Parameters
    ----------
    a: float
        Critical slope [dP/dt]
    Y: np.ndarray
        Array of the simulated Primary Power changes
    dt: float
        Time step between consecutive samples of the Primary power
    grid: int
        Resolution of the Energy pdf vector g(b)

    Returns
    -------
    energies: np.ndarray
        Simulation of the energy ramps
    e_sim: np.ndarray
        Energy grid based on the Energy simulation histogram bins
    f_sim: np.ndarray
        pdf of the energy simulation "f(e)".
    p99_sim: float
        99 percentil of the energy pdf "f(e)"
    """
    Y_tilde = beta * Y
    energies_tilde = Energy_ramps(Y_tilde, a_tilde, dt)

    # --- pdf histogram calculation ---
    e_sim, f_sim_tilde = histogram(energies_tilde, grid)

    # --- cdf histogram calculation ---
    E_sim, F_sim_tilde = histogram_cdf(energies_tilde, 110000, vmin=1e-5)
    
    #p0_sim = count_less_than(BESS, 1e-6) / BESS.size
    p99_sim_tilde = quantile(energies_tilde, q)

    E_pdf = (f_sim_tilde, e_sim)
    E_cdf = (F_sim_tilde, E_sim)
    return energies_tilde, E_pdf, E_cdf, p99_sim_tilde

# =====================================================================



# ======= ENERGY NUMERICAL SOLUTION: LOG-NORMAL FITTING ===============
@njit
def LogNormal_params(m1: float, m2: float):
    m1 = float(m1)
    m2 = float(m2)
               
    sigma2 = np.log(m2 / (m1 * m1))
    sigma = float(np.sqrt(sigma2))
    mu = float(np.log(m1) - 0.5 * sigma2)

    return mu, sigma, float(np.exp(mu))

SQRT2 = math.sqrt(2.0)
SQRT2PI = math.sqrt(2.0 * math.pi)

@njit(fastmath=True)
def _lognormal_pdf_scalar(x, mu, sigma):
    if x <= 0.0:
        return 0.0
    inv_sigma = 1.0 / sigma
    z = (math.log(x) - mu) * inv_sigma
    return math.exp(-0.5 * z * z) / (x * sigma * SQRT2PI)

@njit(fastmath=True)
def _lognormal_cdf_scalar(x, mu, sigma):
    if x <= 0.0:
        return 0.0
    z = (math.log(x) - mu) / (sigma * SQRT2)
    # Phi(z) = 0.5 * (1 + erf(z))
    return 0.5 * (1.0 + math.erf(z))

@njit(fastmath=True)
def LogNormal_fit(mu, sigma, x):
    # x is expected to be a 1D float array
    n = x.size
    pdf = np.empty(n, dtype=np.float64)
    cdf = np.empty(n, dtype=np.float64)

    for i in range(n):
        xi = x[i]
        pdf[i] = _lognormal_pdf_scalar(xi, mu, sigma)
        cdf[i] = _lognormal_cdf_scalar(xi, mu, sigma)

    return pdf, cdf

@njit
def E_LogNorm_solution_norm(a_tilde, dt, Nystrom_grid, e_sim, E_min, grid):
    E1, E2 = Global_Moments(a_tilde, dt, Nystrom_grid, 25.0)

    mu, sigma, scale = LogNormal_params(E1, E2)
    e_grid = linspace(E_min, e_sim[-1], grid)

    LN_pdf, LN_cdf = LogNormal_fit(mu, sigma, e_grid)

    p99 = Fitted_quantile(e_grid, LN_pdf, q=0.99)
    
    params = (mu, sigma, scale, e_grid)
    return params, LN_pdf, LN_cdf, p99
# ---------------------------------------------------------------------

# --- Normalized Global Moments of the Log-Normal ---------------------
@njit(fastmath=True)
def f_X(x_tilde, a_tilde):
    return 0.5 * np.exp(-np.abs(x_tilde + a_tilde))

@njit(fastmath=True)
def F_X(x_tilde, a_tilde):
    z_tilde = x_tilde + a_tilde
    if z_tilde < 0.0:
        return 0.5 * np.exp(z_tilde)
    else:
        return 1.0 - 0.5 * np.exp(-z_tilde)

@njit(cache=True, fastmath=True)
def Global_Moments(a_tilde: float, dt: float, Nystrom_grid: int, z_max_tilde: float):
    I, K, z_grid_tilde, w, h_tilde = Nystrom(z_max_tilde, Nystrom_grid, a_tilde)

    # CONDITIONAL MOMENTS CALCULATION
    m1 = solve(I - K, dt * z_grid_tilde)
    m2 = solve(I - K, 2.0 * dt * z_grid_tilde * m1 - (dt * z_grid_tilde) ** 2)

    # ENTRANCE MIXTURE: GLOBAL MOMENTS
    pdf_grid = f_X(z_grid_tilde, a_tilde)
    Ppos = 1.0 - F_X(0.0, a_tilde)

    E1 = GM_mix(m1, w, h_tilde, pdf_grid, Ppos)
    E2 = GM_mix(m2, w, h_tilde, pdf_grid, Ppos)
    return E1, E2
# ---------------------------------------------------------------------

# --- Normalized Nystrom method for the Log-Normal solution -----------
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
# ---------------------------------------------------------------------

# --- Global Mixture calculation for the Log-Normal solution ----------
@njit
def GM_mix(m_vec, w, h, pdf_at_grid, Ppos):
    total = 0.0
    for i in range(len(m_vec)):
        total += w[i] * m_vec[i] * pdf_at_grid[i]
    return (h * total) / Ppos
# =====================================================================

