from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Sequence, Dict, Tuple
from scipy.stats import lognorm
from numpy.linalg import solve
import importlib
from numba import njit

# --- Core functions ---
import src.core_SynData
importlib.reload(src.core_SynData)
from src.core_SynData import *

# --- Core code helpers ---
import src.core_helpers
importlib.reload(src.core_helpers)
from src.core_helpers import *

# --- Core Power Solvers ---
import src.core_Solvers
importlib.reload(src.core_Solvers)
from src.core_Solvers import *


# === ENERGY ALGORITHMIC SOLUTION: SIMULATION ===
@njit(cache=True)
def Energy_ramps(Y, a, dt):
    N = Y.size
    B = np.zeros(N + 1, dtype=np.float64)
    energies = np.empty(N // 2 + 2, dtype=np.float64)
    n_ener = 0

    in_ramp = False
    ramp_energy_acc = 0.0

    for n in range(N):
        b_next = B[n] + (-Y[n] - a)
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


@dataclass(frozen=True)
class PDF:
    grid: np.ndarray
    f: np.ndarray

@dataclass(frozen=True)
class CDF:
    grid: np.ndarray
    f: np.ndarray

@njit
def E_algorithmic_solution_raw(a: float, Y: np.ndarray, dt:float, 
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

    energies = Energy_ramps(Y, a, dt)

    # --- pdf histogram calculation ---
    f_sim, e_sim = histogram(energies, grid)

    # --- cdf histogram calculation ---
    E_sim, F_sim = histogram_cdf(energies, 110000, vmin=1e-5)
    
    #p0_sim = count_less_than(BESS, 1e-6) / BESS.size
    p99_sim = quantile(energies, q)

    return energies, e_sim, f_sim, E_sim, F_sim, p99_sim


def E_algorithmic_solution(a, Y, dt, grid, q=0.99):
    energies, e_sim, f_sim, E_sim, F_sim, p99_sim = E_algorithmic_solution_raw(a, Y, dt, grid, q)

    E_pdf = PDF(e_sim, f_sim)
    E_cdf = CDF(E_sim, F_sim)

    return energies, E_pdf, E_cdf, p99_sim

# =====================================================================


# ======= ENERGY NUMERICAL SOLUTION: LOG-NORMAL FITTING ===============
# --- Numba friendly grid builder ---
@njit
def geomspace_grid(lo, hi, n):
    grid = np.empty(n, dtype=np.float64)

    # enforce strict positivity (required for log grid)
    if lo <= 0.0:
        lo = 1e-12
    if hi <= lo:
        hi = lo * 10.0

    if n == 1:
        grid[0] = lo
        return grid

    log_lo = np.log(lo)
    log_hi = np.log(hi)
    step = (log_hi - log_lo) / (n - 1)
    for i in range(n):
        grid[i] = np.exp(log_lo + step * i)
    return grid


# --- Fitted model ---
@dataclass(frozen=True)
class LogNormParams:
    mu: float
    sigma: float  # > 0
    scale: float  # exp(mu)
    e_grid: np.ndarray

def LogNormal_params(m1: float, m2: float):
    m1 = float(m1)
    m2 = float(m2)
               
    sigma2 = np.log(m2 / (m1 * m1))
    sigma = float(np.sqrt(sigma2))
    mu = float(np.log(m1) - 0.5 * sigma2)

    return mu, sigma, float(np.exp(mu))

def LogNormal_fit(mu: float, sigma: float, x: np.ndarray):
    x = np.asarray(x, dtype=float)
    rv = lognorm(s=sigma, scale=np.exp(mu), loc=0.0)
    return rv.pdf(x), rv.cdf(x)

def E_LogNorm_solution(e_sim: np.ndarray, E_min: float,
                       E1: float, E2: float, grid: int):
    mu, sigma, scale = LogNormal_params(E1, E2)

    e_grid = geomspace_grid(E_min, e_sim[-1], grid)
    params = LogNormParams(mu, sigma, scale, e_grid)
    LN_pdf, LN_cdf = LogNormal_fit(mu, sigma, e_grid)
    return params, LN_pdf, LN_cdf
# =====================================================================

