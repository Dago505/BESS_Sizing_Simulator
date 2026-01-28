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

@njit
def E_algorithmic_solution(a: float, Y: np.ndarray, dt:float, 
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

    f_sim, e_sim = histogram(energies, grid)

    #p0_sim = count_less_than(BESS, 1e-6) / BESS.size
    p99_sim = quantile(energies, q)

    return energies, e_sim, f_sim, p99_sim
# =====================================================================


# ======= ENERGY NUMERICAL SOLUTION: LOG-NORMAL FITTING ===============
@dataclass(frozen=True)
class LogNormParams:
    mu: float
    sigma: float  # > 0
    scale: float  # exp(mu)
    e_grid: None[np.ndarray]

def LogNormal_params(m1: float, m2: float) -> LogNormParams:
    m1 = float(m1)
    m2 = float(m2)

    if not np.isfinite(m1) or not np.isfinite(m2):
        raise ValueError("m1 and m2 must be finite.")
    if m1 <= 0.0:
        raise ValueError("Need m1 > 0 for a lognormal fit.")
    if m2 <= m1 * m1:
        raise ValueError("Need positive variance: m2 > m1^2 for a lognormal fit.")

    sigma2 = np.log(m2 / (m1 * m1))
    sigma = float(np.sqrt(sigma2))
    mu = float(np.log(m1) - 0.5 * sigma2)

    return mu, sigma, np.exp(mu)

# Log-Normal fit
def LogNormal_fit(mu: float, sigma: float, x: np.ndarray):
    x = np.asarray(x, dtype=float)
    if sigma <= 0:
        raise ValueError("sigma must be > 0")
    if np.any(x <= 0):
        raise ValueError("x must be strictly positive for lognormal PDF/CDF")

    rv = lognorm(s=sigma, scale=np.exp(mu), loc=0.0)
    pdf = rv.pdf(x)
    cdf = rv.cdf(x)
    return pdf, cdf

def E_LogNorm_solution(e_sim: np.ndarray, energies: np.ndarray, 
                       E1: float, E2: float, grid: int) -> tuple[LogNormParams, np.ndarray, np.ndarray]:
    mu, sigma, scale = LogNormal_params(E1, E2)
    e_grid = np.linspace(min(e_sim), np.max(energies), grid)
    params = LogNormParams(mu, sigma, scale, e_grid)
    LN_pdf, LN_cdf = LogNormal_fit(mu, sigma, e_grid)
    return params, LN_pdf, LN_cdf
# =====================================================================