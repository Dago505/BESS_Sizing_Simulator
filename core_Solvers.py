from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import math
from math import comb, factorial
from numba import njit, prange
from typing import Callable, Tuple, Dict, Any
import importlib

# --- Core functions ---
import src.core_SynData
importlib.reload(src.core_SynData)
from src.core_SynData import *

# --- Core code helpers ---
import src.core_helpers
importlib.reload(src.core_helpers)
from src.core_helpers import *



# === ALGORITHMIC SOLUTION: SIMULATION ===
@njit(fastmath=True, cache=True)
def BESS_simulation(P_tilde:np.ndarray, dt:float, a_tilde:float, tau:float) -> tuple[np.ndarray, np.ndarray]:
    """ Simulates the BESS response to negative ramps that smooths the power signal under a normalized critical slope "a".
    The simulation is performed using normalized variables P_tilde = beta * P   and   a_tilde  = beta * a 

    Parameters
    ----------
    P_tilde: np.ndarray
        Primary power time series normalized by the decay rate beta  
    dt: float
        Time step between consecutive samples of the Primary power
    a_tilde: float
        Normalized critical slope [dP/dt]
    tau: float
        Time response parameter of the BESS. tau=0 represents an ideal response.

    Returns
    --------
    S: np.ndarray
        Array of the smoothed Power series after the BESS response to negative ramps
    B: np.ndarray
        Array of the battery response to mitigate negative ramps such that P = S + B
    """
    N = len(P_tilde)
    BESS_t = np.zeros(N)
    SMOOTH_t = np.zeros(N)
    P_t = P_tilde  

    gamma = 1.0 if tau == 0.0 else 1.0 - np.exp(-dt / tau)
    SMOOTH_t[0] = P_t[0]

    for i in range(1, N):
        deltak_t = (P_t[i] - SMOOTH_t[i-1])
        if deltak_t >= -a_tilde:
            S_desired_t = P_t[i]
        else:
            S_desired_t = SMOOTH_t[i-1] - a_tilde

        B_desired_t = S_desired_t - P_t[i]

        if i == 1:
            BESS_t[i] = B_desired_t
        else:
            BESS_t[i] = BESS_t[i-1] + (B_desired_t - BESS_t[i-1]) * gamma

        SMOOTH_t[i] = P_t[i] + BESS_t[i]

    return SMOOTH_t, BESS_t


@njit
def algorithmic_solution(a_tilde: float, beta: float, P: np.ndarray, dt:float, 
                         grid: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """ Computes the pdf "g(b)" of the BESS simulation an returns the main characteristics including the
    point mass "p0" (inactive battery) and the 99 percentile "p99". 

    Parameters
    ----------
    a_tilde: float
        Normalized critical slope [dP/dt]
    beta: float
        Decay rate of the increment distribution Y to normalize the Primary Power series
    P: np.ndarray
        Array of the simulated Primary Power
    dt: float
        Time step between consecutive samples of the Primary power
    grid: int
        Resolution of the BESS pdf vector g(b)

    Returns
    -------
    BESS: np.ndarray
        Simulation of the battery activation
    b_sim: np.ndarray
        Battery grid based on the BESS simulation histogram bins
    g_sim: np.ndarray
        pdf of the battery simulation "g(b)". It ignores the point mass, only returns the tail of the distribution
    p0_sim: float
        Point mass at zero of the battery pdf "g(b)"
    p99_sim: float
        99 percentil of the battery pdf "g(b)"
    """
    P_tilde = beta * P
    _, BESS = BESS_simulation(P_tilde, dt, a_tilde, 0.0)

    B_pos = BESS[BESS > 1e-4]

    g_sim, b_sim = histogram(B_pos, grid)

    p0_sim = count_less_than(BESS, 1e-6) / BESS.size
    p99_sim = quantile(BESS, 0.99)

    return BESS, b_sim, g_sim, p0_sim, p99_sim
# =====================================================================






# ======= NUMERICAL SOLVER: NYSTROM ==========
def numerical_solution(a: float, beta: float, b_max: float, 
                       n_grid: int) -> tuple[np.ndarray, np.ndarray, float, float]:
    """ Solves the fixed point integral equation associated with the SL kernel using the Nystrom
    method discretization on interval "[0, b_max]".  
    (1) Constructs a uniform grid
    (2) Evaluates the Laplace kernel
    (3) Applies a composite trapezoidal quadrature weights
    (4) Solves the linear system "u = f_tilde + K*u" where K is the discretized integral operator

    Parameters
    ----------
    a: float
        Critical slope 
    beta: float
        Decay rate of the primary power changes distribution Y
    b_max: float
        Upper limit of the integral equation
    n_grid: int
        Resolution of the Nystrom discretized operator
    
    Returns
    -------
    b_nys: np.ndarray
        Battery grid of the integral operator
    g_nys: np.ndarray
        pdf of the BESS activation based on the linear nystrom operator
    p0_nys: float
        Point mass at zero of the battery response
    p99_nys: float
        99 percentil of the battery pdf
    """
    def f_tilde(y):
        y = np.asarray(y)
        return 0.5 * np.exp(-beta * np.abs(y / beta + a))

    h_t = b_max / n_grid
    b_nys = np.linspace(0.0, b_max, n_grid + 1)

    w = np.ones(n_grid + 1)
    w[0] = w[-1] = 0.5

    idx = np.arange(n_grid + 1)
    D_t = (idx[:, None] - idx[None, :]) * h_t
    K = h_t * f_tilde(D_t) * w[None, :]

    fvec = f_tilde(b_nys)
    A = np.eye(n_grid + 1) - K
    u_t = np.linalg.solve(A, fvec)

    omega_t = h_t * np.dot(w, u_t)
    p0_nys = 1.0 / (1.0 + omega_t)
    g_nys = p0_nys * u_t

    resid = A @ g_nys - p0_nys * fvec

    p99_nys = analytical_quantile(b_nys, g_nys, p0_nys)

    return b_nys, g_nys, p0_nys, p99_nys
# ===========================================================================






# ================ NEUMANN SOLVER: ANALYTICAL  =============================
@njit
def analytical_solution(a, beta, M, b_max, n_grid):
    """Compute the full analytical stationary solution of the storage model,
    including the PDF g(b), the point mass p0 at b = 0, and the P99
    quantile of the distribution.

    Parameters
    ----------
    a : float
        Drift increment per time–step (Critical slope).
    beta : float
        Decay rate parameter of the primary power change distribution
    M : int
        Truncation order of the analytical series.
    b_max : float
        Maximum physical storage range for the grid over which g(b) is
        evaluated.
    n_grid : int
        Number of grid points used to discretize the interval [0, b_max].

    Returns
    -------
    b_tilde : ndarray
        Grid of storage values (normalized), shape (n_grid,).
    g_vals : ndarray
        Analytical probability density g(b) evaluated on the grid,
        already normalized so that ∫ g(b) db + p0 = 1.
    p0 : float
        Probability mass at b = 0. 
    p99_ana : float
        The 99th percentile (P99) of the stationary distribution obtained
        from the analytical solution.
    """
    b_ana, g_ana, p0_ana = neumann_laplace_solver(a, beta, M, b_max, n_grid)
    p99_ana = analytical_quantile(b_ana, g_ana, p0_ana)
    return b_ana, g_ana, p0_ana, p99_ana

@njit(cache=True, fastmath=True)
def neumann_laplace_solver(A, beta, M, b_max, n_grid):
    """Compute the analytical stationary solution of the Neumann–Laplace
    storage model using the truncated u–series representation
    
    Parameters
    ----------
    A : float
        Drift increment per time–step (Critical slope).
    beta : float
        Decay rate parameter of the primary power change distribution
    M : int
        Truncation order of the analytical series.
    b_max : float
        Maximum physical storage range for the grid over which g(b) is
        evaluated.
    n_grid : int
        Number of grid points used to discretize the interval [0, b_max].
    
    Returns
    -------
    b_tilde : ndarray
        Grid of storage values (normalized), shape (n_grid,).
    g_vals : ndarray
        Analytical probability density g(b) evaluated on the grid,
        already normalized so that ∫ g(b) db + p0 = 1.
    p0 : float
        Probability mass at b = 0. 
    """
    A_tilde = beta * A
    b_tilde = np.linspace(0.0, b_max, n_grid)

    # Precompute factorial logs once and reuse everywhere
    log_fact = _precompute_factorials(M + 1)

    # build alpha table
    abar = build_alpha_table_bar_tilde(M, A_tilde, log_fact)

    # u-series on grid
    u_tilde = u_series_M_tilde(b_tilde, A_tilde, abar)

    # mass
    mU = mass_Omega_tilde(A_tilde, abar, log_fact)

    p0 = 1.0 / (1.0 + mU)
    g_vals = p0 * u_tilde

    return b_tilde, g_vals, p0


# ---------- generic log-space helpers ----------
# A very negative number to represent log(0) in log-space
LOG_ZERO = -1.0e300

@njit(cache=True)
def _precompute_factorials(M):
    fact_log = np.empty(M + 1, dtype=np.float64)
    fact_log[0] = 0.0  # log(0!) = 0
    for j in range(1, M + 1):
        fact_log[j] = fact_log[j - 1] + math.log(j)
    return fact_log


@njit(cache=True)
def _log_add(log_a, log_b):
    if log_a <= LOG_ZERO:
        return log_b
    if log_b <= LOG_ZERO:
        return log_a

    # ensure log_a >= log_b
    if log_a < log_b:
        log_a, log_b = log_b, log_a

    diff = log_b - log_a
    if diff < -50.0:
        return log_a
    return log_a + math.log1p(math.exp(diff))


@njit(cache=True)
def _log_binom_from_logfact(n, k, log_fact):
    if k < 0 or k > n:
        return LOG_ZERO
    return log_fact[n] - log_fact[k] - log_fact[n - k]


@njit(cache=True)
def _safe_exp(log_x):
    if log_x < -700.0:
        return 0.0
    if log_x > 700.0:
        log_x = 700.0
    return math.exp(log_x)


# ---------- L and U coefficients ----------

# L_{k->r} = (1/2) * (1/(k+1)) * C(k+1, r) * A^{k+1-r},  r=1..k+1; else 0
@njit(fastmath=True, cache=True)
def _Lcoef_tilde(k, r, A_tilde, log_fact, log_half, log_A):
    if r < 1 or r > (k + 1) or A_tilde <= 0.0:
        return 0.0

    log_C = _log_binom_from_logfact(k + 1, r, log_fact)
    if log_C <= LOG_ZERO:
        return 0.0

    # log L = log(1/2) - log(k+1) + log C(k+1,r) + (k+1-r)*log(A_tilde)
    log_L = log_half - math.log(k + 1.0) + log_C + (k + 1 - r) * log_A
    return _safe_exp(log_L)


# U_{k->r} = sum_{m=r}^k (1/2) * (k!/m!) * C(m,r) * A^{m-r} * (1/2)^{k-m+1}
@njit(fastmath=True, cache=True)
def _Ucoef_tilde(k, r, A_tilde, log_fact, log_half, log_A):
    if r < 0 or r > k or A_tilde <= 0.0:
        return 0.0

    log_sum = LOG_ZERO

    for m in range(r, k + 1):
        # log(k!/m!)
        log_k_over_m = log_fact[k] - log_fact[m]
        # log C(m, r)
        log_C_mr = log_fact[m] - log_fact[r] - log_fact[m - r]
        # (m-r)*log(A)
        log_A_term = (m - r) * log_A
        # (k-m+1)*log(1/2)
        log_half_power = (k - m + 1) * log_half

        # leading factor 1/2 -> +log(1/2)
        log_term = log_half + log_k_over_m + log_C_mr + log_A_term + log_half_power

        log_sum = _log_add(log_sum, log_term)

    if log_sum <= LOG_ZERO:
        return 0.0

    return _safe_exp(log_sum)


# ---------- alpha-bar table (as ndarray) ----------

@njit(fastmath=True, cache=True)
def build_alpha_table_bar_tilde(M, A_tilde, log_fact):
    abar = np.zeros((M + 1, M + 1), dtype=np.float64)
    abar[0, 0] = 0.5

    log_half = math.log(0.5)
    log_A = math.log(A_tilde)

    for n in range(M):
        # r runs 0..n+1 (inclusive)
        for r in range(n + 2):
            s = 0.0
            for k in range(n + 1):
                ank = abar[n, k]
                if ank == 0.0:
                    continue

                Lkr = _Lcoef_tilde(k, r, A_tilde, log_fact, log_half, log_A)
                Ukr = _Ucoef_tilde(k, r, A_tilde, log_fact, log_half, log_A)

                if Lkr != 0.0:
                    s += ank * Lkr
                if Ukr != 0.0:
                    s += ank * Ukr

            abar[n + 1, r] = s

    return abar


# ---------- u_series and mass in nopython loops ----------

@njit(fastmath=True, cache=True)
def u_series_M_tilde(b_tilde, A_tilde, abar):
    Ngrid = b_tilde.shape[0]
    M = abar.shape[0] - 1
    out = np.zeros(Ngrid, dtype=np.float64)

    for i in range(Ngrid):
        bt = b_tilde[i]
        w = bt + A_tilde
        if w < 0.0:
            w = 0.0

        if w > 0.0:
            logw = math.log(w)
        else:
            logw = 0.0  # unused if w == 0

        acc_i = 0.0

        for n in range(M + 1):
            # log polynomial: sum_{k=0}^n abar[n,k] w^k
            log_poly = LOG_ZERO

            for k in range(n + 1):
                ank = abar[n, k]
                if ank <= 0.0:
                    continue

                if k == 0:
                    log_term = math.log(ank)
                else:
                    if w == 0.0:
                        # w^k = 0 for k>0
                        continue
                    log_term = math.log(ank) + k * logw

                log_poly = _log_add(log_poly, log_term)

            if log_poly <= LOG_ZERO:
                continue

            # combine with exp(-(bt + (n+1)*A_tilde))
            log_contrib = log_poly - (bt + (n + 1) * A_tilde)
            contrib = _safe_exp(log_contrib)
            if contrib == 0.0:
                continue

            acc_i += contrib

        out[i] = acc_i

    return out


@njit(fastmath=True, cache=True)
def mass_Omega_tilde(A_tilde, abar, log_fact):
    M = abar.shape[0] - 1
    total = 0.0
    logA = math.log(A_tilde)

    for n in range(M + 1):
        fn_log = -(n + 1) * A_tilde
        if fn_log < -800.0:
            # negligible
            continue

        inner = 0.0

        for k in range(n + 1):
            ank = abar[n, k]
            if ank == 0.0:
                continue

            # H_k in log-space: sum_{j=0}^k C(k, j) A^{k-j} j!
            log_Hk = LOG_ZERO

            for j in range(k + 1):
                # log C(k, j)
                log_C = log_fact[k] - log_fact[j] - log_fact[k - j]
                # log term = log C(k,j) + (k-j)*logA + log(j!)
                log_term = log_C + (k - j) * logA + log_fact[j]

                log_Hk = _log_add(log_Hk, log_term)

            if log_Hk <= LOG_ZERO:
                continue

            Hk = _safe_exp(log_Hk)
            inner += ank * Hk

        if inner == 0.0:
            continue

        factor = _safe_exp(fn_log)
        total += factor * inner

    return total