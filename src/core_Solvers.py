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
                         grid: int, q: float = 0.99) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
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
    p99_sim = quantile(BESS, q)

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
def analytical_solution(a: float, beta: float, M: int, b_max: float, n_grid: int) -> tuple[np.ndarray, np.ndarray, float, float]:

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

    A_tilde = beta * a
    # --- battery grid ---
    b_ana = linspace(0.0, b_max, n_grid)
    # --- Coefficients Lambda ---
    Lambda = Lambda_bar(A_tilde, M)
    # --- re-scaled BESS power pdf ---
    u_vals = u_series_grid(b_ana, A_tilde, Lambda)
    # --- point mass at zero ---
    Omega = omega_tilde(A_tilde, Lambda)
    p0_ana = 1.0 / (1.0 + Omega)
    # --- normalized pdf ---
    g_ana = p0_ana * u_vals
    # --- 99 percentile ---
    p99_ana = analytical_quantile(b_ana, g_ana, p0_ana, q=0.99)
    return b_ana, g_ana, p0_ana, p99_ana


# ----- Portions of the Lambda coefficient -----
@njit
def L_coef(A_tilde, k, r):
    if r < 1 or r > (k + 1):
        return 0.0
    comb = binom(k + 1, r)
    pow_a = A_tilde ** (k + 1 - r)
    return 0.5 * comb * pow_a / (k + 1.0)

@njit
def U_coef(A_tilde, k, r):
    if r < 0 or r > k:
        return 0.0
    
    total = 0.0
    for m in range(r, k + 1):
        k_over_m = factorial(k) / factorial(m)
        comb_mr = binom(m, r)
        pow_a = A_tilde ** (m - r)
        pow_half = 0.5 ** (k - m + 1)
        total += 0.5 * k_over_m * comb_mr * pow_a * pow_half
    return total

# ---------- Build Lambda ----------
@njit
def Lambda_bar(A_tilde, M):
    Lambda = np.zeros((M + 1, M + 1), dtype=np.float64)
    Lambda[0, 0] = 0.5  # initial coefficient from u0(b) = 1/2 e^{-(b + A_tilde)}

    for n in range(M):
        # r runs from 0..n+1
        for r in range(n + 2):
            s = 0.0
            for k in range(n + 1):
                lam_nk = Lambda[n, k]
                if lam_nk == 0.0:
                    continue
                s += lam_nk * (L_coef(A_tilde, k, r) + U_coef(A_tilde, k, r))
            Lambda[n + 1, r] = s
    return Lambda

# ---------- u-series evaluation on a grid ----------
@njit
def u_series_grid(b_grid, A_tilde, Lambda):
    M = Lambda.shape[0] - 1
    Ngrid = b_grid.shape[0]
    out = np.zeros(Ngrid, dtype=np.float64)

    for i in range(Ngrid):
        b = b_grid[i]
        w = b + A_tilde

        # precompute powers of w up to M
        pow_w = np.ones(M + 1, dtype=np.float64)
        for k in range(1, M + 1):
            pow_w[k] = pow_w[k - 1] * w

        acc = 0.0
        for n in range(M + 1):
            inner = 0.0
            for k in range(n + 1):
                lam_nk = Lambda[n, k]
                if lam_nk == 0.0:
                    continue
                inner += lam_nk * pow_w[k]

            acc += math.exp(-(b + (n + 1) * A_tilde)) * inner
        out[i] = acc
    return out

# ---------- Omega tilde for point mass calculation  ----------
@njit
def omega_tilde(A_tilde, Lambda_bar):
    M = Lambda_bar.shape[0] - 1
    total = 0.0

    for n in range(M + 1):
        sum_k = 0.0
        for k in range(n + 1):
            lam = Lambda_bar[n, k]
            if lam == 0.0:
                continue

            # H_k = sum_{j=0}^k C(k,j) A_tilde^{k-j} j!
            sum_j = 0.0
            for j in range(k + 1):
                c = binom(k, j)
                sum_j += c * (A_tilde ** (k - j)) * factorial(j)

            sum_k += lam * sum_j

        total += math.exp(-(n + 1) * A_tilde) * sum_k

    return total