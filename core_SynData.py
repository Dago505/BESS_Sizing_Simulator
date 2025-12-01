from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import math
import numba as nb
from numba import njit
from typing import Callable, Tuple, Dict, Any

# ===========================================
# --- SL SYNTHETIC TIME SERIES --- 
# ===========================================
@njit(fastmath=True, cache=True)
def sample_trunc_laplace_SL(a_t, c_t, u):
    Fa = 0.5*math.exp(a_t) if a_t <= 0.0 else 1.0 - 0.5*math.exp(-a_t)
    Fc = 0.5*math.exp(c_t) if c_t <= 0.0 else 1.0 - 0.5*math.exp(-c_t)
    w  = Fc - Fa
    if w <= 1e-14:
        return 0.5*(a_t + c_t)
    p = Fa + u*w
    return math.log(2.0*p) if p <= 0.5 else -math.log(2.0*(1.0 - p))

@njit(fastmath=True, cache=True)
def simulate_Y_P_SL(N:int, P_max:float, P0:float, beta:float, seed:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulates the increments of a truncated Simple Laplace distribution (Y) and its cumulative path (P). 
    The limits of the available range of the distribution are bound between the lower and upper limit [L, U]
    
    Parameters
    -----------
    N: int
        Number of time steps to simulate
    P_max: float
        Upper bound for the cumulative variable P. Y state variable is truncated such that P[n] + Y <= P_max
    P0: float
        Initial value of the process P
    beta: float
        Charactistic scale (decay rate) of the Laplace exponential tail of Y
    seed: np.ndarray
        Array of random percentile selector inside the bounded distribution to determine the increment Y[n]
    -----------

    Returns
    -------
    Y: np.ndarray
        Array of simulated random increments following SL law (Primary Power changes)
    P: np.ndarray
        Array of the simulated increments cumulative process (Primary Power)
    """
    Y = np.empty(N, dtype=np.float64)
    P = np.empty(N + 1, dtype=np.float64)
    P[0] = P0
    b = float(beta)

    for n in range(N):
        L = -P[n]
        U = P_max - P[n]
        if U < L:
            y = 0.0
        else:
            a_t = b*L
            c_t = b*U
            y_t = sample_trunc_laplace_SL(a_t, c_t, seed[n])
            y   = y_t / b
        Y[n]   = y
        P[n+1] = P[n] + y

    return Y, P


# ===========================================
# --- AR(p) SL SYNTHETIC TIME SERIES --- 
# ===========================================
@njit(fastmath=True, cache=True)
def simulate_Y_P_ARp(N, P_max, P0, beta, phi, seed):
    p = phi.shape[0]
    Y = np.empty(N, dtype=np.float64)
    P = np.empty(N + 1 + p, dtype=np.float64)  # extra p to simplify indexing
    for i in range(p):
        P[i] = P0  # Initialize with constant or historical values

    b = float(beta)

    for n in range(N):
        # AR(p) prediction
        AR_n = 0.0
        for j in range(p):
            AR_n += phi[j] * P[n + p - 1 - j]  # Most recent to oldest

        # Compute noise bounds so that P[n+p] stays in bounds
        L_noise = -AR_n
        U_noise = P_max - AR_n

        if U_noise < L_noise:
            eps = 0.0
        else:
            a_t = b * L_noise
            c_t = b * U_noise
            eps_t = sample_trunc_laplace_SL(a_t, c_t, seed[n])
            eps = eps_t / b

        P[n + p] = AR_n + eps
        Y[n] = P[n + p] - P[n + p - 1]  # Increment

    return Y, P[p:]


# ===========================================
# --- GL SYNTHETIC TIME SERIES --- 
# ===========================================
@njit(fastmath=True, cache=True)
def trunc_laplace_GL(L, U, r, u):
    FL = 0.5 * math.exp(r * L) if L <= 0.0 else 1.0 - 0.5 * math.exp(-r * L)
    FU = 0.5 * math.exp(r * U) if U <= 0.0 else 1.0 - 0.5 * math.exp(-r * U)

    w = FU - FL
    if w <= 1e-14:                           
        return 0.5 * (L + U)

    p = FL + u * w

    if p <= 0.5:
        return (1.0 / r) * math.log(2.0 * p)
    else:
        return -(1.0 / r) * math.log(2.0 * (1.0 - p))


@njit(fastmath=True, cache=True)
def simulate_Y_P_GL(N:int, P_max:float, P0:float, c:float, c2:float, 
                    b:float, beta_GL:float, bernoulli_exp:np.ndarray, seeds:np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Simulates the increments of a truncated Generalized Laplace distribution (Y) and its cumulative path (P).
    It performes a mixture between the short-tail decay rate (b) and long-tail decay rate (beta_GL) selecting an
    specific increment law weighted by the probability indicator pi1
    
    Parameters
    -----------
    N: int
        Number of time steps to simulate
    P_max: float
        Upper bound for the cumulative variable P. Y state variable is truncated such that P[n] + Y <= P_max
    P0: float
        Initial value of the process P
    c: float
        Weight ratio component for the short-tail of the distribution. Must be between [0,1]
    c2: float
        Weight ratio component for the long-tail of the distribution
    b: float
        Decay rate of the short-tail added of the GL distribution
    beta_GL: float
        Decay rate of the long-tail added of the GL distribution
    bernoulli_exp: np.ndarray
        Uniform selector variable [0,1]. It randomly selects which decay rate drives the next increment based on 
        the probability indicator pi1
    seed: np.ndarray
        Array of random percentile selector inside the bounded distribution to determine the increment Y[n]
    -----------

    Returns
    -------
    Y: np.ndarray
        Array of simulated random increments following SL law (Primary Power changes)
    P: np.ndarray
        Array of the simulated increments cumulative process (Primary Power)
    """

    # R_max_nan: use np.nan to indicate "no ramp cap"
    Y = np.empty(N, dtype=np.float64)
    P = np.empty(N + 1, dtype=np.float64)
    P[0] = P0

    for n in range(N):
        # feasible bounds from state
        L = -P[n]
        U = P_max - P[n]

        if U < L:
            y = 0.0
        else:
            # component masses in [L,U]
            # Comp 1: rate = b
            F1L = 0.5 * math.exp(b * L) if L <= 0.0 else 1.0 - 0.5 * math.exp(-b * L)
            F1U = 0.5 * math.exp(b * U) if U <= 0.0 else 1.0 - 0.5 * math.exp(-b * U)

            # Comp 2: rate = beta_GL
            F2L = 0.5 * math.exp(beta_GL * L) if L <= 0.0 else 1.0 - 0.5 * math.exp(-beta_GL * L)
            F2U = 0.5 * math.exp(beta_GL * U) if U <= 0.0 else 1.0 - 0.5 * math.exp(-beta_GL * U)

            m1 = F1U - F1L
            m2 = F2U - F2L

            Z  = c * m1 + c2 * m2
            if Z <= 1e-16:
                y = 0.5 * (L + U)            # practically zero mass: safe midpoint
            else:
                pi1 = (c * m1) / Z          # posterior weight for comp 1
                # choose component with u_comp[n]
                if bernoulli_exp[n] < pi1:
                    # draw from comp 1 truncated
                    y = trunc_laplace_GL(L, U, b, seeds[n])
                else:
                    # draw from comp 2 truncated
                    y = trunc_laplace_GL(L, U, beta_GL, seeds[n])

        Y[n]   = y
        P[n+1] = P[n] + y

        # rare clamp for numerical drift
        if P[n+1] < 0.0:
            P[n+1] = 0.0
            Y[n]   = P[n+1] - P[n]
        elif P[n+1] > P_max:
            P[n+1] = P_max
            Y[n]   = P[n+1] - P[n]

    return Y, P