import numpy as np
import pandas as pd
import importlib
import os

# --- Core functions ---
import core_SynData_and_p99
importlib.reload(core_SynData_and_p99)
from core_SynData_and_p99 import *

# --- Solvers ---
import core_Solvers
importlib.reload(core_Solvers)
from core_Solvers import *

# --- Real Data s-ratio ---
VKU_Pmax = 126      # [MW]
VKU_beta_SL = 0.7157
s_ratio = (1/VKU_beta_SL) / VKU_Pmax

# --- SIMULATION PARAMETERS ---
N = 1000000
grid = 10001
r = 0.01*60               # 1/HR
tau = 0.0 
dt = 1/60                 # 1/HR
beta = 0.6                # min/MW
Pmax = (1/beta)/s_ratio
alpha = r*Pmax            # MW/HR
a = alpha * dt            # MW
a_tilde = beta * a



# --- SIMPLE LAPLACE ---
rng = np.random.default_rng(7)
seed  = rng.random(N)
_, P_SL = simulate_Y_P(N, Pmax, 55, beta, seed)

# --- Simulation ---
P_tilde_SL = P_SL * beta
Y_SL, b_tilde_SL = smoothing_tilde(P_tilde_SL, dt, a_tilde, tau=0.0)
B = b_tilde_SL
b_mask_SL = (B > 1e-4)
b_tilde_SL_wout_P0 = B[b_mask_SL]

zero_counts = np.asarray(B < 1e-4).nonzero()
p0_sim = len(zero_counts[0])/len(b_tilde_SL)

p99_sim = np.quantile(B, 0.99)


# --- Analytical Solution ---
b_ana3, g_ana3, p0_ana3 = neumann_laplace_solver_tilde(
            A=a, beta=beta, M=3, b_max=60, n_grid=grid
    )
p99_ana3 = analytical_quantile(b_ana3, g_ana3, p0_ana3, q=0.99)

b_ana6, g_ana6, p0_ana6 = neumann_laplace_solver_tilde(
            A=a, beta=beta, M=6, b_max=60, n_grid=grid
    )
p99_ana6 = analytical_quantile(b_ana6, g_ana6, p0_ana6, q=0.99)

b_ana100, g_ana100, p0_ana100 = neumann_laplace_solver_tilde(
            A=a, beta=beta, M=100, b_max=60, n_grid=grid
    )
p99_ana100 = analytical_quantile(b_ana100, g_ana100, p0_ana100, q=0.99)

# --- Numerical Solution ---
b_nys, g_nys, p0_nys = nystrom_solver(a, beta, B_tilde=60, n_grid=1001)
p99_nys = analytical_quantile(b_nys, g_nys, p0_nys, q=0.99)

SL_database_dict = {"Y": Y_SL,"P": P_SL, "Battery_sim": b_tilde_SL, "Battery_sim_wout_p0": b_tilde_SL_wout_P0, "Battery_sim_p0": p0_sim, "battery_p99": p99_sim,
                    "b_grid": b_ana3, "g(b)_ana_M=3": g_ana3, "p0_ana_M=3": p0_ana3, "p99_ana_M=3": p99_ana3, "g(b)_ana_M=6": g_ana6,"p0_ana_M=6": p0_ana6, "p99_ana_M=6": p99_ana6, 
                    "g(b)_ana_M=100": g_ana100, "p0_ana_M=100": p0_ana100, "p99_ana_M=100": p99_ana100, "g(b)_nystrom": g_nys, "p0_nys": p0_nys, "p99_nys": p99_nys}

SL_database_df = pd.DataFrame({k: pd.Series(v) for k, v in SL_database_dict.items()})




# --- GENERALIZED LAPLACE ---
c = 0.5
c2 = (1- c)
zeta = 10
beta_gl = beta * np.sqrt(c/(zeta**2) + c)
b = zeta * beta_gl
R_max = np.nan   # no ramp cap

rng = np.random.default_rng(7)
bernoulli_exp   = rng.random(N)  # component chooser
seeds = rng.random(N)  # within-component uniform

# --- Synthetic data ---
Y_GL, P_GL = simulate_Y_P_GL(N, Pmax, 40, c, b, c2, beta_gl, R_max, bernoulli_exp, seeds)
P_tilde_GL = beta * P_GL

# --- Simulation ---
_, b_tilde_GL = smoothing_tilde(P_tilde_GL, dt, a_tilde, tau=0)
b_mask_GL = (b_tilde_GL > 1e-4)
b_tilde_GL_wout_p0 = b_tilde_GL[b_mask_GL]

zero_counts_GL = np.asarray(b_tilde_GL < 1e-6).nonzero()
p0_sim_GL = len(zero_counts_GL[0])/len(b_tilde_GL)

GL_database_dict = {"Y": Y_GL, "P": P_GL, "Battery_sim": b_tilde_GL, "Battery_sim_wout_p0": b_tilde_GL_wout_p0, "Battery_p0": p0_sim_GL}
GL_database_df = pd.DataFrame({k: pd.Series(v) for k, v in GL_database_dict.items()})


# --- Data download ---
filename = f"TimeSeries_beta-{beta}_a-{round(a,2)}_c-{c}_zeta-{zeta}_Pmax-{round(Pmax,1)}.xlsx"
with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
    SL_database_df.to_excel(writer, sheet_name="SL")
    GL_database_df.to_excel(writer, sheet_name="GL")