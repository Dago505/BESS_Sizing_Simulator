from __future__ import annotations
import numpy as np
from dataclasses import dataclass
import math
from numba import njit
from typing import Callable, Tuple, Dict, Any
import time

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FormatStrFormatter
import importlib
import os


# Asegurar carpeta de salida
os.makedirs("Figures", exist_ok=True)

plt.rcParams.update({
    "text.usetex": True,     # Tipografía LaTeX
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "legend.fontsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16
})

step = 20
def take_every(arr, start, step):
    return arr[start::step]


# ====== FIGURE 1) RQ 1.1 - Comparison of Analytical and algorithmic PDF/CDF ===============
#  --- How does theoretical results predict the empirical pdf/cdf ---
# ================================================================================

def RQ_11_FIGURE(simulation, analytical, p99_values, p99_inf, M_values):
    # --- Simulation ---
    b_sim, g_sim, G_sim = simulation

    # --- Analytical ---
    b_ana, g_ana_list, G_ana3, G_ana6, G_ana100 = analytical

    # --- FIGURE ---
    colors = ['r', "b", 'lightgreen']
    dark_orange = "#f28500"

    fig, ax1 = plt.subplots(figsize=(9, 6), dpi=300)

    # ----- Left y-axis: log(g(b_tilde)) -----
    for i in range(len(g_ana_list)):
        b_ana, g_ana, p0_ana = g_ana_list[i]
        ax1.plot(
            b_ana,
            g_ana/(1 - p0_ana),
            linewidth=3,
            color=colors[i],
            label=fr"$M={M_values[i]}$"
        )

    ax1.plot(b_sim, g_sim, color='k', linewidth=1.5, label=r"Empirical PDF")

    ax1.set_yscale("log")
    ax1.set_ylabel(r"$\log\!\big(\tilde{g}(\tilde{b})\big)$", color='k')
    ax1.tick_params(axis='y', labelcolor='k')
    ax1.yaxis.set_label_position("left")
    ax1.yaxis.tick_left()

    # ----- Right y-axis: CDF -----
    ax2 = ax1.twinx()

    ax2.plot(b_sim, G_sim, color=dark_orange, linewidth=2.8,
            label=r"Empirical CDF")

    ax2.scatter(
        take_every(b_ana, 0, step),
        take_every(G_ana3, 0, step),
        marker="x",
        color=colors[0]
    )
    ax2.scatter(
        take_every(b_ana, 10, step),
        take_every(G_ana6, 10, step),
        marker="^",
        color=colors[1]
    )
    ax2.scatter(
        take_every(b_ana, 20, step),
        take_every(G_ana100, 20, step),
        marker="o",
        color=colors[2]
    )

    ax2.set_ylabel(r"$\tilde{G}(\tilde{b})$", color=dark_orange)
    ax2.tick_params(axis='y', labelcolor=dark_orange)
    ax2.set_ylim(0.5, 1.05)
    ax2.yaxis.set_label_position("right")
    ax2.yaxis.tick_right()

    # --- p99 value (líneas verticales) ---
    colors_P99 = ["#f28500", 'r', "b", "lightgreen"]
    for i, p99 in enumerate(p99_values):
        plt.axvline(x=p99, linestyle="--", linewidth=1.5, color=colors_P99[i])

    plt.axvline(x=p99_inf, linestyle=":", linewidth=1.5, color="k")

    # ----- Shared x-axis -----
    ax1.set_xlabel(r"$\tilde{b}$")
    ax1.set_xlim(0, 10)
    ax1.grid(True)

    # =======================
    # 1) Primera leyenda (PDF, CDF, M)
    # =======================
    custom_handles = [
        Line2D([0], [0], color='k', lw=1.2,
            label=r"Empirical PDF"),
        Line2D([0], [0], color=dark_orange, lw=2.8,
            label=r"Empirical CDF")
    ]

    markers = ["x", "^", "o"]
    for i, M_val in enumerate(M_values):
        handle = Line2D(
            [0], [0],
            color=colors[i],
            marker=markers[i],
            markersize=7,
            linewidth=3,
            label=fr"$M={M_val}$"
        )
        custom_handles.append(handle)

    leg1 = ax1.legend(handles=custom_handles, loc='lower left', frameon=True)
    # Fondo opaco blanco
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.95)

    # MUY IMPORTANTE: fijar esta leyenda como artista para que no la reemplace la siguiente
    ax1.add_artist(leg1)

    # =======================
    # 2) Segunda leyenda (P99)
    # =======================
    handles_p99 = []
    for i in range(len(p99_values)-1):
        p99 = p99_values[i+1]
        handle = Line2D(
            [0], [0],
            color=colors_P99[i+1],
            linestyle="--",
            linewidth=1.5,
            label = rf"$P_{{99}}^{{\mathrm{{ana}}}}\left(M={M_values[i]}\right) = {p99:.2f}$"
        )
        handles_p99.append(handle)

    handle = Line2D(
            [0], [0],
            color=colors_P99[0],
            linestyle="--",
            linewidth=1.5,
            label=rf"$P_{{99}}^{{\mathrm{{sim, P_0=150}}}} = {p99_values[0]:.2f}$"
        )
    handles_p99.append(handle)

    handle = Line2D(
            [0], [0],
            color="k",
            linestyle=":",
            linewidth=0.8,
            label=rf"$P_{{99}}^{{\mathrm{{sim, P_0=\infty}}}} = {p99_inf:.2f}$"
        )
    handles_p99.append(handle)

    leg2 = ax1.legend(handles=handles_p99, loc="lower right", frameon=True, fontsize=13)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.95)

    # ----- Title -----
    plt.title(r"PDF and CDF comparison for different $M$")

    plt.tight_layout()

    # ----- Save PDF -----
    output_path = "Figures/RQ1_figure.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()



# ====== RQ 1.2 - Comparison of P99 curve between analytical and algorithmic solution ===============
#  --- How does results translate into calculations for p99 ---
# ===================================================================================================

def RQ_12_FIGURE(a_tilde_values, p99_vals):

    p99_sim = [item["p99_sim"] for item in p99_vals]
    p99_ana = [item["p99_ana"] for item in p99_vals]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # ----- Figura principal -----
    ax.plot(a_tilde_values, p99_sim, label=r"Simulation", color="blue")


    # ax.axvline(x=0.16, linestyle="--", color="orange")  # <-- ya no se usa

    ax.scatter(a_tilde_values, p99_ana, label=r"Analytical ($M=100$)", color="red")

    ax.set_title(r"$P_{99}$ vs. $\tilde{b}$" "\n" r"Analytical vs. Simulation Comparison")
    ax.set_xlabel(r"$\tilde{a}$")
    ax.set_ylabel(r"$P_{99}(\tilde{b})$")
    ax.grid(True)
    ax.legend(
        bbox_to_anchor=(0.075, 0.95),
        frameon=True,
        fontsize=12
    )

    # Calcular una altura baja para la etiqueta (cerca del borde inferior del eje y)
    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.1 * (ymax - ymin)

    # Sombrear región 0 <= x <= 0.16 en gris
    ax.axvspan(0.0, 0.16, color="gray", alpha=0.4, zorder=0.0)

    ax.annotate(r"$e > 10\%$", xy=(0.16, y_text), xytext=(0.3, y_text-0.5), 
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    # ----- Subfigura en la parte superior derecha -----
    ax_inset = inset_axes(ax,
                        width="45%",
                        height="45%",
                        bbox_to_anchor=(-0.12, -0.17, 1.1, 1.1),
                        bbox_transform=ax.transAxes,
                        loc="upper right")

    ax_inset.plot(a_tilde_values, p99_sim, color="blue")
    ax_inset.scatter(a_tilde_values, p99_ana, color="red")

    ax_inset.set_yscale("log")
    ax_inset.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax_inset.set_title(r"log-scale", fontsize=12)

    ax_inset.tick_params(axis='both', which='major', labelsize=10)
    ax_inset.set_ylabel(r"$\log\Big(P_{99}(\tilde{b})\Big)$", fontsize=11)
    ax_inset.set_xlabel(r"$\tilde{a}$", fontsize=11)

    plt.tight_layout()

    # Guardar PDF
    output_path = "Figures/RQ1.2_figure.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.show()



# ====== RQ 1.3 - M values needed for convergence =======================
#  --- Convergence analysis to 95% to stationary state vs M terms ---
# =======================================================================

def RQ_13_FIGURE(a_tilde_values, M_convergence):
    M_conv = np.asarray([item["M"] for item in M_convergence])

    idx_M10 = np.argmax(M_conv == 10)
    idx_M5 = np.argmax(M_conv == 5)
    idx_M2 = np.argmax(M_conv == 2)

    a_tilde_M10 = M_convergence[idx_M10]["a_tilde"]; print(a_tilde_M10)
    a_tilde_M5 = M_convergence[idx_M5]["a_tilde"]
    a_tilde_M2 = M_convergence[idx_M2]["a_tilde"]; print(a_tilde_M2)

    plt.figure(figsize=[4, 6])
    plt.plot(a_tilde_values, M_conv, marker="o", color="k")

    c = 0.5
    cy = 5

    plt.annotate(r"$M=10$", xy=(a_tilde_M10, 10+0.1), xytext=(a_tilde_M10 + c, 10 + cy+4),
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    plt.annotate(r"$M=5$", xy=(a_tilde_M5, 5+0.3), xytext=(a_tilde_M5 + c, 5 + cy-0.5),
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    plt.annotate(r"$M=2$", xy=(a_tilde_M2, 2+0.1), xytext=(a_tilde_M2 + c, 2 + cy-3),
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    plt.title(r"M required for convergence of $\tilde{g}(\tilde{b})$" + "\n" + r"at $D_{L^1}=0.05$")
    plt.xlabel(r"$\tilde{a}$")
    plt.ylabel(r"\#N")
    plt.yscale("log")
    plt.xlim([min(a_tilde_values)-0.1, 2.5])
    plt.grid(True, alpha=0.5)
    plt.show()



# ====== FIGURE 2) RQ 1.2 and RQ 1.3 =======================
#  --- Combined figure ---
# ================================================

def RQ_12_13_FIGURE(a_tilde_values, p99_vals, M_convergence):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 6),
        dpi=300,
        gridspec_kw={"width_ratios": [8, 4]}
    )

    # --- Panel 1 ---
    ax = axes[0]
    p99_sim = [item["p99_sim"] for item in p99_vals]
    p99_ana = [item["p99_ana"] for item in p99_vals]

    ax.plot(a_tilde_values, p99_sim, label=r"Simulation", color="blue")
    ax.scatter(a_tilde_values, p99_ana, label=r"Analytical ($M=100$)", color="red")

    ax.set_title(r"$P_{99}$ vs.\ $\tilde{b}$" + "\n" + r"Analytical vs Simulation")
    ax.set_xlabel("(a) " + r"$\tilde{a}$")
    ax.set_ylabel(r"$P_{99}(\tilde{b})$")
    ax.grid(True)

    leg = ax.legend(bbox_to_anchor=(0.1, 0.92), frameon=True, fontsize=11)
    leg.get_frame().set_alpha(0.95)

    # Shading region x <= 0.16
    ymin, ymax = ax.get_ylim()
    y_text = ymin + 0.10 * (ymax - ymin)

    ax.axvspan(0, 0.16, color="gray", alpha=0.4, zorder=0)
    ax.annotate(r"$e > 10\%$", xy=(0.16, y_text),
                xytext=(0.30, y_text - 0.5),
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    # -------- inset --------
    ax_inset = inset_axes(
        ax, width="45%", height="45%",
        bbox_to_anchor=(-0.12, -0.17, 1.1, 1.1),
        bbox_transform=ax.transAxes,
        loc="upper right"
    )

    ax_inset.plot(a_tilde_values, p99_sim, color="blue")
    ax_inset.scatter(a_tilde_values, p99_ana, color="red")

    ax_inset.set_yscale("log")
    ax_inset.grid(True, linestyle="--", linewidth=0.5)
    ax_inset.set_title("log-scale", fontsize=11)
    ax_inset.set_xlabel(r"$\tilde{a}$", fontsize=10)
    ax_inset.set_ylabel(r"$\log(P_{99}(\tilde{b}))$", fontsize=10)
    ax_inset.tick_params(axis='both', labelsize=9)

    # --- Panel 2 ---

    ax2 = axes[1]
    M_conv = np.asarray([item["M"] for item in M_convergence])

    idx_M10 = np.argmax(M_conv == 10)
    idx_M5 = np.argmax(M_conv == 5)
    idx_M2 = np.argmax(M_conv == 2)

    a_tilde_M10 = M_convergence[idx_M10]["a_tilde"]
    a_tilde_M5  = M_convergence[idx_M5]["a_tilde"]
    a_tilde_M2  = M_convergence[idx_M2]["a_tilde"]

    ax2.plot(a_tilde_values, M_conv, marker="o", color="k")

    # annotations
    c = 0.5; cy = 5
    ax2.annotate(r"$M=10$", xy=(a_tilde_M10, 10),
                xytext=(a_tilde_M10 + c, 10 + cy + 4),
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    ax2.annotate(r"$M=5$", xy=(a_tilde_M5, 5),
                xytext=(a_tilde_M5 + c, 5 + cy - 0.5),
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    ax2.annotate(r"$M=2$", xy=(a_tilde_M2, 2),
                xytext=(a_tilde_M2 + c, 2 + cy - 3),
                fontsize=12, arrowprops=dict(arrowstyle="->"))

    ax2.set_title(
        r"$M$ required for convergence of $\tilde{g}(\tilde{b})$" + "\n" + r"at $D_{L^1} < 0.05$"
    )
    ax2.set_xlabel("(b) " + r"$\tilde{a}$")
    ax2.set_ylabel(r"$M$")
    ax2.set_yscale("log")
    ax2.set_xticks(np.linspace(min(a_tilde_values), 2.5, 5))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.grid(True)
    ax2.set_xlim(min(a_tilde_values) - 0.1, 2.5)

    plt.tight_layout()
    plt.savefig("Figures/RQ1.2-RQ1.3_figure.pdf", dpi=300, bbox_inches="tight")
    plt.show()



# ======= RQ 1.4.1 - P99 values for different a_tilde = a1 beta1 = a2 beta2 ...  =============
# --- Are results only dependent on a_tilde or they depend on a and beta independently ---
# ============================================================================================
def RQ_141_FIGURE(a_tilde_values, p99_simVals, p99_aTilde_Vals):
    # --- P99 values ---
    a_tilde_ana = np.asarray([item["A_tilde"][0] for item in p99_aTilde_Vals])
    p99_ana1 = np.asarray([item["p99"][0] for item in p99_aTilde_Vals])
    p99_ana2 = np.asarray([item["p99"][1] for item in p99_aTilde_Vals])
    p99_ana3 = np.asarray([item["p99"][2] for item in p99_aTilde_Vals])

    # --- a and beta values ---
    a_case1 = np.array([item["a"][0] for item in p99_aTilde_Vals]); 
    beta_case1 = np.array([item["beta"][0] for item in p99_aTilde_Vals])

    c = 3
    a_tilde_iso = beta_case1[::c] * (a_case1[::c])

    # --- P99 FIGURE ---
    plt.figure(figsize=[8,6])
    plt.plot(a_tilde_values, p99_simVals, label=r"Simulation $P_{99}$", color="k", linewidth=1.5)
    plt.scatter(a_tilde_ana, p99_ana1, label=r"Analytical $P_{99}$: $\tilde{a}(a_1, \beta_1)$", marker="o", s=80, facecolors='none', edgecolors='b', zorder=2)
    plt.scatter(a_tilde_ana, p99_ana2, label=r"Analytical $P_{99}$: $\tilde{a}(a_2, \beta_2)$", marker="s", c="g", zorder=2)
    plt.scatter(a_tilde_ana, p99_ana3, label=r"Analytical $P_{99}$: $\tilde{a}(a_3, \beta_3)$", marker="o", c="r", s=11, zorder=2)
    for i in range(len(a_tilde_iso)):
        plt.axvline(a_tilde_iso[i], color="k", alpha=0.4)
    plt.title(r"$P99(\tilde{b})$ vs $\tilde{a}(a_i, \beta_i)$")
    plt.xlabel(r"$\tilde{a}$")
    plt.ylabel(r"$P_{99}(\tilde{b})$")
    plt.xlim([0.0, 3.3])
    plt.legend(loc="upper right")
    plt.show()



# ======= RQ 1.4.2 - P99 values for different a_tilde = a1 beta1 = a2 beta2 ...  =============
# --- Are results only dependent on a_tilde or they depend on a and beta independently ---
# ============================================================================================

def RQ_142_FIGURE(p99_aTilde_Vals):
    # --- a AND beta EVOLUTION ---
    a_case1 = np.array([item["a"][0] for item in p99_aTilde_Vals]); beta_case1 = np.array([item["beta"][0] for item in p99_aTilde_Vals])
    a_case2 = np.array([item["a"][1] for item in p99_aTilde_Vals]); beta_case2 = np.array([item["beta"][1] for item in p99_aTilde_Vals])
    a_case3 = np.array([item["a"][2] for item in p99_aTilde_Vals]); beta_case3 = np.array([item["beta"][2] for item in p99_aTilde_Vals])

    c = 3
    a_tilde_iso = beta_case1[::c] * (a_case1[::c])
    a_iso = np.linspace(0.0000001, 10, 500)

    # --- a ~ beta RELATION ---
    iso_beta_curves = np.zeros([len(a_tilde_iso), len(a_iso)])
    for i, a_tilde_val in enumerate(a_tilde_iso):
        beta_iso = a_tilde_val / a_iso
        iso_beta_curves[i,:] = beta_iso

    x_marker = np.array([item["a"][3] for item in p99_aTilde_Vals]); 
    y_marker = np.array([item["beta"][3] for item in p99_aTilde_Vals])

    x_marker = x_marker[::3]; #x_marker += 0.02  
    x_marker[0] -= 0.015;  #x_marker[1] = 0.63; x_marker[2] += -0.005 
    y_marker = y_marker[::3]; #y_marker += 0.01  
    y_marker[0] += 0.03; #y_marker[1] = 0.61; y_marker[2] += 0.012

    plt.figure(figsize=[8,6])
    # --- Case 1 ---
    plt.scatter(a_case1[::c], beta_case1[::c],
                label=r"$\tilde{a}(a_1, \beta_1)$", marker="o", s=80,
                facecolors='none', edgecolors='blue')
    plt.plot(a_case1, beta_case1, color="b", linestyle="--")

    # --- Case 2 ---
    plt.scatter(a_case2[::c], beta_case2[::c],
                label=r"$\tilde{a}(a_2, \beta_2)$", marker="s", c="g")
    plt.plot(a_case2, beta_case2, color="g", linestyle="--")

    # --- Case 3 ---
    plt.scatter(a_case3[::c], beta_case3[::c],
                label=r"$\tilde{a}(a_3, \beta_3)$", marker="o", c="r", s=20)
    plt.plot(a_case3, beta_case3, color="r", linestyle="--")

    # --- Curvas iso-beta ---
    for i in range(len(a_tilde_iso)-1):
        plt.plot(a_iso, iso_beta_curves[i,:], color="k", alpha=0.4)

    plt.plot(a_iso, iso_beta_curves[-1,:], color="k", alpha=0.4,
            label=r"Constant $\tilde{a}$")

    plt.xlim([min(a_case1)-0.2, max(a_case3)+0.2])
    plt.ylim([min(beta_case3)-0.05, max(beta_case1)+0.05])

    # --- Etiquetas en lugar de scatter ---
    for i in range(len(x_marker)-2):
        plt.text(
            x_marker[i],
            y_marker[i],
            rf"$\tilde{{a}} = {a_tilde_iso[i]:.2f}$",
            fontsize=9.5,          # Fuente más pequeña
            color="k",
            ha="left",
            va="bottom"
        )

    plt.title(r"$a$ vs. $\beta$ for constant $\tilde{a}(a_i, \beta_i)$")
    plt.xlabel(r"$a\,[\mathrm{MW}]$")
    plt.ylabel(r"$\beta\,[1/\mathrm{MW}]$")
    plt.grid(True)
    plt.legend(loc="lower right")
    plt.show()

# ======= FIGURE 3) RQ 1.4.1 and RQ 1.4.2 Combined ====================

def RQ_141_142_FIGURE(a_tilde_values, p99_simVals, p99_aTilde_Vals):
    # --- P99 Figure ---
    a_tilde_ana = np.asarray([item["A_tilde"][0] for item in p99_aTilde_Vals])
    p99_ana1 = np.asarray([item["p99"][0] for item in p99_aTilde_Vals])
    p99_ana2 = np.asarray([item["p99"][1] for item in p99_aTilde_Vals])
    p99_ana3 = np.asarray([item["p99"][2] for item in p99_aTilde_Vals])

    # --- a AND beta EVOLUTION ---
    a_case1 = np.array([item["a"][0] for item in p99_aTilde_Vals]); beta_case1 = np.array([item["beta"][0] for item in p99_aTilde_Vals])
    a_case2 = np.array([item["a"][1] for item in p99_aTilde_Vals]); beta_case2 = np.array([item["beta"][1] for item in p99_aTilde_Vals])
    a_case3 = np.array([item["a"][2] for item in p99_aTilde_Vals]); beta_case3 = np.array([item["beta"][2] for item in p99_aTilde_Vals])

    c = 3
    a_tilde_iso = beta_case1[::c] * (a_case1[::c])
    a_iso = np.linspace(0.0000001, 10, 500)

    # --- a ~ beta RELATION ---
    iso_beta_curves = np.zeros([len(a_tilde_iso), len(a_iso)])
    for i, a_tilde_val in enumerate(a_tilde_iso):
        beta_iso = a_tilde_val / a_iso
        iso_beta_curves[i,:] = beta_iso

    x_marker = np.array([item["a"][3] for item in p99_aTilde_Vals]); 
    y_marker = np.array([item["beta"][3] for item in p99_aTilde_Vals])

    x_marker = x_marker[::3]; x_marker[0] -= 0.015 
    y_marker = y_marker[::3]; y_marker[0] += 0.03

    # Figura con dos subplots, pero anchos distintos: [6, 8]
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 6),        # altura 6, ancho total acorde a 6:8
        dpi=300,
        gridspec_kw={"width_ratios": [5.5, 8]}
    )

    # --------  P99 FIGURE (Panel 1, ratio ~6x6) --------
    ax = axes[0]

    ax.plot(a_tilde_values, p99_simVals,
            label=r"Simulation $P_{99}$", color="k", linewidth=1.5)

    ax.scatter(a_tilde_ana, p99_ana1,
            label=r"Analytical $P_{99}$: $\tilde{a}(a_1,\beta_1)$",
            marker="o", s=80, facecolors='none', edgecolors='b', zorder=2)

    ax.scatter(a_tilde_ana, p99_ana2,
            label=r"Analytical $P_{99}$: $\tilde{a}(a_2,\beta_2)$",
            marker="s", c="g", zorder=2)

    ax.scatter(a_tilde_ana, p99_ana3,
            label=r"Analytical $P_{99}$: $\tilde{a}(a_3,\beta_3)$",
            marker="o", c="r", s=11, zorder=2)

    for i in range(len(a_tilde_iso)):
        ax.axvline(a_tilde_iso[i], color="k", alpha=0.4)

    ax.set_title(r"$P_{99}(\tilde{a})$ vs. $\tilde{a}(a_i,\beta_i)$")
    ax.set_xlabel("(a) " + r"$\tilde{a}$")
    ax.set_ylabel(r"$P_{99}(\tilde{a})$")
    ax.set_xlim([0.0, 3.2])
    ax.legend(loc="upper right")

    # --------  a ~ beta RELATION (Panel 2, ratio ~8x6) --------
    ax = axes[1]

    # --- Case 1 ---
    ax.scatter(a_case1[::c], beta_case1[::c],
            label=r"$\tilde{a}(a_1,\beta_1)$",
            marker="o", s=80, facecolors='none', edgecolors='blue')
    ax.plot(a_case1, beta_case1, color="b", linestyle="--")

    # --- Case 2 ---
    ax.scatter(a_case2[::c], beta_case2[::c],
            label=r"$\tilde{a}(a_2,\beta_2)$",
            marker="s", c="g")
    ax.plot(a_case2, beta_case2, color="g", linestyle="--")

    # --- Case 3 ---
    ax.scatter(a_case3[::c], beta_case3[::c],
            label=r"$\tilde{a}(a_3,\beta_3)$",
            marker="o", c="r", s=20)
    ax.plot(a_case3, beta_case3, color="r", linestyle="--")

    # --- iso-beta curves ---
    for i in range(len(a_tilde_iso)-1):
        ax.plot(a_iso, iso_beta_curves[i, :], color="k", alpha=0.4)

    ax.plot(a_iso, iso_beta_curves[-1, :], color="k", alpha=0.4,
            label=r"Constant $\tilde{a}$")

    # --- Etiquetas texto ---
    for i in range(len(x_marker)-1):
        ax.text(
            x_marker[i],
            y_marker[i],
            rf"$\tilde{{a}}={a_tilde_iso[i]:.2f}$",
            fontsize=10,
            color="k",
            ha="left",
            va="bottom"
        )

    ax.set_title(r"$a$ vs.\ $\beta$ for constant $\tilde{a}(a_i,\beta_i)$")
    ax.set_xlabel("(b) " + r"$a\,[\mathrm{MW}]$")
    ax.set_ylabel(r"$\beta\,[1/\mathrm{MW}]$")
    ax.set_xlim([min(a_case1)-0.2, max(a_case3)+0.25])
    ax.set_ylim([min(beta_case3)-0.05, max(beta_case1)+0.05])
    ax.legend(loc="lower right")

    plt.tight_layout()

    # --- Guardar PDF ---
    output_path = "Figures/RQ1.4_figure.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()




# ======= RQ 2.1 - Nystrom vs Analytical pdf/cdf comparison ====================
# --- Consistency between analytical and numerical solutions ---
# ==============================================================================
def RQ_21_FIGURE(res_sim, res_nys, res_ana):

    b_sim, g_sim, _, p99_sim, _ = res_sim
    b_ana, g_ana, p0_ana, p99_ana, _ = res_ana
    b_nys, g_nys, p0_nys, p99_nys, _ = res_nys

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # --- Curvas PDF ---
    ax.plot(
        b_sim, g_sim,
        label=r"Empirical PDF (synthetic data)",
        zorder=1,
        alpha=0.7,
        linewidth=1,
        color="k"
    )

    # --- Analytical PDF ---
    g_ana = g_ana / (1 - p0_ana)
    c_step = 20   # salto visual
    ax.scatter(
        b_ana[::c_step], g_ana[::c_step],
        label=r"Analytical",
        marker="o",
        s=100,
        facecolors="none",
        edgecolors="b",
        zorder=2
    )

    # --- Nystrom PDF ---
    g_nys = g_nys / (1 - p0_nys)
    ax.scatter(
        b_nys[::c_step], g_nys[::c_step],
        label=r"Nystrom",
        marker="s",
        c="g",
        zorder=2
    )

    # --- Líneas verticales P99 ---
    colors_p99 = ["k", "b", "g"]
    p99_vals = [p99_sim, p99_ana, p99_nys]
    labels_p99 = [
        r"$P_{99}^{\mathrm{sim}, P_0 \approx \infty}$",
        r"$P_{99}^{\mathrm{ana}}$",
        r"$P_{99}^{\mathrm{nys}}$"
    ]

    offset = 0.03
    linestyles = ["--", ":", "-."]

    for i, (col, p99, ls) in enumerate(zip(colors_p99, p99_vals, linestyles)):
        ax.axvline(
            x=p99 + i * offset,
            linestyle=ls,
            color=col,
            linewidth=1.5
        )

    # --- Escalas y etiquetas ---
    ax.set_xlabel(r"$\tilde{b}$")
    ax.set_ylabel(r"$\log\big(\tilde{g}(\tilde{b})\big)$")
    ax.set_title(r"Numerical and Analytical solvers consistency")

    ax.set_xlim([-0.2, 10.5])
    ax.set_ylim([1e-4, 1.1])
    ax.set_yscale("log")
    ax.grid(zorder=0)

    # ========= Leyenda 1 (PDFs) =========
    handles_main = [
        Line2D([0], [0], color="k", lw=0.8, label=r"Empirical PDF"),
        Line2D([0], [0], marker="o", markersize=8, linestyle="None",
            markerfacecolor="none", markeredgecolor="b",
            label=r"Analytical (M=100)"),
        Line2D([0], [0], marker="s", markersize=8, linestyle="None",
            markerfacecolor="g", markeredgecolor="g",
            label=r"Nystrom")
    ]

    leg1 = ax.legend(handles=handles_main, loc="upper right", frameon=True)
    leg1.get_frame().set_facecolor("white")
    leg1.get_frame().set_alpha(0.95)
    ax.add_artist(leg1)

    # ========= Leyenda 2 (P99) =========
    handles_p99 = [
        Line2D([0], [0], color=col, linestyle=ls, linewidth=1.5,
            label=rf"{lab} = {val:.2f}")
        for col, val, lab, ls in zip(colors_p99, p99_vals, labels_p99, linestyles)
    ]

    leg2 = ax.legend(handles=handles_p99,
                    loc="lower left",
                    frameon=True,
                    fontsize=13)
    leg2.get_frame().set_facecolor("white")
    leg2.get_frame().set_alpha(0.95)

    plt.tight_layout()

    # Guardar PDF
    output_path = "Figures/RQ2.1_single_plot.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()


# ======= RQ 2.2 - Computing time of analytical solution against M =============
# --- O(M) curve for analytical solution ---
# =============================================================================
def RQ_22_FIGURE(M_values, time_vs_M):
    fig, ax2 = plt.subplots(figsize=(6, 5), dpi=300)

    # Extraer tiempos (ya en tu estructura)
    time_ana_values = np.asarray([item["time"] for item in time_vs_M])

    # Curva tiempo vs M
    ax2.plot(
        M_values, time_ana_values,
        marker="o",
        color="k",
        linewidth=1,
        markersize=5,
        alpha=0.9,
    )

    # Etiquetas y título
    ax2.set_xlabel("M")
    ax2.set_ylabel("time [s]")
    ax2.set_title(r"$\tilde{g}(\tilde{b})^{ana}$ computing time vs M")

    # Cuadrícula y límites
    ax2.grid(True, zorder=0)
    ax2.set_xlim([min(M_values) - 2, max(M_values) + 2])

    plt.tight_layout()

    # Guardar figura
    output_path = "Figures/RQ2.2_single_plot.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()



# ======= FIGURE 4) RQ 2.1 and RQ 2.2 Combined Figure ==============
def RQ_21_22_FIGURE(res_sim, res_nys, res_ana, M_values, time_vs_M):
    b_sim, g_sim, _, p99_sim, _ = res_sim
    b_ana, g_ana, p0_ana, p99_ana, _ = res_ana
    b_nys, g_nys, p0_nys, p99_nys, _ = res_nys

    fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 6),
    dpi=300,
    gridspec_kw={"width_ratios": [9, 3]}
    )
    plt.subplots_adjust(wspace=0.35)

    # --- Panel 1 ---
    ax = axes[0]

    # --- Curvas PDF ---
    ax.plot(
        b_sim, g_sim,
        label=r"Empirical PDF",
        zorder=1,
        alpha=0.7,
        linewidth=1,
        color="k"
    )

    g_ana = g_ana/(1 - p0_ana)
    c = 20
    ax.scatter(
        b_ana[::c], g_ana[::c],
        label=r"Analytical",
        marker="o",
        s=100,
        facecolors='none',
        edgecolors='b',
        zorder=2
    )

    g_nys = g_nys/(1-p0_nys)
    ax.scatter(
        b_nys[::c], g_nys[::c],
        label=r"Nystrom",
        marker="s",
        c="g",
        zorder=2
    )

    # --- Líneas verticales P99 ---
    colors_p99 = ["k", "b", "g"]
    p99_vals = [p99_sim, p99_ana, p99_nys]
    labels_p99 = [r"$P_{99}^{\mathrm{sim}, P_0 \approx \infty}$",
                r"$P_{99}^{\mathrm{ana}}$",
                r"$P_{99}^{\mathrm{nys}}$"]

    offset = 0.03   # puedes ajustar a 0.01, 0.05, según la escala
    for i, (c, x) in enumerate(zip(colors_p99, p99_vals)):
        ax.axvline(x=x+i*offset, linestyle="--", color=c, linewidth=1.5)

    # Escalas y etiquetas
    ax.set_xlabel("(a) " + r"$\tilde{b}$")
    ax.set_ylabel(r"$\log\big(\tilde{g}(\tilde{b})\big)$")
    ax.set_title(r"Numerical and Analytical solvers consistency")
    ax.set_xlim([-0.2, 10.5])
    ax.set_ylim([1e-4, 1.1])
    ax.set_yscale("log")
    ax.grid(zorder=0)

    # ======= Leyenda 1 (PDFs) ========
    handles_main = [
        Line2D([0], [0], color="k", lw=0.8, label=r"Empirical PDF (synthetic data)"),
        Line2D([0], [0], marker="o", markersize=8, linestyle="None",
            markerfacecolor="none", markeredgecolor="b",
            label=r"Analytical (M=100)"),
        Line2D([0], [0], marker="s", markersize=8, linestyle="None",
            markerfacecolor="g", markeredgecolor="g",
            label=r"Nystrom")
    ]

    leg1 = ax.legend(handles=handles_main, loc="upper right", frameon=True, fontsize=14)
    leg1.get_frame().set_facecolor('white')
    leg1.get_frame().set_alpha(0.95)
    ax.add_artist(leg1)

    # ======= Leyenda 2 (P99) ========
    handles_p99 = []
    linestyle_cases = ["--", ":", "-."]
    for i, (c, x, lab) in enumerate(zip(colors_p99, p99_vals, labels_p99)):
        h = Line2D(
            [0], [0], color=c, linestyle=linestyle_cases[i], linewidth=1.5,
            label=rf"{lab} = {x:.2f}"
        )
        handles_p99.append(h)

    leg2 = ax.legend(handles=handles_p99,
                    loc="lower left",
                    frameon=True,
                    fontsize=13)
    leg2.get_frame().set_facecolor('white')
    leg2.get_frame().set_alpha(0.95)



    # --- Panel 2 ---
    ax2 = axes[1]

    time_ana_values = np.asarray([item["time"] for item in time_vs_M])

    ax2.plot(
        M_values, time_ana_values,
        marker="o",
        color="k",
        linewidth=1,
        markersize=5,
        alpha=0.9
    )

    ax2.set_xlabel("(b) M")
    ax2.set_ylabel("time [s]")
    ax2.set_title(r"$\tilde{g}(\tilde{b})^{ana}$ computing time vs M")
    ax2.grid(True, zorder=0)

    # Limpiar rango por estética
    ax2.set_xlim([min(M_values) - 2, max(M_values) + 2])

    plt.tight_layout()

    # Guardar PDF
    output_path = "Figures/RQ2.1-RQ2.2_figure.pdf"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()



# ======= RQ 3.1 - P99 curve comparison for SL vs GL ====================
# --- Impact of GL distribution for algorithmic solution ---
# =======================================================================

def RQ_31_FIGURE(a_tilde_values, p99_SL_vec, p99_GL_vec):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # ----- Figura principal -----
    ax.plot(a_tilde_values, p99_SL_vec,
            label=r"$P_{99}^{SL, sim}$",
            color="blue", marker="o")
    ax.plot(a_tilde_values, p99_GL_vec,
            label=r"$P_{99}^{GL, sim}$",
            color="red", marker="x")

    ax.set_title(r"$P_{99}$ vs.\ $\tilde{a}$" "\n" r"SL vs.\ GL Comparison")
    ax.set_xlabel(r"$\tilde{a}$")
    ax.set_ylabel(r"$P_{99}(\tilde{b})$")
    ax.grid(True)

    leg1 = ax.legend(
        bbox_to_anchor=(0.12, 0.92),
        frameon=True,
    )
    leg1.get_frame().set_facecolor("white")
    leg1.get_frame().set_alpha(0.95)

    # ----- Subfigura (inset) arriba-derecha -----
    ax_inset = inset_axes(
        ax,
        width="45%", height="45%",
        bbox_to_anchor=(-0.12, -0.17, 1.1, 1.1),
        bbox_transform=ax.transAxes,
        loc="upper right"
    )

    ax_inset.plot(a_tilde_values, p99_SL_vec, color="blue")
    ax_inset.plot(a_tilde_values, p99_GL_vec, color="red")

    ax_inset.set_yscale("log")
    ax_inset.grid(True, which="both", linestyle="--", linewidth=0.6)
    ax_inset.set_title(r"log-scale", fontsize=12)

    ax_inset.tick_params(axis='both', which='major', labelsize=10)
    ax_inset.set_ylabel(r"$\log\!\big(P_{99}\big)$", fontsize=11)
    ax_inset.set_xlabel(r"$\tilde{a}$", fontsize=11)

    plt.tight_layout()

    plt.savefig("Figures/RQ3.1_figure.pdf", 
                dpi=300, bbox_inches="tight")
    plt.show()



# ======= RQ 3.2 - L1 norm agreement for SL and GL ====================
# --- Impact of a_tilde in the agreement between SL and GL---
# =====================================================================
def RQ_32_FIGURE(a_tilde_values, L1_norm):
    fig, ax2 = plt.subplots(figsize=(6, 5), dpi=300)

    ax2.plot(
        a_tilde_values,
        L1_norm,
        color="k",
        linewidth=2,
        marker="o",
        label=r"$D_{L^1}$"
    )

    ax2.set_title(r"$D_{L^1}$ vs.\ $\tilde{a}$")
    ax2.set_xlabel(r"$\tilde{a}$")
    ax2.set_ylabel(r"$D_{L^1}$")
    ax2.grid(True, alpha=0.6)

    plt.tight_layout()

    # Guardar la figura individual
    plt.savefig("Figures/RQ3.2_single_plot.pdf", dpi=300, bbox_inches="tight")
    plt.show()



# ======= FIGURE 5) RQ 3.2 and RQ 3.1 combined figure ====================
def RQ_31_32_FIGURE(a_tilde_values, p99_SL_vec, p99_GL_vec, L1_norm):

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=300,
                             gridspec_kw={"width_ratios": [8, 4]})

    # ==========================================================
    # =========== SUBFIGURA 1: P99 SL vs GL + INSET ============
    # ==========================================================
    ax = axes[0]

    ax.plot(a_tilde_values, p99_SL_vec,
            label=r"$P_{99}^{SL, sim}$",
            color="blue", marker="o")
    ax.plot(a_tilde_values, p99_GL_vec,
            label=r"$P_{99}^{GL, sim}$",
            color="red", marker="x")

    ax.set_title(r"$P_{99}$ vs.\ $\tilde{a}$" "\n" r"SL vs.\ GL Comparison")
    ax.set_xlabel("(a) " + r"$\tilde{a}$")
    ax.set_ylabel(r"$P_{99}(\tilde{b})$")
    ax.grid(True)

    leg1 = ax.legend(bbox_to_anchor=(0.12, 0.92), frameon=True)
    leg1.get_frame().set_facecolor("white")
    leg1.get_frame().set_alpha(0.95)

    # ----- Inset interno -----
    ax_inset = inset_axes(
        ax,
        width="45%", height="45%",
        loc="upper right",
        bbox_to_anchor=(-0.12, -0.17, 1.1, 1.1),
        bbox_transform=ax.transAxes
    )

    ax_inset.plot(a_tilde_values, p99_SL_vec, color="blue")
    ax_inset.plot(a_tilde_values, p99_GL_vec, color="red")

    # ======== GRID ESTÉTICO REGULAR (HORIZONTAL + VERTICAL) ========
    ax_inset.set_yscale("log")
    ax_inset.grid(False)  # apagar grid por defecto

    # Limites actuales
    y_min, y_max = ax_inset.get_ylim()
    x_min, x_max = ax_inset.get_xlim()

    # Número de líneas (ajustable)
    N_lines = 6

    # ---- Líneas horizontales (uniformes en pantalla) ----
    for frac in np.linspace(0, 1, N_lines):
        y_val = 10 ** (np.log10(y_min) + frac * (np.log10(y_max) - np.log10(y_min)))
        ax_inset.axhline(y_val, color="gray", alpha=0.35, linewidth=0.6, zorder=0)

    # ---- Líneas verticales (uniformes en pantalla) ----
    for frac in np.linspace(0, 1, N_lines):
        x_val = x_min + frac * (x_max - x_min)
        ax_inset.axvline(x_val, color="gray", alpha=0.35, linewidth=0.6, zorder=0)
    # ================================================================

    ax_inset.set_title(r"log-scale", fontsize=12)
    ax_inset.tick_params(axis='both', which='major', labelsize=10)
    ax_inset.set_ylabel(r"$\log\!\big(P_{99}(\tilde{b})\big)$", fontsize=11)
    ax_inset.set_xlabel(r"$\tilde{a}$", fontsize=11)

    # ==========================================================
    # =========== SUBFIGURA 2: Norma L1 vs a ===================
    # ==========================================================
    ax2 = axes[1]

    ax2.plot(
        a_tilde_values,
        L1_norm,
        color="k",
        linewidth=2,
        marker="o",
        label=r"$D_{L^1}$"
    )

    ax2.set_title(r"$D_{L^1}$ vs.\ $\tilde{a}$")
    ax2.set_xlabel("(b) " + r"$\tilde{a}$")
    ax2.set_ylabel(
        r"$\left\|\tilde{g}(\tilde{b})^{(sim)}_{SL} "
        r"- \tilde{g}(\tilde{b})^{(sim)}_{GL}\right\|_{L^1}$",
        fontsize=15)
    ax2.grid(True, alpha=0.6)

    plt.tight_layout()
    plt.savefig("Figures/RQ3.1-RQ3.2_figure.pdf", dpi=300, bbox_inches="tight")
    plt.show()



# ======= FIGURE 6) RQ 3.3 - p99 curve for GL for different c values ====================
# --- Influence of c on the GL distribution ---
# =======================================================================================
def RQ_33_FIGURE(a_tilde_values, p99_GL_c_vs_a, p99_ana_values, c_values):
    colors = ["b", "g", "r"]
    indices = [0, 1, 2]

    fig, ax = plt.subplots(figsize=(8,6), dpi=300)

    # --------- Figura principal (lineal) ---------
    for j in range(3):
        c = round(c_values[j], 2)
        ax.plot(a_tilde_values,
                p99_GL_c_vs_a[j, :],
                color=colors[j],
                label=fr"$P_{{99}}^{{GL,\,sim}}\,(a_1={c})$")

    ax.plot(a_tilde_values, p99_ana_values,
            linestyle="--", linewidth=2, color="k",
            label=r"$P_{99}^{SL, ana}(M=100)$")

    ax.set_title(r"$P_{99}$ vs.\ $\tilde{a}$" "\n" r"Comparison of $P_{{99}}^{GL}$ simulation and $P_{{99}}^{SL}$ analytical solution")
    ax.set_xlabel(r"$\tilde{a}$")
    ax.set_ylabel(r"$P_{99}(\tilde{b})$")
    ax.grid(True)

    leg = ax.legend(bbox_to_anchor=(0.09, 0.97), frameon=True, fontsize=11)
    leg.get_frame().set_alpha(0.95)


    # --------- Subfigura (log-scale) ---------
    ax_inset = inset_axes(
        ax, width="45%", height="45%",
        bbox_to_anchor=(-0.12, -0.17, 1.1, 1.1),
        bbox_transform=ax.transAxes,
        loc="upper right"
    )

    for j, idx in enumerate(indices):
        ax_inset.plot(a_tilde_values,
                    p99_GL_c_vs_a[idx, :],
                    color=colors[j])

    ax_inset.plot(a_tilde_values, p99_ana_values,
                linestyle="--", linewidth=2, color="k")

    ax_inset.set_yscale("log")
    ax_inset.grid(True, linestyle="--", linewidth=0.6)
    ax_inset.set_title(r"log-scale", fontsize=12)

    ax_inset.set_xlabel(r"$\tilde{a}$", fontsize=11)
    ax_inset.set_ylabel(r"$\log(P_{99}(\tilde{b}))$", fontsize=11)
    ax_inset.tick_params(axis="both", labelsize=10)

    plt.tight_layout()

    plt.savefig("Figures/RQ3.3_figure.pdf",
                dpi=300, bbox_inches="tight")
    plt.show()


# ======= FIGURE 6) RQ 4.1 - PDF and CDF comparison against real data ====================
# --- Comparison between SL and GL simulation and  Neumann solution ---
# ========================================================================================
def RQ_411_FIGURE(b_grid, g_VKU, solutions_list):
    
    labels=("SL Simulation", "GL Simulation", "Neumann solution")
    colors=('r', "b", 'lightgreen')

    fig, ax_pdf = plt.subplots(figsize=(9, 6), dpi=300)

    # --- PDF curves (simulations and analytical) ---
    for i, g in enumerate(solutions_list):
        ax_pdf.plot(
            b_grid,
            g,
            linewidth=3,
            color=colors[i],
            label=labels[i]
        )

    # --- Real data PDF ---
    ax_pdf.plot(
        b_grid,
        g_VKU,
        color='k',
        linewidth=2,
        label="Real data (PDF)"
    )

    # --- Log scale + cosmetics ---
    ax_pdf.set_yscale("log")
    ax_pdf.set_ylim([1e-6, 1])
    ax_pdf.set_xlabel(r"$\tilde{b}$")
    ax_pdf.set_ylabel(r"$\log\!\big(\tilde{g}(\tilde{b})\big)$")
    ax_pdf.set_xlim(0, 20)
    ax_pdf.grid(True)

    # --- Build legend manually (curves + real data) ---
    handles_pdf = [
        Line2D([0], [0], color=colors[i], linewidth=3, label=labels[i])
        for i in range(len(solutions_list))
    ]

    handles_pdf.append(
        Line2D([0], [0], color='k', linewidth=1.1, label="Real data (PDF)")
    )

    leg_pdf = ax_pdf.legend(handles=handles_pdf, loc="lower left", frameon=True)
    frame_pdf = leg_pdf.get_frame()
    frame_pdf.set_facecolor('white')
    frame_pdf.set_alpha(0.8)

    ax_pdf.set_title(r"PDF Comparison between real data and integral solutions")

    plt.tight_layout()

    # Save if a path is provided
    plt.savefig("Figures/RQ_4.1.1_figure.pdf", dpi=300, bbox_inches='tight')

    return fig, ax_pdf

def RQ_412_FIGURE(b_grid, G_solutions, p99_values):    
    # --- G solutions ---
    G_VKU, G_SL_sim, G_GL_sim, G_ana = G_solutions

    # Colors and labels
    p99_labels=("VKU", "SL", "GL", "ana")
    colors       = ['r', "b", 'lightgreen']
    labels       = ["SL Simulation", "GL Simulation", "Neumann solution"]
    dark_orange  = "#f28500"
    colors_P99   = ["#f28500", 'r', "b", "lightgreen"]
    markers      = ["x", "^", "o"]
    step=8

    fig, ax_cdf = plt.subplots(figsize=(9, 6), dpi=300)

    # Real CDF curve
    ax_cdf.plot(
        b_grid,
        G_VKU,
        color=dark_orange,
        linewidth=2.8,
        label="Real data (CDF)"
    )

    # Scattered solver CDFs
    ax_cdf.scatter(
        take_every(b_grid, 0, step),
        take_every(G_SL_sim, 0, step),
        marker="x",
        color=colors[0],
        label="SL Simulation"
    )
    ax_cdf.scatter(
        take_every(b_grid, 2, step),
        take_every(G_GL_sim, 2, step),
        marker="^",
        color=colors[1],
        label="GL Simulation"
    )
    ax_cdf.scatter(
        take_every(b_grid, 5, step),
        take_every(G_ana, 5, step),
        marker="o",
        color=colors[2],
        label="Neumann solution"
    )

    # Axis formatting
    ax_cdf.set_xlabel(r"$\tilde{b}$")
    ax_cdf.set_ylabel(r"$\tilde{G}(\tilde{b})$")
    ax_cdf.set_xlim(0, 20)
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.grid(True)

    # p99 vertical lines
    for i, p99 in enumerate(p99_values):
        ax_cdf.axvline(
            x=p99,
            linestyle="--",
            linewidth=1.5,
            color=colors_P99[i]
        )

    # -------- First legend (CDF curves + scatters) --------
    handles_cdf = []

    handles_cdf.append(
        Line2D([0], [0],
               color=dark_orange,
               linewidth=2.8,
               label="Real data (CDF)")
    )

    for i, label_str in enumerate(labels):
        handles_cdf.append(
            Line2D([0], [0],
                   color=colors[i],
                   marker=markers[i],
                   markersize=7,
                   linewidth=0,
                   label=label_str)
        )

    leg1 = ax_cdf.legend(
        handles=handles_cdf,
        loc=(0.65, 0.02),
        frameon=True,
        fontsize=15
    )
    frame1 = leg1.get_frame()
    frame1.set_facecolor("white")
    frame1.set_alpha(0.85)

    # Keep legend 1 when adding legend 2
    ax_cdf.add_artist(leg1)

    # -------- Second legend (P99 values) --------
    handles_p99 = []
    for i, (p99, lab) in enumerate(zip(p99_values, p99_labels)):
        handles_p99.append(
            Line2D([0], [0],
                   linestyle="--",
                   linewidth=1.5,
                   color=colors_P99[i],
                   label=rf"$P_{{99}}^{{\mathrm{{{lab}}}}} = {p99:.2f}$")
        )

    leg2 = ax_cdf.legend(
        handles=handles_p99,
        loc=(0.7, 0.32),
        frameon=True,
        fontsize=15
    )
    frame2 = leg2.get_frame()
    frame2.set_facecolor("white")
    frame2.set_alpha(0.85)

    ax_cdf.set_title(r"CDF Comparison between real data and integral solutions")

    plt.tight_layout()

    plt.savefig("Figures/RQ_4.1.2_figure.pdf", dpi=300, bbox_inches='tight')
    return fig, ax_cdf


# ========== RQ 4.1 combined ========================
def RQ_41_FIGURE(b_grid, g_VKU, solutions_list, G_solutions, p99_values, savepath="Figures/RQ_4.1_figure.pdf"):
    """
    Combina las gráficas de PDF y CDF en una sola figura con arreglo [1,2].
    - b_grid: eje x
    - g_VKU: datos reales PDF
    - solutions_list: lista de soluciones para PDF
    - G_solutions: (G_VKU, G_SL_sim, G_GL_sim, G_ana) para CDF
    - p99_values: valores P99 para CDF
    - savepath: ruta para guardar la figura
    """

    # --- Configuración general ---
    labels = ("SL Simulation", "GL Simulation", "Neumann solution")
    colors = ('r', "b", 'lightgreen')

    # --- Crear figura con dos subplots ---
    fig, (ax_pdf, ax_cdf) = plt.subplots(1, 2, figsize=(12, 6), dpi=300)

    # ============================================================
    # Subplot 1: PDF
    # ============================================================
    for i, g in enumerate(solutions_list):
        ax_pdf.plot(b_grid, g, linewidth=3, color=colors[i], label=labels[i])

    ax_pdf.plot(b_grid, g_VKU, color='k', linewidth=2, label="Real data (PDF)")

    ax_pdf.set_yscale("log")
    ax_pdf.set_ylim([1e-6, 1])
    ax_pdf.set_xlim(0, 20)
    ax_pdf.set_xlabel("(a) "+r"$\tilde{b}$")
    ax_pdf.set_ylabel(r"$\log\!\big(\tilde{g}(\tilde{b})\big)$")
    ax_pdf.grid(True)
    ax_pdf.set_title("PDF Comparison")

    handles_pdf = [Line2D([0], [0], color=colors[i], linewidth=3, label=labels[i]) for i in range(len(solutions_list))]
    handles_pdf.append(Line2D([0], [0], color='k', linewidth=1.1, label="Real data (PDF)"))
    leg_pdf = ax_pdf.legend(handles=handles_pdf, loc="lower left", frameon=True)
    frame_pdf = leg_pdf.get_frame()
    frame_pdf.set_facecolor('white')
    frame_pdf.set_alpha(0.8)

    # ============================================================
    # Subplot 2: CDF
    # ============================================================
    G_VKU, G_SL_sim, G_GL_sim, G_ana = G_solutions
    p99_labels = ("VKU", "SL", "GL", "ana")
    dark_orange = "#f28500"
    colors_P99 = ["#f28500", 'r', "b", "lightgreen"]
    markers = ["x", "^", "o"]
    step = 8

    ax_cdf.plot(b_grid, G_VKU, color=dark_orange, linewidth=2.8, label="Real data (CDF)")
    ax_cdf.scatter(b_grid[0::step], G_SL_sim[0::step], marker="x", color=colors[0], label="SL Simulation")
    ax_cdf.scatter(b_grid[2::step], G_GL_sim[2::step], marker="^", color=colors[1], label="GL Simulation")
    ax_cdf.scatter(b_grid[5::step], G_ana[5::step], marker="o", color=colors[2], label="Neumann solution")

    ax_cdf.set_xlim(0, 20)
    ax_cdf.set_ylim(0, 1.05)
    ax_cdf.set_xlabel("(b) "+r"$\tilde{b}$")
    ax_cdf.set_ylabel(r"$\tilde{G}(\tilde{b})$")
    ax_cdf.grid(True)
    ax_cdf.set_title("CDF Comparison")

    # Líneas verticales P99
    for i, p99 in enumerate(p99_values):
        ax_cdf.axvline(x=p99, linestyle="--", linewidth=1.5, color=colors_P99[i])

    # Leyendas
    handles_cdf = [Line2D([0], [0], color=dark_orange, linewidth=2.8, label="Real data (CDF)")]
    for i, label_str in enumerate(labels):
        handles_cdf.append(Line2D([0], [0], color=colors[i], marker=markers[i], markersize=7, linewidth=0, label=label_str))
    leg1 = ax_cdf.legend(handles=handles_cdf, loc=(0.65, 0.02), frameon=True, fontsize=12)
    ax_cdf.add_artist(leg1)

    handles_p99 = [Line2D([0], [0], linestyle="--", linewidth=1.5, color=colors_P99[i],
                          label=rf"$P_{{99}}^{{\mathrm{{{lab}}}}} = {p99:.2f}$") for i, (p99, lab) in enumerate(zip(p99_values, p99_labels))]
    leg2 = ax_cdf.legend(handles=handles_p99, loc=(0.7, 0.32), frameon=True, fontsize=12)

    # ============================================================
    # Ajustes finales
    # ============================================================
    plt.tight_layout()
    plt.savefig(savepath, dpi=300, bbox_inches='tight')

    return fig, (ax_pdf, ax_cdf)


# ======= FIGURE 6) RQ 4.2 - P99 curveF comparison against real data ====================
# --- Comparison between SL and GL simulation and  Neumann solution ---
# ========================================================================================
def RQ_42_FIGURE(a_values, p99_VKU_curve, p99_SL_curve, p99_GL_curve, p99_ana_curve):

    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    # --------- Main figure (linear scale) ---------
    ax.plot(a_values, p99_VKU_curve,
            color="b",
            label=r"$P_{99}^{VKU,\;sim}$")

    ax.plot(a_values, p99_SL_curve,
            color="g",
            label=r"$P_{99}^{SL,\;sim}$")

    ax.plot(a_values, p99_GL_curve,
            color="r",
            label=r"$P_{99}^{GL,\;sim}$")

    ax.plot(a_values, p99_ana_curve,
            linestyle="--", linewidth=2, color="k", 
            label=r"$P_{99}^{SL,\;ana}(M=100)$")

    ax.set_title(
        r"$P_{99}$ vs.\ $\tilde{a}$" "\n"
        r"Comparison of simulated $P_{99}$ (VKU, SL, GL) and analytical $P_{99}^{SL}$"
    )
    ax.set_xlabel(r"$\tilde{a}$")
    ax.set_ylabel(r"$P_{99}(\tilde{b})$")
    ax.grid(True)

    leg = ax.legend(bbox_to_anchor=(0.1, 0.98), frameon=True, fontsize=11, loc="upper left")
    leg.get_frame().set_alpha(0.95)

    # --------- Inset (log-scale) ---------
    ax_inset = inset_axes(
        ax, width="45%", height="45%",
        bbox_to_anchor=(-0.12, -0.17, 1.1, 1.1),
        bbox_transform=ax.transAxes,
        loc="upper right"
    )

    ax_inset.plot(a_values, p99_VKU_curve, color="b")
    ax_inset.plot(a_values, p99_SL_curve, color="g")
    ax_inset.plot(a_values, p99_GL_curve, color="r")
    ax_inset.plot(a_values, p99_ana_curve, linestyle="--", linewidth=2, alpha=0.5, color="k")

    ax_inset.set_yscale("log")
    ax_inset.grid(True, linestyle="--", linewidth=0.6)
    ax_inset.set_title(r"log-scale", fontsize=12)

    ax_inset.set_xlabel(r"$\tilde{a}$", fontsize=11)
    ax_inset.set_ylabel(r"$\log(P_{99}(\tilde{b}))$", fontsize=11)
    ax_inset.tick_params(axis="both", labelsize=10)

    plt.tight_layout()

    plt.savefig("Figures/RQ_4.2_figure.pdf", dpi=300, bbox_inches="tight")
    plt.show()

