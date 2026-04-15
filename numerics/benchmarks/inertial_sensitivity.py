"""
End-to-end numerical sensitivity benchmark for the chip-scale quantum-comb
gyroscope at physical rotation rates.

Computes and plots:
  - Strategy A (classical CW/CCW beat per comb line, optionally with
    injected external squeezed light) σ_Ω(N_lines, S, T)
  - Strategy B (FWM-entangled differential phase, pure χ^(3))
    σ_Ω(N_lines, S, T)
  - Scaling vs device radius R, photons per line n, squeezing S

Uses the analytical Fisher-information formulas derived in qnm_inertial.tex
Sec 8.5 (corrected Sagnac formula + proper quantum advantage).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def sagnac_coupling(R, lam):
    """Classical Sagnac coupling m_p = R*omega/c = 2*pi*R/lam."""
    return 2*np.pi*R/lam


def strategy_A_SQL(m_p, N_lines, n_photon, T_sec):
    """Classical Sagnac beat, shot-noise-limited (no quantum enhancement).
    F = N * (2 m_p)^2 * n * T^2   (Fisher info for Ω from CW/CCW beat)
    sigma_Omega = 1/sqrt(F)
    """
    F = N_lines * (2*m_p)**2 * n_photon * T_sec**2
    sigma = 1/np.sqrt(F)
    return F, sigma


def strategy_A_squeezed(m_p, N_lines, n_photon, T_sec, S):
    """Strategy A with injected squeezed-light enhancement (1+S)/(1-S)."""
    F_sql, _ = strategy_A_SQL(m_p, N_lines, n_photon, T_sec)
    F = F_sql * (1+S)/(1-S)
    return F, 1/np.sqrt(F)


def strategy_B(N_lines, n_photon, T_sec, S):
    """Pure χ^(3) FWM-entangled differential strategy.
    F = 16 * n * T^2 * (1+S)/(1-S) * N(N+1)(2N+1)/6
       (signal = 2*m * Omega for m-th pair; sum over m = 1 to N_lines)
    """
    sum_m2 = N_lines*(N_lines+1)*(2*N_lines+1)/6
    F = 16 * n_photon * T_sec**2 * (1+S)/(1-S) * sum_m2
    return F, 1/np.sqrt(F)


def main():
    lam = 1550e-9
    n_photon = 1000
    T_sec = 1.0
    Omega_earth = 7.3e-5  # rad/s

    # -------- Panel (a): sigma vs number of comb lines --------
    R = 10e-6
    m_p = sagnac_coupling(R, lam)
    print(f"=== R = {R*1e6} μm, m_p = {m_p:.1f} ===\n")
    N_vals = np.arange(1, 101)
    sig_A_SQL = []
    sig_A_sq = []
    sig_B_S09 = []
    sig_B_S099 = []
    for N in N_vals:
        _, s_Asql = strategy_A_SQL(m_p, N, n_photon, T_sec)
        _, s_Asq = strategy_A_squeezed(m_p, N, n_photon, T_sec, 0.99)
        _, s_B09 = strategy_B(N, n_photon, T_sec, 0.9)
        _, s_B099 = strategy_B(N, n_photon, T_sec, 0.99)
        sig_A_SQL.append(s_Asql)
        sig_A_sq.append(s_Asq)
        sig_B_S09.append(s_B09)
        sig_B_S099.append(s_B099)
    sig_A_SQL = np.array(sig_A_SQL)
    sig_A_sq = np.array(sig_A_sq)
    sig_B_S09 = np.array(sig_B_S09)
    sig_B_S099 = np.array(sig_B_S099)

    # -------- Panel (b): sigma vs device radius --------
    R_vals = np.logspace(-6, -3, 40)
    N_fix = 21
    sig_R_Asql, sig_R_Aq, sig_R_B = [], [], []
    for R_v in R_vals:
        m = sagnac_coupling(R_v, lam)
        _, s_A = strategy_A_SQL(m, N_fix, n_photon, T_sec)
        _, s_Aq = strategy_A_squeezed(m, N_fix, n_photon, T_sec, 0.99)
        _, s_B = strategy_B(N_fix, n_photon, T_sec, 0.99)
        sig_R_Asql.append(s_A)
        sig_R_Aq.append(s_Aq)
        sig_R_B.append(s_B)
    sig_R_Asql = np.array(sig_R_Asql)
    sig_R_Aq = np.array(sig_R_Aq)
    sig_R_B = np.array(sig_R_B)

    # -------- Panel (c): sigma vs squeezing S --------
    S_vals = np.linspace(0, 0.999, 60)
    sig_S_A, sig_S_B = [], []
    N_fix2 = 50
    R_fix = 10e-6
    m_fix = sagnac_coupling(R_fix, lam)
    for S in S_vals:
        _, s_A = strategy_A_squeezed(m_fix, N_fix2, n_photon, T_sec, S)
        _, s_B = strategy_B(N_fix2, n_photon, T_sec, S)
        sig_S_A.append(s_A)
        sig_S_B.append(s_B)
    sig_S_A = np.array(sig_S_A)
    sig_S_B = np.array(sig_S_B)

    # -------- Panel (d): integration time to detect earth rate with SNR=1 --------
    T_vals = np.logspace(-3, 3, 50)
    sig_T_A, sig_T_B = [], []
    for T in T_vals:
        _, s_A = strategy_A_squeezed(m_p, 10, n_photon, T, 0.99)
        _, s_B = strategy_B(50, n_photon, T, 0.99)
        sig_T_A.append(s_A)
        sig_T_B.append(s_B)
    sig_T_A = np.array(sig_T_A)
    sig_T_B = np.array(sig_T_B)

    # Gyroscope performance classes
    AUTO_GRADE = 1e-3
    TACTICAL = 1e-4
    NAVIGATION = 1e-6

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # (a)
    ax = axes[0, 0]
    ax.loglog(N_vals, sig_A_SQL, 'b-', lw=1.8, label='Strategy A: SQL (no squeezing)')
    ax.loglog(N_vals, sig_A_sq, 'b--', lw=1.8, label='Strategy A: + injected squeeze $S\!=\!0.99$')
    ax.loglog(N_vals, sig_B_S09, 'r-', lw=1.8, label='Strategy B: FWM $S\!=\!0.9$')
    ax.loglog(N_vals, sig_B_S099, 'r--', lw=1.8, label='Strategy B: FWM $S\!=\!0.99$')
    for y, lab in [(AUTO_GRADE, 'auto MEMS'), (TACTICAL, 'tactical'), (NAVIGATION, 'navigation')]:
        ax.axhline(y, color='gray', ls=':', alpha=0.5)
        ax.text(1.2, y*1.1, lab, fontsize=8, color='gray')
    ax.set_xlabel(r'Number of comb lines $N$', fontsize=11)
    ax.set_ylabel(r'$\sigma_\Omega$ (rad/s/$\sqrt{\rm Hz}$)', fontsize=11)
    ax.set_title(f'(a) Sensitivity vs comb size, $R={R*1e6:.0f}\,\\mu$m', fontsize=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(alpha=0.3, which='both')

    # (b)
    ax = axes[0, 1]
    ax.loglog(R_vals*1e6, sig_R_Asql, 'b-', lw=1.8,
              label=f'Strategy A SQL, $N\!=\!{N_fix}$')
    ax.loglog(R_vals*1e6, sig_R_Aq, 'b--', lw=1.8,
              label=f'Strategy A + inject sq., $S\!=\!0.99$')
    ax.loglog(R_vals*1e6, sig_R_B, 'r-', lw=1.8,
              label=f'Strategy B, $S\!=\!0.99$')
    for y, lab in [(AUTO_GRADE, 'auto'), (TACTICAL, 'tactical'), (NAVIGATION, 'nav')]:
        ax.axhline(y, color='gray', ls=':', alpha=0.5)
    ax.axvline(10, color='green', ls=':', alpha=0.5)
    ax.text(11, 1e-3, '$R\!=\!10\,\mu$m', color='green', fontsize=9)
    ax.set_xlabel(r'Device radius $R$ ($\mu$m)', fontsize=11)
    ax.set_ylabel(r'$\sigma_\Omega$ (rad/s/$\sqrt{\rm Hz}$)', fontsize=11)
    ax.set_title(f'(b) Sensitivity vs radius (21-line comb)', fontsize=10)
    ax.legend(fontsize=8, loc='lower left')
    ax.grid(alpha=0.3, which='both')

    # (c)
    ax = axes[1, 0]
    ax.semilogy(S_vals, sig_S_A, 'b-', lw=2,
                label=f'Strategy A ({N_fix2} lines, $R\!=\!10\,\mu$m)')
    ax.semilogy(S_vals, sig_S_B, 'r-', lw=2,
                label=f'Strategy B ({N_fix2} lines, $R\!=\!10\,\mu$m)')
    for y, lab in [(AUTO_GRADE, 'auto'), (TACTICAL, 'tactical'), (NAVIGATION, 'navigation')]:
        ax.axhline(y, color='gray', ls=':', alpha=0.5)
        ax.text(0.02, y*1.3, lab, fontsize=8, color='gray')
    ax.set_xlabel(r'Squeezing $S = 2\chi/\kappa^{\rm eff}$', fontsize=11)
    ax.set_ylabel(r'$\sigma_\Omega$ (rad/s/$\sqrt{\rm Hz}$)', fontsize=11)
    ax.set_title('(c) Sensitivity vs parametric drive (at threshold $S\!\\to\!1$)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    # (d)
    ax = axes[1, 1]
    ax.loglog(T_vals, sig_T_A / np.sqrt(T_vals), 'b-', lw=2,
              label='Strategy A + inject sq., 10 lines')
    ax.loglog(T_vals, sig_T_B / np.sqrt(T_vals), 'r-', lw=2,
              label='Strategy B, 50 lines, $S\!=\!0.99$')
    ax.axhline(Omega_earth, color='green', lw=1.5, label='Earth rate')
    for y, lab in [(AUTO_GRADE, 'auto'), (TACTICAL, 'tactical'), (NAVIGATION, 'nav')]:
        ax.axhline(y, color='gray', ls=':', alpha=0.5)
    ax.set_xlabel(r'Integration time $T$ (s)', fontsize=11)
    ax.set_ylabel(r'Detectable $\Omega$ at SNR$\!=\!1$ (rad/s)', fontsize=11)
    ax.set_title('(d) Earth-rate detectability vs integration time', fontsize=10)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3, which='both')

    fig.suptitle(r'GaAs chip-scale quantum-comb gyroscope: full sensitivity map',
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR/"fig_inertial_sensitivity.pdf", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR/"fig_inertial_sensitivity.png", dpi=150, bbox_inches='tight')
    print(f"Saved {FIG_DIR}/fig_inertial_sensitivity.pdf")

    # Summary numbers
    print(f"\n=== Headline numbers ({R*1e6}μm ring, n={n_photon}, T={T_sec}s) ===")
    print(f"Strategy A SQL, 1 line:   σ = {strategy_A_SQL(m_p, 1, n_photon, T_sec)[1]:.2e} rad/s/√Hz")
    print(f"Strategy A SQL, 10 lines: σ = {strategy_A_SQL(m_p, 10, n_photon, T_sec)[1]:.2e}")
    print(f"Strategy A+sq(S=.99), 10: σ = {strategy_A_squeezed(m_p, 10, n_photon, T_sec, 0.99)[1]:.2e}")
    print(f"Strategy B (S=.99, 10):   σ = {strategy_B(10, n_photon, T_sec, 0.99)[1]:.2e}")
    print(f"Strategy B (S=.99, 50):   σ = {strategy_B(50, n_photon, T_sec, 0.99)[1]:.2e}")
    print(f"Strategy B (S=.99, 100):  σ = {strategy_B(100, n_photon, T_sec, 0.99)[1]:.2e}")

    # Save JSON
    with open(DATA_DIR/"inertial_sensitivity.json", "w") as f:
        json.dump(dict(R=R, m_p=m_p, n_photon=n_photon, T_sec=T_sec,
                       N_vals=N_vals.tolist(),
                       sig_A_SQL=sig_A_SQL.tolist(),
                       sig_A_sq_S099=sig_A_sq.tolist(),
                       sig_B_S09=sig_B_S09.tolist(),
                       sig_B_S099=sig_B_S099.tolist(),
                       earth_rate=Omega_earth,
                       auto_grade=AUTO_GRADE,
                       tactical=TACTICAL,
                       navigation=NAVIGATION), f, indent=2)


if __name__ == "__main__":
    main()
