"""
Numerical benchmark: Mandel photon-counting vs homodyne Fisher information
for a Kerr-comb pair.

Question addressed (qnm_inertial.tex, App.~PNR_FI):
-----------------------------------------------------
The Gaussian FI formula
    F_hom(Omega) = mu'^T C^-1 mu' + (1/2) Tr[(C^-1 C')^2]
applies ONLY to homodyne/heterodyne (Gaussian POVMs) on Gaussian states.
Photon counting of a Gaussian state produces the Mandel distribution
    P(n_+, n_- | Omega) = <n_+, n_- | rho(Omega) | n_+, n_->
which is non-Gaussian.  Its Fisher information
    F_PNR(Omega) = sum_n (1/P)(dP/dOmega)^2
may be much smaller than F_hom when Omega enters as a phase.

Subtlety (mode-swap symmetry).  For a perfectly symmetric DTMSS
|psi> = D_+(alpha e^{i phi}) D_-(alpha e^{-i phi}) S_2(r) |00>
the state satisfies R_z(2 phi) |psi(0)> = |psi(phi)> with
R_z diagonal in the Fock basis.  PNR in the lab basis is
phase-blind: F_PNR_lab = 0 exactly.  The physical photon-counting
Sagnac detection always includes a 50:50 beamsplitter step that
mixes + and - modes; F_PNR is then nonzero and this is the
relevant quantity.

This script:
  1. Constructs a symmetric DTMSS in qutip.
  2. Computes three Fisher informations:
       (i)  F_PNR_lab     -- direct PNR on (+, -) modes  [= 0 by symmetry]
       (ii) F_PNR_BS      -- PNR after 50:50 beamsplitter [the real PNR FI]
       (iii) F_hom        -- Gaussian covariance formula  [optimised homodyne]
  3. Compares F_PNR_BS / F_hom vs the analytic prediction.
"""
from __future__ import annotations
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def comb_pair_state(N_fock, alpha_abs, r_sq, phase):
    """Build |psi> = D_+(alpha e^{+i phase}) D_-(alpha e^{-i phase}) S(r) |00>
    in a two-mode Fock basis of size N_fock each.
    """
    a = qt.tensor(qt.destroy(N_fock), qt.qeye(N_fock))
    b = qt.tensor(qt.qeye(N_fock), qt.destroy(N_fock))
    vac = qt.tensor(qt.basis(N_fock, 0), qt.basis(N_fock, 0))
    # Two-mode squeezer:  S_2(xi) = exp(xi* a b - xi a^dag b^dag)
    # Take xi real positive -> standard TMSV with tanh r = S
    S2 = (r_sq * (a * b - a.dag() * b.dag())).expm()
    # Displacements
    alpha_p = alpha_abs * np.exp(+1j * phase)
    alpha_m = alpha_abs * np.exp(-1j * phase)
    Dp = (alpha_p * a.dag() - np.conj(alpha_p) * a).expm()
    Dm = (alpha_m * b.dag() - np.conj(alpha_m) * b).expm()
    psi = Dp * Dm * S2 * vac
    return psi / psi.norm(), a, b


def pnr_distribution(psi, N_fock):
    """P(n_+, n_- | phase) from |psi>."""
    vec = psi.full().flatten()
    vec2 = np.abs(vec)**2
    P = vec2.reshape(N_fock, N_fock)
    return P


def beamsplitter_50_50(N_fock):
    """U_BS such that  a_s = (a_+ + a_-)/sqrt2,  a_d = (a_+ - a_-)/sqrt2.
    Equivalently, U_BS = exp((pi/4)(a_+^dag a_- - a_+ a_-^dag))."""
    a = qt.tensor(qt.destroy(N_fock), qt.qeye(N_fock))
    b = qt.tensor(qt.qeye(N_fock), qt.destroy(N_fock))
    theta = np.pi / 4
    U = (theta * (a.dag() * b - a * b.dag())).expm()
    return U


def pnr_distribution_BS(psi, U_BS, N_fock):
    """PNR after 50:50 beamsplitter:
         P(n_s, n_d | phase) = |<n_s, n_d | U_BS | psi>|^2.
    """
    psi_out = U_BS * psi
    vec = psi_out.full().flatten()
    P = np.abs(vec) ** 2
    return P.reshape(N_fock, N_fock)


def gaussian_cov_and_mean(psi, a, b):
    """Compute Gaussian mean vector (mu) and covariance matrix (C) for
    (X_a, P_a, X_b, P_b) quadratures of state psi."""
    def X(op): return (op + op.dag()) / np.sqrt(2)
    def P(op): return (op - op.dag()) / (1j * np.sqrt(2))
    quads = [X(a), P(a), X(b), P(b)]
    n = len(quads)
    mu = np.array([float(qt.expect(q, psi).real) for q in quads])
    C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            qq = (quads[i] * quads[j] + quads[j] * quads[i]) / 2
            C[i, j] = float(qt.expect(qq, psi).real) - mu[i] * mu[j]
    return mu, C


def fisher_PNR_numerical(N_fock, alpha_abs, r_sq, phase0, dphase, use_BS):
    """Compute F_PNR by finite difference.
    use_BS=False: PNR directly on (+, -) modes (lab basis).  Expected 0.
    use_BS=True:  PNR after 50:50 BS (measure s, d modes).  Nonzero.
    """
    psi_p, _, _ = comb_pair_state(N_fock, alpha_abs, r_sq, phase0 + dphase)
    psi_m, _, _ = comb_pair_state(N_fock, alpha_abs, r_sq, phase0 - dphase)
    psi_0, _, _ = comb_pair_state(N_fock, alpha_abs, r_sq, phase0)
    if use_BS:
        U = beamsplitter_50_50(N_fock)
        P_p = pnr_distribution_BS(psi_p, U, N_fock)
        P_m = pnr_distribution_BS(psi_m, U, N_fock)
        P_0 = pnr_distribution_BS(psi_0, U, N_fock)
    else:
        P_p = pnr_distribution(psi_p, N_fock)
        P_m = pnr_distribution(psi_m, N_fock)
        P_0 = pnr_distribution(psi_0, N_fock)
    dP = (P_p - P_m) / (2 * dphase)
    mask = P_0 > 1e-14
    F = np.sum((dP[mask] ** 2) / P_0[mask])
    return F, P_0


def fisher_homodyne(N_fock, alpha_abs, r_sq, phase0, dphase):
    """Compute F_hom from the Gaussian covariance formula."""
    psi_p, a, b = comb_pair_state(N_fock, alpha_abs, r_sq, phase0 + dphase)
    psi_m, _, _ = comb_pair_state(N_fock, alpha_abs, r_sq, phase0 - dphase)
    psi_0, _, _ = comb_pair_state(N_fock, alpha_abs, r_sq, phase0)
    mu_p, C_p = gaussian_cov_and_mean(psi_p, a, b)
    mu_m, C_m = gaussian_cov_and_mean(psi_m, a, b)
    mu_0, C_0 = gaussian_cov_and_mean(psi_0, a, b)
    dmu = (mu_p - mu_m) / (2 * dphase)
    dC = (C_p - C_m) / (2 * dphase)
    C_inv = np.linalg.inv(C_0)
    term1 = float(dmu @ C_inv @ dmu)
    term2 = 0.5 * float(np.trace((C_inv @ dC) @ (C_inv @ dC)))
    return term1 + term2


def main():
    N_fock = 24          # truncation of Fock basis per mode
    alpha_abs = 0.8      # coherent amplitude per comb line (keep low to fit)
    phase0 = 0.01        # small baseline phase (small-Omega regime)
    dphase = 0.002       # finite-difference step for d/dphase
    # Note: phase == m*Omega*T; FI per (m*T)^2 is F_phase.

    # Squeezing sweep (keep S <= 0.8 for Fock truncation convergence)
    S_vals = np.array([0.0, 0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8])
    r_vals = np.arctanh(S_vals)

    F_PNR_lab = []
    F_PNR_BS = []
    F_hom = []
    n_mean = []
    print("=== PNR vs Homodyne FI for displaced TMSS ===\n")
    print(f"N_fock={N_fock}, |alpha|={alpha_abs}, n_coh={alpha_abs**2:.2f}, "
          f"phase0={phase0}")
    print(f"{'S':>6} {'r':>6} {'<n_tot>':>8} {'F_lab':>10} "
          f"{'F_BS':>11} {'F_hom':>11} {'F_BS/F_hom':>11} {'theory':>10}")
    for S, r in zip(S_vals, r_vals):
        F_lab, _ = fisher_PNR_numerical(N_fock, alpha_abs, r, phase0, dphase,
                                         use_BS=False)
        F_bs, _ = fisher_PNR_numerical(N_fock, alpha_abs, r, phase0, dphase,
                                        use_BS=True)
        F_H = fisher_homodyne(N_fock, alpha_abs, r, phase0, dphase)
        psi, a, b = comb_pair_state(N_fock, alpha_abs, r, phase0)
        n_tot = float((qt.expect(a.dag()*a + b.dag()*b, psi)).real)
        ratio = F_bs / F_H if F_H > 0 else np.nan
        theory = (1 - S)**2 / (2 * (1 + S)**2)
        print(f"{S:>6.2f} {r:>6.3f} {n_tot:>8.3f} {F_lab:>10.2e} "
              f"{F_bs:>11.3e} {F_H:>11.3e} {ratio:>11.3e} {theory:>10.3e}")
        F_PNR_lab.append(F_lab)
        F_PNR_BS.append(F_bs)
        F_hom.append(F_H)
        n_mean.append(n_tot)

    F_PNR_lab = np.array(F_PNR_lab)
    F_PNR_BS = np.array(F_PNR_BS)
    F_hom = np.array(F_hom)
    F_PNR = F_PNR_BS

    # --- plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.semilogy(S_vals, F_hom, 'bo-', ms=8, lw=1.8,
                label='homodyne (Gaussian formula)')
    ax.semilogy(S_vals, F_PNR, 'r^-', ms=8, lw=1.8,
                label='PNR (Mandel distribution)')
    # Analytic predictions (only sensible in small-phase limit)
    F_hom_analytic = 4 * (2 * alpha_abs**2) * (1 + S_vals) / (1 - S_vals + 1e-12)
    F_PNR_analytic = 4 * (2 * alpha_abs**2) * (1 - S_vals) / (1 + S_vals)
    ax.semilogy(S_vals, F_hom_analytic, 'b:', lw=1.2,
                label=r'$8|\alpha|^2(1+S)/(1-S)$ theory')
    ax.semilogy(S_vals, F_PNR_analytic, 'r:', lw=1.2,
                label=r'$8|\alpha|^2(1-S)/(1+S)$ theory')
    ax.set_xlabel(r'Squeezing $S = \tanh r$', fontsize=11)
    ax.set_ylabel(r'$F(\phi)\;[(m T)^{-2}\!\times F(\Omega)]$', fontsize=11)
    ax.set_title(r'Fisher info per (m T)$^2$ vs squeezing'
                 f'\n|α|={alpha_abs}, N_fock={N_fock}, φ₀={phase0}',
                 fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, which='both')

    ax = axes[1]
    ratio = F_PNR / F_hom
    theory = (1 - S_vals)**2 / (2 * (1 + S_vals)**2)
    ax.semilogy(S_vals, ratio, 'ko-', ms=8, lw=1.8,
                label='numerical  $F_{\\rm PNR}/F_{\\rm hom}$')
    ax.semilogy(S_vals, theory, 'g:', lw=1.6,
                label=r'theory  $(1-S)^2/[2(1+S)^2]$')
    ax.axhline(0.5, color='gray', ls='--', alpha=0.6,
               label='coherent limit = 1/2 (S=0)')
    ax.set_xlabel(r'Squeezing $S$', fontsize=11)
    ax.set_ylabel(r'$F_{\rm PNR}/F_{\rm hom}$', fontsize=11)
    ax.set_title('PNR penalty vs optimised homodyne\n'
                 'PNR loses squeezing advantage as $S\\to 1$',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle(r'Comb-pair Fisher information: photon counting vs homodyne'
                 '\n(validates Eq. FI_PNR_vs_hom of qnm_inertial.tex App. PNR_FI)',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_comb_PNR_vs_homodyne.pdf", dpi=200,
                bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_comb_PNR_vs_homodyne.png", dpi=150,
                bbox_inches='tight')
    print(f"\nSaved {FIG_DIR}/fig_comb_PNR_vs_homodyne.pdf")

    # Save data
    with open(DATA_DIR / "comb_PNR_vs_homodyne.json", "w") as f:
        json.dump(dict(
            N_fock=N_fock, alpha_abs=alpha_abs, phase0=phase0, dphase=dphase,
            S=S_vals.tolist(), r=r_vals.tolist(),
            F_PNR_lab=F_PNR_lab.tolist(),
            F_PNR_BS=F_PNR_BS.tolist(),
            F_hom=F_hom.tolist(),
            ratio_BS=(F_PNR_BS / F_hom).tolist(),
            ratio_theory=((1 - S_vals)**2 / (2 * (1 + S_vals)**2)).tolist(),
            n_mean=n_mean,
        ), f, indent=2)


if __name__ == "__main__":
    main()
