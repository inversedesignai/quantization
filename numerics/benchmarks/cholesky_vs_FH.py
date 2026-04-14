"""
REP-1: Cholesky-dressed bosonic operators vs Franke-Hughes Fano dressed
operators, in the SINGLE-MODE limit of a 1D Fabry-Perot cavity.

Setup: 1D FP cavity at varying mirror index n_c, giving Q from ~10 to ~10^4.
For each Q, compute:
  (a) The Cholesky-dressed normalization: O_11 = integral eps |f̃|^2 dx
      The Cholesky operator is â_C = â_bare / sqrt(O_11).
  (b) The Franke-Hughes Fano normalization, expanded in 1/Q.
      To leading order O(1/Q^2): the bare-Fano operator commutator is
      [b̂, b̂^†] = 1 - eps_FH where eps_FH = O((kappa/omega)^2) = O(1/Q^2).

Theoretical claim: |O_11 - 1| = O(1/Q^2), so the Cholesky and Fano operators
agree to O(1/Q^2) for a single mode.

Numerical check: compute O_11 from the analytic 1D FP QNM at various Q, fit
|O_11 - 1| vs 1/Q.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/home/zlin/quantization/numerics/qnm_synthetic')
from test_chi2_fabryperot import fp_qnm_complex_k, fp_qnm_field
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def hermitian_overlap_O11(Ex, dx, eps=1.0):
    """O_11 = integral eps |f̃|^2 dx (Hermitian inner product)."""
    return eps * (Ex.conj() * Ex).sum().real * dx


def bilinear_norm_B11(Ex, dx, eps=1.0):
    """B_11 = integral eps f̃·f̃ dx (bilinear, no conjugation)."""
    return eps * (Ex * Ex).sum() * dx


def main():
    L = 1.0
    Nx = 8192
    x = np.linspace(0.0001, 0.9999, Nx)
    dx = x[1] - x[0]

    n_c_vals = np.array([1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0,
                          200.0, 500.0, 1000.0, 2000.0, 5000.0])

    Qs = []
    O11s = []
    B11s = []
    for n_c in n_c_vals:
        f, Q = fp_qnm_field(x, m=1, L=L, n_c=n_c)   # already bilinear-normalized to B=1
        # raw QNM (renormalize to bare amplitude=1 at center)
        k_tilde, _ = fp_qnm_complex_k(1, L, n_c)
        f_raw = np.sin(k_tilde * x)
        # normalize so peak Re ≈ 1 (= closed-cavity convention)
        f_raw = f_raw / abs(f_raw[Nx//2])
        # compute O_11 and B_11
        O11 = hermitian_overlap_O11(f_raw, dx)
        B11 = bilinear_norm_B11(f_raw, dx)
        # closed-cavity reference: integral sin^2(pi x/L) dx = L/2
        O_closed = 0.5
        # ratio
        Qs.append(Q)
        O11s.append(O11)
        B11s.append(abs(B11))
        print(f"  n_c={n_c:7.1f}  Q={Q:8.1f}  O_11={O11:.6f}  |B_11|={abs(B11):.6f}  "
              f"|O11/Oclosed - 1|={abs(O11/O_closed - 1):.3e}")

    Qs = np.array(Qs)
    O11s = np.array(O11s)
    O11_norm = O11s / 0.5    # normalize to closed-cavity value

    # fit: |O11_norm - 1| vs 1/Q
    devs = np.abs(O11_norm - 1.0)
    # fit at high Q only
    mask_highQ = Qs > 30
    log_dev = np.log(devs[mask_highQ])
    log_invQ = np.log(1.0/Qs[mask_highQ])
    slope, intercept = np.polyfit(log_invQ, log_dev, 1)
    print(f"\nHigh-Q fit: |O_11/O_closed - 1| ~ Q^{-slope:.2f}")
    print(f"   prefactor = {np.exp(intercept):.3f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(1/Qs, devs, 'bo-', ms=7,
              label=r'Cholesky overlap deviation $|O_{11}/O_{\rm closed} - 1|$')
    inv_Q = np.logspace(-5, -1, 50)
    ax.loglog(inv_Q, np.exp(intercept) * inv_Q**slope, 'k--', lw=1,
              label=fr'fit: $\propto Q^{{{-slope:.2f}}}$')
    # comparison curves
    ax.loglog(inv_Q, inv_Q, 'g:', lw=1, label=r'$1/Q$ (FH single-mode error)')
    ax.loglog(inv_Q, inv_Q**2, 'r:', lw=1, label=r'$1/Q^2$ (Cholesky theoretical)')
    ax.set_xlabel(r'$1/Q$', fontsize=11)
    ax.set_ylabel(r'$|O_{11}/O_{\rm closed} - 1|$', fontsize=11)
    ax.set_title(r'Cholesky single-mode overlap deviation vs $1/Q$:'
                 r' $\approx Q^{-2}$ scaling', fontsize=10)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_cholesky_vs_FH.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_cholesky_vs_FH.png", dpi=150)
    print(f"\nSaved {FIG_DIR}/fig_cholesky_vs_FH.pdf")

    if abs(slope - (-2)) < 0.3:
        print("=== [PASS] Cholesky single-mode overlap deviates as O(1/Q^2) ===")
        print("=== Equivalent to FH agreement to O(1/Q^2) for single mode ===")
    else:
        print(f"=== Slope {slope:.2f} differs from theoretical -2 ===")


if __name__ == "__main__":
    main()
