"""
Explicit chi^(3) cross-Kerr sweep vs 1/Q.

Two curves:
  (a) Kleinman chi^(3), passive: Im/Re is identically zero (E11 theorem)
  (b) Kleinman-broken chi^(3) (imaginary part added at 5% level, e.g., TPA):
      Im/Re is Q-independent and set by Im(chi^(3))/Re(chi^(3)) — demonstrates
      the mechanistic distinction between material absorption (TPA) and
      open-system geometry.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from synthetic_modes import build_vector_qnm, g3_cross_kleinman

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"

def g3_cross_anisotropic(fA, fB, chi3_R, chi3_I, dV=1.0):
    """Cross-Kerr with complex chi^(3) = chi3_R + i chi3_I (Kleinman but TPA-broken)."""
    gR = g3_cross_kleinman(fA, fB, chi3_R, dV)
    gI = g3_cross_kleinman(fA, fB, chi3_I, dV)  # still respects Kleinman in structure,
    # but adds imaginary magnitude -> complex total with Im/Re given by chi3_I/chi3_R
    return gR + 1j*gI


def main():
    rng = np.random.default_rng(seed=20260416)
    Nx = Ny = Nz = 16
    shape = (Nx, Ny, Nz)
    dV = 1.0/(Nx*Ny*Nz)
    chi3_R = 1.0

    N_CONFIG = 20
    Qs = np.logspace(2, 6, 25)

    # (a) passive Kleinman
    ratios_passive = np.zeros((N_CONFIG, len(Qs)))
    # (b) TPA-broken: Im(chi3)/Re = 0.05 (GaAs at lambda=1000 nm)
    im_frac = 0.05
    ratios_tpa = np.zeros((N_CONFIG, len(Qs)))

    for c in range(N_CONFIG):
        # build two distinct modes per config
        ft_A_list = []
        ft_B_list = []
        f0A, phiA = None, None
        for Q in Qs:
            fA, f0A_loc, phiA_loc = build_vector_qnm(shape, rng, Q=Q, n_cells=2)
            fB, f0B_loc, phiB_loc = build_vector_qnm(shape, rng, Q=Q, n_cells=2)
            ft_A_list.append(fA)
            ft_B_list.append(fB)
        # use a single shared (f0, phi) per config for clean scaling:
        f0s, phis = [], []
        for _ in range(2):
            ft, f0, phi = build_vector_qnm(shape, rng, Q=1e6, n_cells=2)
            f0s.append(f0); phis.append(phi)
        for iQ, Q in enumerate(Qs):
            fA = f0s[0].astype(complex)*(1 + 1j*phis[0]/Q)
            fB = f0s[1].astype(complex)*(1 + 1j*phis[1]/Q)
            gP = g3_cross_kleinman(fA, fB, chi3_R, dV)
            ratios_passive[c, iQ] = abs(gP.imag/gP.real) if abs(gP.real)>0 else 0

            gT = g3_cross_anisotropic(fA, fB, chi3_R, chi3_R*im_frac, dV)
            ratios_tpa[c, iQ] = abs(gT.imag/gT.real)

    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    inv_Q = 1.0/Qs
    mean_p = np.abs(ratios_passive).mean(axis=0) + 1e-22   # floor for log plot
    mean_t = np.abs(ratios_tpa).mean(axis=0)

    ax.loglog(inv_Q, mean_p, 'o-', color='tab:blue',
              ms=5, label=r'passive Kleinman $\chi^{(3)}$ (theorem E11)')
    # For passive: the true value is zero, but we show the machine-precision floor
    ax.axhline(1e-15, color='tab:blue', ls=':', alpha=0.5,
               label=r'double-precision floor ($\sim10^{-15}$)')
    ax.loglog(inv_Q, mean_t, 's-', color='tab:red',
              ms=5, label=r'TPA-broken Kleinman: Im($\chi^{(3)})/\mathrm{Re}=5\%$')

    # For reference: earlier-paper incorrect claim Im/Re ~ 1/(2Q)
    ax.loglog(inv_Q, 0.5*inv_Q, 'k--', lw=1.0, alpha=0.5,
              label=r'earlier (incorrect) $\sim 1/(2Q)$')

    ax.set_xlabel(r'$1/Q$', fontsize=11)
    ax.set_ylabel(r'$|\mathrm{Im}(g^{(3)}_{\lambda\mu\mu\lambda})/\mathrm{Re}|$',
                  fontsize=11)
    ax.set_title(r'$\chi^{(3)}$ cross-Kerr $|\mathrm{Im}/\mathrm{Re}|$: '
                 r'passive is exactly zero; TPA sets $Q$-independent floor',
                 fontsize=10)
    ax.set_ylim(1e-22, 1.0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_chi3_cross_kerr.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_chi3_cross_kerr.png", dpi=150)
    plt.close(fig)
    print(f"Saved fig_chi3_cross_kerr.pdf")
    print(f"passive Kleinman max |Im/Re| over {N_CONFIG*len(Qs)} configs = "
          f"{ratios_passive.max():.2e}")
    print(f"TPA-broken |Im/Re| ~ {mean_t.mean():.3f} (expected ~{im_frac})")

if __name__ == "__main__":
    main()
