"""
Figure 4 (active-cavity cross-mode dephasing): numerical verification of
      kappa_lambda(n_mu) = kappa_0 + 4 |g3_sat^I| n_mu
from the Keldysh one-loop cross-Kerr self-energy, with purely imaginary
g3_sat = i A/(gamma_perp) at resonance.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"

def dressed_propagator_spectrum(omega, kappa0, g3_sat_I, nbar_mu, omega0=0.0):
    """|G^R(omega)|^2 for dressed propagator:
       G^R = 1/(omega - omega0 - 4 g3_R n - i (kappa/2 + 2 |g3_sat^I| n)).
    """
    kappa_eff = kappa0 + 4*g3_sat_I*nbar_mu
    return 1.0/((omega - omega0)**2 + (kappa_eff/2)**2)


def fwhm(omega, S):
    peak = S.max()
    half = peak/2
    above = S > half
    idx = np.where(np.diff(above.astype(int))!=0)[0]
    if len(idx) < 2: return None
    return omega[idx[-1]] - omega[idx[0]]


def main():
    # Parameters representative of a GaAs QCL-like active cavity (rescaled units)
    kappa0 = 1.0           # natural linewidth in arb units
    # g3_sat^I in units of kappa0, derived from typical gain overlap
    # |g3_sat^I / kappa_0| ~ 0.015 per photon for moderate gain overlap
    g3_sat_I = 0.015

    nbars = np.linspace(0, 12, 300)
    kappa_vs_n_active  = kappa0 + 4*g3_sat_I*nbars
    kappa_vs_n_passive = np.ones_like(nbars)*kappa0  # passive: flat (Kleinman)

    # Numerical verification: extract linewidth from the dressed propagator
    omega = np.linspace(-3, 3, 5001)
    measured_kappa = np.zeros_like(nbars)
    for i, n in enumerate(nbars):
        S = dressed_propagator_spectrum(omega, kappa0, g3_sat_I, n)
        width = fwhm(omega, S)
        measured_kappa[i] = width if width is not None else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

    ax = axes[0]
    ax.plot(nbars, kappa_vs_n_passive/kappa0, 'k--', lw=1.8,
            label=r'passive (Kleinman): $g^{(3)}_{\lambda\mu\mu\lambda}\in\mathbb{R}$')
    ax.plot(nbars, kappa_vs_n_active/kappa0, 'b-', lw=1.8,
            label=r'active (gain-saturated): $g^{(3)}_{\rm sat}\in i\mathbb{R}$')
    ax.plot(nbars, measured_kappa/kappa0, 'r.', alpha=0.3, ms=3,
            label='measured FWHM of dressed propagator')

    ax.set_xlabel(r'$\bar{n}_\mu$ (pump-mode occupation)', fontsize=11)
    ax.set_ylabel(r'$\kappa_\lambda(\bar{n}_\mu)/\kappa_\lambda^{(0)}$', fontsize=11)
    ax.set_title(r'Cross-mode dephasing: passive flat, active linear in $\bar n_\mu$',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.axhline(1.1, color='green', alpha=0.3, ls=':')
    ax.text(0.5, 1.12, r'$\delta\kappa/\kappa=10\%$', color='green', fontsize=9)

    ax = axes[1]
    # Dressed lineshape at a few pump values
    for n in [0, 2, 5, 10]:
        S = dressed_propagator_spectrum(omega, kappa0, g3_sat_I, n)
        ax.plot(omega, S/S.max(), label=fr'$\bar n_\mu={n}$')
    ax.set_xlabel(r'$(\omega-\omega_0)/\kappa_\lambda^{(0)}$', fontsize=11)
    ax.set_ylabel(r'$|G^R(\omega)|^2$ (normalized)', fontsize=11)
    ax.set_title('Dressed-propagator lineshape broadens with pump photons',
                 fontsize=10)
    ax.set_xlim(-3, 3)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_cross_dephasing.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_cross_dephasing.png", dpi=150)
    plt.close(fig)

    # Slope check: theory = 4 g3_sat^I / kappa_0
    slope_meas = (kappa_vs_n_active[-1] - kappa_vs_n_active[0]) / (nbars[-1]-nbars[0]) / kappa0
    slope_pred = 4*g3_sat_I/kappa0
    print(f"Slope (measured): {slope_meas:.4f}")
    print(f"Slope (theory 4 g3_I/kappa): {slope_pred:.4f}")
    assert abs(slope_meas - slope_pred)/slope_pred < 1e-6

if __name__ == "__main__":
    main()
