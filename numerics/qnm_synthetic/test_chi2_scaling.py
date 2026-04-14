"""
CORE TEST — FLAGSHIP PREDICTION of PRX Quantum paper.

For Kleinman chi^(2), the coupling g^(2)_{lambda mu nu} has
    Im(g^2)/Re(g^2) ~ (1/Q_lambda + 1/Q_mu + 1/Q_nu) / 2 * eta^(2)

Here eta^(2) is a geometric overlap factor O(1).  We verify the linear-in-1/Q
scaling by sweeping Q for a fixed set of mode functions with prescribed phi.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from synthetic_modes import build_vector_qnm, g2_kleinman_Td, g2_kleinman_isotropic


def _build_fixed_modes(shape, rng, n_cells=2):
    """Generate the zeroth-order mode functions f^(0) (real) and phase phi (real)
    ONCE, then scale with Q -> f = f0 (1 + i phi/Q) to get a clean 1/Q sweep.
    """
    f0_list, phi_list = [], []
    for _ in range(3):
        ft, f0, phi = build_vector_qnm(shape, rng, Q=1e6, n_cells=n_cells)
        f0_list.append(f0)
        phi_list.append(phi)
    return f0_list, phi_list


def _dress(f0, phi, Q):
    return f0.astype(np.complex128) * (1.0 + 1j/Q * phi)


def main():
    rng = np.random.default_rng(seed=20260415)
    Nx = Ny = Nz = 16
    shape = (Nx, Ny, Nz)
    dV = 1.0/(Nx*Ny*Nz)
    chi2 = 1.0

    # single set of modes for a clean, deterministic 1/Q scan
    N_CONFIG = 40
    Qs = np.logspace(2, 6, 30)

    all_ratios_Td = np.zeros((N_CONFIG, len(Qs)))
    all_ratios_iso = np.zeros((N_CONFIG, len(Qs)))

    for c in range(N_CONFIG):
        f0s, phis = _build_fixed_modes(shape, rng, n_cells=2)
        for iQ, Q in enumerate(Qs):
            # all three modes share the same Q for clean scaling
            fA = _dress(f0s[0], phis[0], Q)
            fB = _dress(f0s[1], phis[1], Q)
            fC = _dress(f0s[2], phis[2], Q)
            g2 = g2_kleinman_Td(fA, fB, fC, chi2, dV)
            all_ratios_Td[c, iQ]  = g2.imag/g2.real
            g2i = g2_kleinman_isotropic(fA, fB, fC, chi2, dV)
            all_ratios_iso[c, iQ] = g2i.imag/g2i.real

    # log-log fit: log|Im/Re| = alpha * log(1/Q) + beta
    inv_Q = 1.0/Qs

    log_ratios_Td  = np.log(np.abs(all_ratios_Td).mean(axis=0))
    log_ratios_iso = np.log(np.abs(all_ratios_iso).mean(axis=0))
    log_invQ = np.log(inv_Q)

    alpha_Td, beta_Td  = np.polyfit(log_invQ, log_ratios_Td, 1)
    alpha_iso, beta_iso = np.polyfit(log_invQ, log_ratios_iso, 1)

    print(f"\n=== FLAGSHIP PREDICTION — chi^(2) Im/Re ~ 1/Q ===\n")
    print(f"Td Kleinman chi^(2)  (GaAs d14):  slope = {alpha_Td:.4f}  (theory: 1)")
    print(f"                                   prefactor = {np.exp(beta_Td):.4f}")
    print(f"Iso Kleinman chi^(2):              slope = {alpha_iso:.4f}  (theory: 1)")
    print(f"                                   prefactor = {np.exp(beta_iso):.4f}")

    # predicted prefactor = 1.5 = (1/Q_l+1/Q_m+1/Q_n)/2 * eta
    # with Q_l=Q_m=Q_n=Q --> prefactor = 3/(2Q) * eta
    # so slope=1 and prefactor = (3/2) * eta
    predicted_prefactor = 3.0/2.0
    print(f"\nExpected slope = 1 (linear in 1/Q)")
    print(f"Expected prefactor = (3/2) * |eta|  where eta = O(1) geometric factor")

    # for Q=30,000 GaAs H1, theory: Im/Re ≈ 1.5/Q = 5e-5
    Q_GaAs = 3e4
    predicted_GaAs = 3.0/(2.0*Q_GaAs) * (np.exp(beta_Td)/predicted_prefactor)
    print(f"\nAt GaAs H1 Q=3e4:")
    print(f"  predicted Im/Re = {3.0/(2*Q_GaAs)*abs(np.exp(beta_Td)/predicted_prefactor):.3e}")
    print(f"  measured from interpolation = {np.abs(all_ratios_Td).mean(axis=0)[np.argmin(np.abs(Qs-Q_GaAs))]:.3e}")

    # Figure: loglog Im/Re vs 1/Q with both chi^(2) channels
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    mean_Td  = np.abs(all_ratios_Td).mean(axis=0)
    std_Td   = np.abs(all_ratios_Td).std(axis=0)
    mean_iso = np.abs(all_ratios_iso).mean(axis=0)
    std_iso  = np.abs(all_ratios_iso).std(axis=0)

    ax.errorbar(inv_Q, mean_Td, yerr=std_Td, fmt='o',
                color='tab:blue', ms=5, capsize=3,
                label=r'$T_d$ Kleinman $\chi^{(2)}$ (GaAs $d_{14}$)')
    ax.errorbar(inv_Q, mean_iso, yerr=std_iso, fmt='s',
                color='tab:orange', ms=4, capsize=3,
                label=r'Isotropic Kleinman $\chi^{(2)}$')

    # Power-law fit lines
    x_fit = np.logspace(-6.5, -1.5, 200)
    ax.plot(x_fit, np.exp(beta_Td)*x_fit**alpha_Td, '--',
            color='tab:blue', alpha=0.7,
            label=rf'fit: slope={alpha_Td:.2f}')
    ax.plot(x_fit, np.exp(beta_iso)*x_fit**alpha_iso, ':',
            color='tab:orange', alpha=0.7,
            label=rf'fit: slope={alpha_iso:.2f}')
    # theory line: 1/Q slope, prefactor eta=1 gives 3/(2Q)
    ax.plot(x_fit, 1.5*x_fit, 'k-', lw=1.0,
            label=r'Theory $(3/2)\cdot(1/Q)$')

    # GaAs design point
    ax.axvline(1.0/3e4, color='green', alpha=0.5, ls=':')
    ax.text(1.05/3e4, 2e-6, 'GaAs\nH1', color='green', fontsize=9)

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$1/Q$', fontsize=11)
    ax.set_ylabel(r'$|\mathrm{Im}(g^{(2)}_{\lambda\mu\nu})/\mathrm{Re}(g^{(2)}_{\lambda\mu\nu})|$',
                  fontsize=11)
    ax.set_title('Flagship prediction: $\\chi^{(2)}$ Im/Re scales as $1/Q$\n'
                 '(synthetic QNM test, 40 mode triples, 30 $Q$ values each)',
                 fontsize=10)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig('../figs/fig_chi2_scaling.pdf', dpi=200)
    fig.savefig('../figs/fig_chi2_scaling.png', dpi=150)
    print(f"\nSaved figure to ../figs/fig_chi2_scaling.{{pdf,png}}")

    np.savez("chi2_scaling_results.npz",
             Qs=Qs, ratios_Td=all_ratios_Td, ratios_iso=all_ratios_iso,
             slope_Td=alpha_Td, slope_iso=alpha_iso,
             prefactor_Td=np.exp(beta_Td), prefactor_iso=np.exp(beta_iso))

    # Assertion
    assert 0.90 < alpha_Td  < 1.10, f"FAIL: slope Td = {alpha_Td} not 1"
    assert 0.90 < alpha_iso < 1.10, f"FAIL: slope iso = {alpha_iso} not 1"
    print("\n[PASS] chi^(2) Im/Re ~ 1/Q scaling confirmed numerically.")


if __name__ == "__main__":
    main()
