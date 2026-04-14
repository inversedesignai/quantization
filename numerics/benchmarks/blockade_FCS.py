"""
Route (c): Full Counting Statistics via tilted Liouvillian.

The classical Fisher information for the photon-counting POVM in the
Keldysh framework is computed WITHOUT ever constructing the full density
matrix: the tilted Liouvillian

    L_lambda[rho] = -i[H(Phi), rho] + sum_j ( e^{i lambda_j} L_j rho L_j^dag
                                              - (1/2) {L_j^dag L_j, rho} )

has the property that its dominant eigenvalue theta_max(lambda) is the
cumulant-generating function of the photon count distribution:

    ln M(lambda, t) = theta_max(lambda) * t + constant,

where M(lambda) = <e^{i lambda . N}> and N_j is the number of photons
emitted through channel j up to time t.  This is the Full Counting
Statistics (FCS) framework of Levitov-Lesovik, extended to open
quantum systems (see Esposito/Harbola/Mukamel 2009).

For steady-state counting:
  - First cumulant <N_j>/t = d theta_max / d(i lambda_j) | lambda=0
    = steady-state photon-flux through channel j
  - Fisher information for estimation of a parameter Phi entering H:
      F(Phi) = <(d ln P/dPhi)^2>
    which can be computed from theta_max(lambda, Phi) via its
    lambda-derivatives and the partial derivative with Phi.

For a time-window T measurement of photon counts, the CGF is
theta_max(lambda) * T.  The Fisher information for Phi then reads:

    F(Phi; T) = T * sum_j |d^2 theta_max / d lambda_j d Phi|^2_{lambda=0}
                / (d^2 theta_max / d lambda_j^2)_{lambda=0}

(from the Gaussian approximation to the counting distribution at long T;
full distribution requires inverse Fourier transform of e^{theta(lambda)T}).

This is a Keldysh-native non-perturbative numerical method:
  - Construct the tilted super-operator L_lambda (same dim as L_0).
  - Find largest eigenvalue theta_max(lambda) -> cumulant-generating fn.
  - TopOpt gradient = derivative of theta_max w.r.t. design parameters,
    via standard non-Hermitian eigenvalue perturbation theory:
      d theta_max / d theta = <l_max | (d L / d theta) | r_max>
    where l, r are left/right eigenvectors of L_lambda.
    -> Gradient cost = ONE eigensolve per design parameter.

This is the analog for open systems of "ground-state FI gradient" used in
closed-system quantum Fisher information TopOpt.

Demonstration on the 3-site blockade ring: compute the photon-flux
cumulant-generating function via the tilted Liouvillian, extract the
Fisher info for Sagnac flux Phi, compare to direct photon-counting FI.
"""
from __future__ import annotations
import numpy as np
import qutip as qt
from qutip import liouvillian, steadystate
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
from scipy.sparse.linalg import eigs
from scipy.linalg import eig

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi):
    """Same as blockade_comb_FI.py: 3-site ring with Peierls Sagnac phase."""
    a_list = []
    for i in range(N_sites):
        ops = [qt.qeye(N_fock)] * N_sites
        ops[i] = qt.destroy(N_fock)
        a_list.append(qt.tensor(*ops))
    H_hop = 0
    for j in range(N_sites):
        jp1 = (j + 1) % N_sites
        phase = np.exp(1j * Phi / N_sites)
        H_hop = H_hop - J * (phase * a_list[j].dag() * a_list[jp1]
                             + np.conj(phase) * a_list[jp1].dag() * a_list[j])
    H_U = 0
    for j in range(N_sites):
        nj = a_list[j].dag() * a_list[j]
        H_U = H_U + U * nj * (nj - 1)
    H_drive = sum(E_drive * (a + a.dag()) for a in a_list)
    H = H_hop + H_U + H_drive
    c_ops = [np.sqrt(kappa) * a for a in a_list]
    return H, c_ops, a_list


def tilted_liouvillian(H, c_ops, lambdas):
    """Construct the tilted Liouvillian L_lambda.

    L_lambda[rho] = -i[H, rho] + sum_j( exp(i lambda_j) c_j rho c_j^dag
                                         - 0.5 {c_j^dag c_j, rho} ).
    Returns a super-operator that can be converted to a matrix.
    """
    # standard Lindbladian
    L_0 = liouvillian(H, c_ops).full()
    D = H.shape[0]
    # Correction for tilting: the "jump" part sum_j exp(i lambda_j) c_j rho c_j^dag
    # needs e^{i lambda_j} - 1 prefactor on the jump term (the dissipator term
    # is unchanged, but the EXPECTED-JUMP term picks up the tilt).
    # Easier: rebuild the tilted Lindbladian from scratch.
    L_tilt = -1j * (qt.spre(H).full() - qt.spost(H).full())
    for lam, c in zip(lambdas, c_ops):
        spre_c = qt.spre(c).full()
        spost_cdag = qt.spost(c.dag()).full()
        cdag_c = (c.dag() * c)
        spre_cdag_c = qt.spre(cdag_c).full()
        spost_cdag_c = qt.spost(cdag_c).full()
        # Tilted jump: e^{i lambda} c rho c^dag
        jump = np.exp(1j*lam) * (spre_c @ spost_cdag)
        L_tilt += jump - 0.5 * (spre_cdag_c + spost_cdag_c)
    return L_tilt


def largest_eigenvalue(L_matrix, num=1):
    """Find the largest-real-part eigenvalue of a (generally non-Hermitian) matrix.
    For steady state (L_0), this is 0.  For tilted L_lambda, it's the
    cumulant-generating function.
    """
    if L_matrix.shape[0] < 60:
        # dense
        w, vl, vr = eig(L_matrix, left=True, right=True)
        idx = np.argmax(w.real)
        return w[idx], vr[:, idx], vl[:, idx]
    else:
        # sparse
        from scipy.sparse.linalg import eigs as seigs
        w, vr = seigs(L_matrix, k=num, which='LR')
        return w[0], vr[:, 0], None


def cumulant_generating_function(H, c_ops, lambdas):
    """theta_max(lambda) = largest eigenvalue of L_lambda."""
    L = tilted_liouvillian(H, c_ops, lambdas)
    theta, r, l = largest_eigenvalue(L)
    return theta, r, l


def fisher_from_CGF(H_central, H_plus, H_minus, c_ops, dPhi, T=1.0):
    """Compute Fisher information from the tilted-Liouvillian CGF.

    In the long-time limit, the counts {N_j} over time T have mean T*<j_j>
    and covariance T*C_{jj'} where C_{jj'} = d^2 theta / d(i lambda_j) d(i lambda_{j'})|_{0}.
    For a SCALAR parameter Phi, the Fisher information from the Gaussian
    counts is
      F(Phi) = T * [d<j>/dPhi]^T C^{-1} [d<j>/dPhi]
    where <j> = d theta/d(i lambda)|_0 is the vector of mean photon fluxes.

    We compute d theta / d(i lambda_j) by finite difference at lambda=0.
    """
    N_ch = len(c_ops)
    # baseline (untilted)
    theta_0, _, _ = cumulant_generating_function(H_central, c_ops, np.zeros(N_ch))
    dlam = 1e-3
    # First derivatives: mean fluxes
    j_mean = np.zeros(N_ch)
    for ch in range(N_ch):
        lam_p = np.zeros(N_ch); lam_p[ch] = dlam
        theta_p, _, _ = cumulant_generating_function(H_central, c_ops, lam_p)
        j_mean[ch] = (theta_p - theta_0).imag / dlam
    # Second derivatives: covariance matrix
    C = np.zeros((N_ch, N_ch))
    for a_ in range(N_ch):
        for b_ in range(a_, N_ch):
            lam_pp = np.zeros(N_ch); lam_pp[a_] += dlam; lam_pp[b_] += dlam
            lam_pm = np.zeros(N_ch); lam_pm[a_] += dlam; lam_pm[b_] -= dlam
            lam_mp = np.zeros(N_ch); lam_mp[a_] -= dlam; lam_mp[b_] += dlam
            lam_mm = np.zeros(N_ch); lam_mm[a_] -= dlam; lam_mm[b_] -= dlam
            theta_pp, _, _ = cumulant_generating_function(H_central, c_ops, lam_pp)
            theta_pm, _, _ = cumulant_generating_function(H_central, c_ops, lam_pm)
            theta_mp, _, _ = cumulant_generating_function(H_central, c_ops, lam_mp)
            theta_mm, _, _ = cumulant_generating_function(H_central, c_ops, lam_mm)
            # d^2 theta / d(i lambda_a) d(i lambda_b):
            # theta is in general complex, but for real POVM outputs the
            # cumulants are real.
            mixed = ((theta_pp - theta_pm - theta_mp + theta_mm) / (4*dlam**2))
            # In i-lambda variables, d^2/d(i lambda)^2 = -d^2/d(lambda)^2
            C[a_, b_] = -mixed.real
            C[b_, a_] = C[a_, b_]
    # Derivative of mean flux w.r.t. Phi (finite diff through H)
    dj_dPhi = np.zeros(N_ch)
    for ch in range(N_ch):
        lam_p = np.zeros(N_ch); lam_p[ch] = dlam
        theta_p_plus, _, _ = cumulant_generating_function(H_plus, c_ops, lam_p)
        theta_0_plus, _, _ = cumulant_generating_function(H_plus, c_ops, np.zeros(N_ch))
        j_plus = (theta_p_plus - theta_0_plus).imag / dlam
        theta_p_minus, _, _ = cumulant_generating_function(H_minus, c_ops, lam_p)
        theta_0_minus, _, _ = cumulant_generating_function(H_minus, c_ops, np.zeros(N_ch))
        j_minus = (theta_p_minus - theta_0_minus).imag / dlam
        dj_dPhi[ch] = (j_plus - j_minus) / dPhi
    # Gaussian Fisher info (long-time limit)
    try:
        C_inv = np.linalg.inv(C)
        F = T * float(dj_dPhi @ C_inv @ dj_dPhi)
    except np.linalg.LinAlgError:
        F = float('nan')
    return F, j_mean, C, dj_dPhi


def main():
    N_sites = 3
    N_fock = 2
    J = 1.0
    U = 40.0
    kappa = 1.0
    E_drive = 0.4

    Phi_vals = np.linspace(0, 2*np.pi, 13)
    dPhi = 0.05
    T_window = 10.0   # detection time window

    FI_FCS = []
    flux_totals = []
    cov_diag = []

    print("=== FCS tilted-Liouvillian Fisher information ===\n")
    for Phi in Phi_vals:
        H, c_ops, a_list = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi)
        H_p, _, _ = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi + dPhi/2)
        H_m, _, _ = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi - dPhi/2)
        F, j_mean, C, dj_dPhi = fisher_from_CGF(H, H_p, H_m, c_ops, dPhi, T=T_window)
        FI_FCS.append(F)
        flux_totals.append(float(j_mean.sum()))
        cov_diag.append(float(C[0, 0]))
        print(f"  Phi={Phi:.3f}  <j_tot>={j_mean.sum():.3f}  "
              f"C[0,0]={C[0,0]:.3f}  dj/dPhi={dj_dPhi[0]:.3e}  "
              f"F={F:.3e}")

    FI_FCS = np.array(FI_FCS)

    # Cross-check against direct photon-counting FI from earlier script
    # (full-distribution method)
    print("\n=== Cross-check: direct photon-counting FI ===")
    from blockade_comb_FI import (keldysh_steady_state,
                                    photon_count_distribution,
                                    fisher_info_photon_count)
    FI_direct = []
    for Phi in Phi_vals:
        rho_p, a_p, _, _ = keldysh_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi + dPhi/2)
        rho_m, a_m, _, _ = keldysh_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi - dPhi/2)
        P_p = photon_count_distribution(rho_p, a_p, n_max=N_fock-1)
        P_m = photon_count_distribution(rho_m, a_m, n_max=N_fock-1)
        F_direct = fisher_info_photon_count(P_m, P_p, dPhi)
        FI_direct.append(F_direct)
        print(f"  Phi={Phi:.3f}  FI_direct={F_direct:.3e}  FI_FCS={FI_FCS[int(np.argmin(np.abs(Phi_vals-Phi)))]:.3e}")

    FI_direct = np.array(FI_direct)

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    ax = axes[0]
    ax.plot(Phi_vals, FI_FCS, 'ro-', ms=8,
            label=r'FCS tilted-Liouvillian ($T\!=\!10/\kappa$)')
    ax.plot(Phi_vals, FI_direct * T_window, 'bx--', ms=9, lw=1,
            label='direct photon-count (scaled by $T$)')
    ax.set_xlabel(r'Sagnac flux $\Phi$', fontsize=11)
    ax.set_ylabel(r'Fisher information $F(\Phi)$', fontsize=11)
    ax.set_title('FCS vs direct photon-count FI\n(long-time Gaussian limit)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(Phi_vals, flux_totals, 'g-', lw=2, label=r'$\sum_j \langle j_j\rangle$ (photon flux)')
    ax.plot(Phi_vals, cov_diag, 'm-', lw=2, label=r'$C_{00}$ (flux variance)')
    ax.set_xlabel(r'$\Phi$', fontsize=11)
    ax.set_ylabel('steady-state flux / variance', fontsize=11)
    ax.set_title('Flux and variance from tilted-Liouvillian derivatives',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Route (c): Keldysh Full Counting Statistics via tilted Liouvillian',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_blockade_FCS.pdf", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR/"fig_blockade_FCS.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved {FIG_DIR}/fig_blockade_FCS.pdf")

    with open(DATA_DIR/"blockade_FCS.json", "w") as f:
        json.dump(dict(Phi=Phi_vals.tolist(),
                       FI_FCS=FI_FCS.tolist(),
                       FI_direct=FI_direct.tolist(),
                       flux_totals=flux_totals,
                       cov_diag=cov_diag), f, indent=2)


if __name__ == "__main__":
    main()
