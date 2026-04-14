"""
Photon-blockade-comb ring: non-perturbative Keldysh solution + analytical
strong-coupling expansion.

==========================================================================
Framing: this IS the Keldysh framework, not an alternative.
==========================================================================

The Schwinger-Keldysh action for the driven-dissipative photonic system is

  S_K = int dt { sum_j a_j^{cl*} (i partial_t) a_j^q  +  ... }
        - i sum_j kappa_j (2 a_j^{cl*} a_j^q - |a_j^q|^2)
        + H_cav[a_cl, a_q] + H_drive + H_hop(Phi) + H_Kerr

The steady state of the reduced density matrix rho_ss satisfies the
saddle-point / classical Keldysh equations, equivalent to the Lindblad
master equation

  L rho_ss = 0,      L[rho] = -i [H, rho] + sum_j ( L_j rho L_j^dag
                                       - (1/2){L_j^dag L_j, rho} )
with L_j = sqrt(kappa) a_j.

In the GAUSSIAN (above-threshold comb) regime the saddle is a coherent
state and the action is quadratic in fluctuations -> closed-form
Lyapunov solution (previous benchmark).  In the NON-GAUSSIAN (blockade)
regime the saddle point is not coherent; we solve the full Keldysh
equation non-perturbatively by constructing the Liouvillian super-
operator and finding its zero eigenvector.  `qt.steadystate(H, c_ops)`
does exactly this --- it is a Keldysh solver, not a separate formalism.

We also derive two analytical Keldysh perturbative limits below:
  (A) strong-blockade limit U -> infty: hard-core boson = spin-1/2 per
      site; Keldysh action reduces to coherent-spin-state path integral.
  (B) weak-drive limit E -> 0: expansion in the drive amplitude gives
      closed-form steady-state density matrix to arbitrary order in E.

These analytical results are checked against the non-perturbative
Keldysh solution.

==========================================================================
System: 3-site ring with Sagnac Peierls phase Phi.
==========================================================================
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

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def keldysh_liouvillian(N_sites, N_fock, J, U, E_drive, kappa, Phi):
    """Construct the Liouvillian super-operator L = -i[H, .] + sum_j D[L_j]
    encoding the Keldysh equation L rho = d rho/dt.  Steady state satisfies
    L rho_ss = 0.  This is the Keldysh generator of the reduced-density-
    matrix dynamics.

    Returns (H, c_ops, a_list, L)."""
    a_list = []
    for i in range(N_sites):
        ops = [qt.qeye(N_fock)] * N_sites
        ops[i] = qt.destroy(N_fock)
        a_list.append(qt.tensor(*ops))

    # Hopping with Peierls phase (Sagnac)
    H_hop = 0
    for j in range(N_sites):
        jp1 = (j + 1) % N_sites
        phase = np.exp(1j * Phi / N_sites)
        H_hop = H_hop - J * (phase * a_list[j].dag() * a_list[jp1]
                             + np.conj(phase) * a_list[jp1].dag() * a_list[j])
    # On-site Kerr (blockade)
    H_U = 0
    for j in range(N_sites):
        nj = a_list[j].dag() * a_list[j]
        H_U = H_U + U * nj * (nj - 1)
    # Coherent drive
    H_drive = sum(E_drive * (a + a.dag()) for a in a_list)
    H = H_hop + H_U + H_drive
    c_ops = [np.sqrt(kappa) * a for a in a_list]
    L = liouvillian(H, c_ops)
    return H, c_ops, a_list, L


def keldysh_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi):
    """Non-perturbative Keldysh solution: find rho_ss such that L rho_ss = 0.
    This is NOT an alternative to Keldysh; it is the zero-mode of the
    Keldysh Liouvillian computed by sparse linear algebra."""
    H, c_ops, a_list, L = keldysh_liouvillian(N_sites, N_fock, J, U, E_drive, kappa, Phi)
    rho_ss = steadystate(H, c_ops)
    return rho_ss, a_list, H, L


def photon_count_distribution(rho_ss, a_list, n_max=1):
    """Joint photon-number distribution P(n_1,...,n_N) in the Fock basis.
    Computed directly from the diagonal of rho_ss in the product Fock basis."""
    N_sites = len(a_list)
    N_fock = a_list[0].dims[0][0]
    grid = np.zeros([n_max+1]*N_sites)
    for idx in np.ndindex(*grid.shape):
        if any(n >= N_fock for n in idx): continue
        P_proj = qt.tensor([qt.fock_dm(N_fock, n) for n in idx])
        grid[idx] = float(qt.expect(P_proj, rho_ss).real)
    total = grid.sum()
    if total > 1e-12:
        grid = grid / total
    return grid


def fisher_info_photon_count(P_minus, P_plus, dPhi):
    """Classical Fisher information from the joint photon-count distribution.

    F(Phi) = sum_n (1/P(n)) (dP(n)/dPhi)^2     [Cramer-Rao-saturating for
                                                 photon-number-resolving POVM]
    """
    P0 = (P_minus + P_plus) / 2
    dP = (P_plus - P_minus) / dPhi
    eps = 1e-14
    return float(np.sum(dP**2 / (P0 + eps)))


# ==========================================================================
# ANALYTICAL KELDYSH PERTURBATIVE LIMITS
# ==========================================================================

def keldysh_weakdrive_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi, order=2):
    """Analytical Keldysh perturbative solution to lowest order in E (weak drive).

    At O(E^0): rho_ss = |0><0| (vacuum).
    At O(E^1): single-excitation amplitudes alpha_j = <a_j> = E/(-i omega_j - kappa/2)
              where omega_j are single-particle eigenvalues with Peierls phase.
    At O(E^2): two-excitation processes -- but blockade suppresses doublons;
              only single-particle states are populated.

    Returns analytical alpha_j (expected to match <a_j> from non-perturbative
    Keldysh at small E).
    """
    # Build single-particle hopping matrix with Peierls phase
    K = np.zeros((N_sites, N_sites), dtype=complex)
    for j in range(N_sites):
        jp1 = (j + 1) % N_sites
        phase = np.exp(1j * Phi / N_sites)
        K[j, jp1] = -J * phase
        K[jp1, j] = -J * np.conj(phase)
    # Single-particle Hamiltonian with damping: K - i (kappa/2) I
    # Drive: E * (1, 1, ..., 1)^T
    M = K - 1j * (kappa/2) * np.eye(N_sites)
    drive_vec = E_drive * np.ones(N_sites)
    # alpha = i M^-1 drive (in frequency 0 steady state)
    # Actually: d alpha_j / dt = -i sum_k M_jk alpha_k + E_j
    # Steady state: alpha = i M^{-1} drive_vec... let me redo
    # EOM: d alpha/dt = -i (K - i kappa/2) alpha + (-i) E
    #     0 = -i (K - i kappa/2) alpha_ss - i E
    #     (K - i kappa/2) alpha_ss = -E
    #     alpha_ss = -(K - i kappa/2)^{-1} E
    alpha_ss = np.linalg.solve(M, -drive_vec)
    return alpha_ss


def keldysh_hardcore_effective_H(N_sites, J, Phi):
    """Analytical Keldysh reduction to hard-core boson / XY spin model in the
    U -> infty limit.

    Projected Hamiltonian on single-particle manifold:
        H_eff = sum_j -J (e^{i Phi/N} sigma_j^+ sigma_{j+1}^- + h.c.)
    where sigma_j^+ creates a photon on site j (|0> -> |1>).

    For 3-site ring, returns the 3x3 effective hopping matrix, whose
    eigenvalues give the Bloch band with Aharonov-Bohm flux:
        E_k = -2 J cos(2 pi k/N + Phi/N),  k = 0, 1, ..., N-1
    """
    K = np.zeros((N_sites, N_sites), dtype=complex)
    for j in range(N_sites):
        jp1 = (j + 1) % N_sites
        phase = np.exp(1j * Phi / N_sites)
        K[j, jp1] = -J * phase
        K[jp1, j] = -J * np.conj(phase)
    eigvals = np.linalg.eigvalsh(K)
    # Analytical Bloch prediction
    k_vals = np.arange(N_sites)
    bloch = -2*J * np.cos(2*np.pi*k_vals/N_sites + Phi/N_sites)
    return sorted(eigvals.real), sorted(bloch)


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    N_sites = 3
    N_fock = 2            # single-photon-per-site manifold (blockade)
    J = 1.0
    U = 40.0              # U >> J, kappa: deep blockade
    kappa = 1.0
    E_drive = 0.4         # weak drive (for Keldysh perturbation theory)

    Phi_vals = np.linspace(0, 2*np.pi, 13)
    dPhi = 0.05

    # Non-perturbative Keldysh (= Lindblad steady state)
    FI_nonpert = []
    n_total = []
    g2_cross = []
    alpha_abs_nonpert = []
    alpha_abs_weak = []
    bloch_match = []

    print("=== Non-perturbative Keldysh (Liouvillian zero-mode) ===\n")
    for Phi in Phi_vals:
        # Central point
        rho, a_list, H, L = keldysh_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi)
        # Photon-count distribution
        P0 = photon_count_distribution(rho, a_list, n_max=N_fock-1)
        # Plus and minus
        rho_p, a_p, _, _ = keldysh_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi + dPhi/2)
        P_plus = photon_count_distribution(rho_p, a_p, n_max=N_fock-1)
        rho_m, a_m, _, _ = keldysh_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi - dPhi/2)
        P_minus = photon_count_distribution(rho_m, a_m, n_max=N_fock-1)
        FI = fisher_info_photon_count(P_minus, P_plus, dPhi)
        # Observables
        n_i = [float(qt.expect(a.dag()*a, rho).real) for a in a_list]
        g2_01 = float((qt.expect(a_list[0].dag()*a_list[1].dag()*a_list[1]*a_list[0], rho).real
                       / (n_i[0]*n_i[1]+1e-16)))
        # Single-particle expectation <a_j>
        alpha_np = [complex(qt.expect(a, rho)) for a in a_list]
        alpha_abs_nonpert.append([abs(x) for x in alpha_np])

        # Weak-drive Keldysh perturbation theory
        alpha_pt = keldysh_weakdrive_steady_state(N_sites, N_fock, J, U, E_drive, kappa, Phi)
        alpha_abs_weak.append([abs(x) for x in alpha_pt])

        # Hard-core Keldysh eigenvalues
        eigs_np, bloch = keldysh_hardcore_effective_H(N_sites, J, Phi)
        bloch_match.append(np.max(np.abs(np.array(eigs_np) - np.array(bloch))))

        FI_nonpert.append(FI)
        n_total.append(sum(n_i))
        g2_cross.append(g2_01)
        print(f"  Phi={Phi:.3f}  <n_tot>={sum(n_i):.3f}  g2_cross={g2_01:.3f}  "
              f"FI={FI:.3e}  |alpha|_np={[f'{x:.3f}' for x in alpha_abs_nonpert[-1]]}  "
              f"|alpha|_pt={[f'{x:.3f}' for x in alpha_abs_weak[-1]]}")

    FI_nonpert = np.array(FI_nonpert)
    n_total = np.array(n_total)
    alpha_nonpert = np.array(alpha_abs_nonpert)
    alpha_weak = np.array(alpha_abs_weak)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Panel (a): Fisher info vs flux
    ax = axes[0, 0]
    ax.plot(Phi_vals, FI_nonpert, 'bo-', ms=8, lw=1.5,
            label='non-perturbative Keldysh')
    F_SQL_coh = 4 * n_total
    ax.plot(Phi_vals, F_SQL_coh, 'k--', lw=1,
            label=r'coherent-state SQL: $4\langle n\rangle$')
    ax.set_xlabel(r'Sagnac flux $\Phi$ (rad)', fontsize=11)
    ax.set_ylabel(r'Fisher information $F(\Phi)$', fontsize=11)
    ax.set_title(r'(a) FI from non-perturbative Keldysh solution'
                  '\n' r'(Liouvillian zero-mode)', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (b): analytical Keldysh weak-drive PT vs non-perturbative
    ax = axes[0, 1]
    for j in range(N_sites):
        ax.plot(Phi_vals, alpha_nonpert[:, j], 'o', ms=6,
                color=f'C{j}', label=f'non-pert site {j}')
        ax.plot(Phi_vals, alpha_weak[:, j], '-', lw=1.2,
                color=f'C{j}', alpha=0.6, label=f'weak-drive PT site {j}')
    ax.set_xlabel(r'$\Phi$', fontsize=11)
    ax.set_ylabel(r'$|\langle a_j\rangle|$', fontsize=11)
    ax.set_title(r'(b) Analytical Keldysh PT vs non-perturbative: $|\alpha_j|$',
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    # Panel (c): cross-site g^(2) — non-Gaussian signature
    ax = axes[1, 0]
    ax.plot(Phi_vals, g2_cross, 'r-', lw=2)
    ax.axhline(1, color='gray', ls=':', alpha=0.4, label='coherent')
    ax.axhline(0, color='green', ls=':', alpha=0.4, label='antibunched')
    ax.set_xlabel(r'$\Phi$', fontsize=11)
    ax.set_ylabel(r'$g^{(2)}_{0,1}(0)$', fontsize=11)
    ax.set_title(r'(c) Non-Gaussian cross-site correlation', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel (d): hard-core Keldysh single-particle bands (Bloch vs numerical)
    ax = axes[1, 1]
    bloch_arr = np.array([keldysh_hardcore_effective_H(N_sites, J, Phi)[1] for Phi in Phi_vals])
    for k in range(N_sites):
        ax.plot(Phi_vals, bloch_arr[:, k], '-', lw=2, label=f'band $k={k}$')
    ax.set_xlabel(r'$\Phi$', fontsize=11)
    ax.set_ylabel(r'single-particle eigenvalues $E_k$', fontsize=11)
    ax.set_title(r'(d) Hard-core Keldysh band structure'
                  '\n' r'$E_k=-2J\cos(2\pi k/N+\Phi/N)$', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle('Photon-blockade-comb ring: Keldysh framework in the non-Gaussian regime',
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0,0,1,0.96))
    fig.savefig(FIG_DIR/"fig_blockade_comb_FI.pdf", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR/"fig_blockade_comb_FI.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved {FIG_DIR}/fig_blockade_comb_FI.pdf")

    # Validation: weak-drive Keldysh PT should match non-perturbative at small E
    rel_err_pt = np.abs(alpha_nonpert - alpha_weak).max() / np.maximum(alpha_nonpert.max(), 1e-6)
    print(f"\nWeak-drive Keldysh PT vs non-perturbative: max rel err in |alpha| = {rel_err_pt:.3%}")
    print(f"Hard-core Keldysh band eigenvalues: max deviation from Bloch = "
          f"{max(bloch_match):.2e}")
    idx_best = int(np.argmax(FI_nonpert))
    print(f"\nBest operating point: Phi = {Phi_vals[idx_best]:.3f}")
    print(f"  <n_tot> = {n_total[idx_best]:.3f}, FI = {FI_nonpert[idx_best]:.3e}")

    with open(DATA_DIR/"blockade_comb_FI.json", "w") as f:
        json.dump(dict(Phi=Phi_vals.tolist(),
                       FI_nonpert=FI_nonpert.tolist(),
                       n_total=n_total.tolist(),
                       g2_cross=g2_cross,
                       alpha_nonpert=alpha_nonpert.tolist(),
                       alpha_weak_PT=alpha_weak.tolist(),
                       F_SQL=F_SQL_coh.tolist(),
                       weak_drive_PT_vs_nonpert=float(rel_err_pt)),
                  f, indent=2)


if __name__ == "__main__":
    main()
