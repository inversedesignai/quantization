"""
End-to-end Keldysh FI for a Sagnac N-mode Kerr ring, with no linearization.
==========================================================================

Scaling up the kerr_bistability_FI.py demonstration to an actual ring
geometry (inertial-sensor topology), and pushing the number of modes N
up to the limit of what qutip's exact Liouvillian solve can handle --
then pushing further with MCWF (Monte Carlo wavefunction) unraveling of
the same Keldysh generator, which only stores a state-vector of size D
(not a rho of size D^2) and scales to many more modes.

Model: N-site ring with
    H(Phi) = sum_j [ -Delta n_j + (U/2) n_j(n_j - 1) + eps (a_j + a_j^dag) ]
           - J sum_j [ e^{i Phi/N} a_j^dag a_{j+1} + h.c. ]
    L_j   = sqrt(kappa) a_j                    (per-site photon loss)
Sagnac parameter: Phi (Peierls phase threading the ring; proportional
to rotation rate via Phi = 2 m_p * Omega * tau).

Parameter regime: blockade / strongly-correlated (U >> kappa) so that
the state is genuinely non-Gaussian and Bogoliubov is guaranteed to
fail.  We estimate Phi from photon-number-resolving detection at each
ring site: P(n_1, ..., n_N | Phi) = <n_1 ... n_N | rho_ss(Phi) | n_1 ... n_N>.
No closed form expected.

Pipeline:
  (A) Small N (N = 2, 3, 4):
       1. Build Lindbladian L(Phi).
       2. Solve L[rho] = 0 by sparse super-operator diagonalization.
       3. Read off P(n|Phi) from the diagonal of rho.
       4. F_cl(Phi) by finite-difference over Phi.
       5. F_Q(Phi) by Uhlmann fidelity.

  (B) Larger N (N = 3, 4, 5, 6):
       1. Build H, c_ops (same generator).
       2. MCWF: sample M trajectories from the Keldysh unraveling.
       3. Empirical P_emp(n|Phi) by histogramming final states.
       4. F_emp(Phi) by common-random-numbers (CRN) gradient estimator.
       5. Compare (B) to (A) at overlap (N = 3, 4).

Report: wall time, memory footprint, and accuracy of F(Phi) from the
pure quantum generator as a function of N.
"""
from __future__ import annotations
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time
import gc

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_ring_operators(N_sites, N_fock):
    """Return list of site annihilation operators a_j in the
    N_sites-fold tensor product with Fock truncation N_fock."""
    a_list = []
    for j in range(N_sites):
        ops = [qt.qeye(N_fock)] * N_sites
        ops[j] = qt.destroy(N_fock)
        a_list.append(qt.tensor(*ops))
    return a_list


def build_hamiltonian(a_list, Delta, U, eps, J, Phi):
    """H = sum_j[ -Delta n + (U/2)n(n-1) + eps(a+a^dag) ]
           - J sum_j [ e^{iPhi/N} a_j^dag a_{j+1} + h.c. ].
    """
    N = len(a_list)
    phase = np.exp(1j * Phi / N)
    H = 0
    for j in range(N):
        n_j = a_list[j].dag() * a_list[j]
        H = H + (-Delta) * n_j + (U / 2.0) * n_j * (n_j - 1) \
              + eps * (a_list[j] + a_list[j].dag())
    for j in range(N):
        k = (j + 1) % N
        H = H - J * (phase * a_list[j].dag() * a_list[k]
                     + np.conj(phase) * a_list[k].dag() * a_list[j])
    return H


def build_Lindblad_system(N_sites, N_fock, Delta, U, eps, J, kappa, Phi):
    a_list = build_ring_operators(N_sites, N_fock)
    H = build_hamiltonian(a_list, Delta, U, eps, J, Phi)
    c_ops = [np.sqrt(kappa) * a for a in a_list]
    return H, c_ops, a_list


# ==================================================================
# Route A: Exact Liouvillian solve
# ==================================================================

def ED_steady_state(N_sites, N_fock, Delta, U, eps, J, kappa, Phi):
    H, c_ops, a_list = build_Lindblad_system(N_sites, N_fock, Delta, U,
                                              eps, J, kappa, Phi)
    rho = qt.steadystate(H, c_ops)
    return rho, a_list


def joint_photon_distribution(rho, N_sites, N_fock):
    """P(n_1, ..., n_N) = <n_1, ..., n_N | rho | n_1, ..., n_N>."""
    diag = np.real(np.diag(rho.full()))
    return diag.reshape([N_fock] * N_sites)


def FI_classical_from_rho(rho_p, rho_m, rho_0, dPhi, N_sites, N_fock):
    P_p = joint_photon_distribution(rho_p, N_sites, N_fock).flatten()
    P_m = joint_photon_distribution(rho_m, N_sites, N_fock).flatten()
    P_0 = joint_photon_distribution(rho_0, N_sites, N_fock).flatten()
    dP = (P_p - P_m) / (2 * dPhi)
    mask = P_0 > 1e-12
    return float(np.sum((dP[mask] ** 2) / P_0[mask]))


def FI_quantum_from_rho(rho_p, rho_m, dPhi):
    """Uhlmann-fidelity QFI estimate: F_Q = 8 (1-sqrt(F))/ (2 dPhi)^2."""
    try:
        sqrt_m = rho_m.sqrtm()
        inner = sqrt_m * rho_p * sqrt_m
        Fid = (inner.sqrtm()).tr()
        Fid = float(np.real(Fid)) ** 2
        Fid = max(min(Fid, 1.0), 0.0)
        F_Q = 8.0 * (1.0 - np.sqrt(Fid)) / (2 * dPhi) ** 2
    except Exception:
        F_Q = float('nan')
    return F_Q


def run_route_A(N_sites, N_fock, Delta, U, eps, J, kappa, Phi_vals, dPhi):
    print(f"\n--- Route A (ED) N_sites={N_sites}, N_fock={N_fock} ---")
    print(f"  Hilbert dim D = {N_fock ** N_sites},"
          f"  Liouvillian dim L = {(N_fock ** N_sites) ** 2}")
    F_cl, F_Q = [], []
    t0 = time.time()
    for Phi in Phi_vals:
        rho_p, _ = ED_steady_state(N_sites, N_fock, Delta, U, eps, J, kappa,
                                     Phi + dPhi)
        rho_m, _ = ED_steady_state(N_sites, N_fock, Delta, U, eps, J, kappa,
                                     Phi - dPhi)
        rho_0, _ = ED_steady_state(N_sites, N_fock, Delta, U, eps, J, kappa,
                                     Phi)
        F_cl.append(FI_classical_from_rho(rho_p, rho_m, rho_0, dPhi,
                                           N_sites, N_fock))
        F_Q.append(FI_quantum_from_rho(rho_p, rho_m, dPhi))
        gc.collect()
    wall = time.time() - t0
    print(f"  wall time: {wall:.2f} s for {len(Phi_vals)} points")
    return np.array(F_cl), np.array(F_Q), wall


# ==================================================================
# Route B: MCWF with Common Random Numbers
# ==================================================================

def MCWF_distribution(H, c_ops, a_list, psi0, T_eq, N_traj, seeds, N_fock,
                      N_sites):
    """Run MCWF trajectories and return the empirical
    P(n_1, ..., n_N) = <n_1,...,n_N | rho_ens(T_eq) | n_1,...,n_N>
    where rho_ens = (1/N_traj) sum_m |psi_m><psi_m|.

    In qutip 5.x, mcsolve(...).average_states[-1] IS the ensemble-averaged
    density matrix at the final time.  The diagonal in the Fock basis is
    exactly the joint photon-number distribution.

    CRN variance reduction is preserved because the *same seed list* is
    passed to the two +/- perturbation runs -- identical trajectory-level
    noise histories, so the difference (P_plus - P_minus) has low variance.
    """
    tlist = [0.0, T_eq]
    opts = dict(nsteps=50000, keep_runs_results=False)
    # e_ops irrelevant; we just want the final density matrix
    result = qt.mcsolve(H, psi0, tlist, c_ops, [], ntraj=N_traj,
                         options=opts, seeds=seeds)
    rho_ens = result.average_states[-1]
    if not rho_ens.isoper:
        rho_ens = rho_ens * rho_ens.dag()
    diag = np.real(np.diag(rho_ens.full()))
    P = diag.reshape([N_fock] * N_sites)
    return P


def MCWF_FI_CRN(N_sites, N_fock, Delta, U, eps, J, kappa, Phi,
                 dPhi, T_eq, N_traj, seed_base):
    """Common-random-numbers estimator of d P(n)/dPhi.

    Uses the SAME seeds for the trajectories at Phi+dPhi/2 and
    Phi-dPhi/2, giving a low-variance pathwise-like finite
    difference.  The value and dF estimate come from one forward
    pass of 2*N_traj trajectories total.
    """
    H_p, c_p, a_p = build_Lindblad_system(N_sites, N_fock, Delta, U, eps, J,
                                           kappa, Phi + dPhi / 2)
    H_m, c_m, a_m = build_Lindblad_system(N_sites, N_fock, Delta, U, eps, J,
                                           kappa, Phi - dPhi / 2)
    H_0, c_0, a_0 = build_Lindblad_system(N_sites, N_fock, Delta, U, eps, J,
                                           kappa, Phi)
    psi0 = qt.tensor(*[qt.basis(N_fock, 0) for _ in range(N_sites)])
    # Use same seeds for + and - (CRN) for variance reduction
    P_p = MCWF_distribution(H_p, c_p, a_p, psi0, T_eq, N_traj,
                              seed_base, N_fock, N_sites)
    P_m = MCWF_distribution(H_m, c_m, a_m, psi0, T_eq, N_traj,
                              seed_base, N_fock, N_sites)
    # Baseline = mean of the two CRN-paired runs.  This keeps the
    # denominator consistent with the numerator (dP) in the shot
    # noise, giving a lower-variance F estimator.
    P_0 = 0.5 * (P_p + P_m)
    dP = (P_p - P_m) / dPhi
    P_0f = P_0.flatten()
    dPf = dP.flatten()
    # Laplace regularization to protect against zero-count bins.
    regularization = 0.5 / N_traj
    P_0f_reg = P_0f + regularization
    F = float(np.sum((dPf ** 2) / P_0f_reg))
    return F, P_0, P_p, P_m


def run_route_B(N_sites, N_fock, Delta, U, eps, J, kappa, Phi_vals, dPhi,
                T_eq, N_traj):
    print(f"\n--- Route B (MCWF) N_sites={N_sites}, N_fock={N_fock}, "
          f"N_traj={N_traj} ---")
    print(f"  Hilbert dim D = {N_fock ** N_sites} (MCWF needs only D)")
    F_mcwf = []
    t0 = time.time()
    for i, Phi in enumerate(Phi_vals):
        F, P0, _, _ = MCWF_FI_CRN(N_sites, N_fock, Delta, U, eps, J, kappa,
                                    Phi, dPhi, T_eq, N_traj,
                                    seed_base=100 * (i + 1))
        print(f"  Phi={Phi:.2f}  F_mcwf={F:.3e}  (max P0={P0.max():.3f}, "
              f"populated bins={int(np.sum(P0>0))})")
        F_mcwf.append(F)
        gc.collect()
    wall = time.time() - t0
    print(f"  wall time: {wall:.2f} s for {len(Phi_vals)} points "
          f"({wall/len(Phi_vals):.2f} s/point, "
          f"{wall/(2 * N_traj * len(Phi_vals))*1e3:.2f} ms/trajectory)")
    return np.array(F_mcwf), wall


# ==================================================================
# Main demonstration
# ==================================================================

def main():
    # Physical parameters (in units of kappa)
    kappa = 1.0
    Delta = -0.5       # red-detuned drive for higher occupation
    U = 5.0            # moderate Kerr (strongly non-Gaussian but not frozen)
    eps = 2.0          # strong drive
    J = 1.0            # hopping
    dPhi = 0.05        # for ED
    dPhi_MC = 0.30     # for MCWF (needs larger step to overcome shot noise)

    # Sweep grid for Sagnac flux
    Phi_vals = np.array([0.5, 1.5, 2.5, 3.5, 4.5])

    results = {}

    # ---------------------- Route A: ED for small N --------------------
    ED_sizes = [(2, 4), (3, 3), (4, 2)]    # (N_sites, N_fock)
    for N_sites, N_fock in ED_sizes:
        F_cl, F_Q, wall = run_route_A(N_sites, N_fock, Delta, U, eps, J,
                                        kappa, Phi_vals, dPhi)
        results[f"ED_N{N_sites}_F{N_fock}"] = dict(
            method="ED", N_sites=N_sites, N_fock=N_fock, wall=wall,
            F_cl=F_cl.tolist(), F_Q=F_Q.tolist())
        print(f"  Phi={Phi_vals.tolist()}")
        print(f"  F_cl (ED)  = {np.array2string(F_cl, precision=3)}")
        print(f"  F_Q  (ED)  = {np.array2string(F_Q, precision=3)}")

    # ---------------------- Route B: MCWF for larger N ----------------
    T_eq = 15.0      # equilibration time (multiples of 1/kappa)
    N_traj = 300     # trajectories per parameter point per phi offset
    # Overlap N=3 with ED to cross-check, then go to N=4, 5, 6, 7
    MCWF_sizes = [(3, 3, 500), (4, 2, 500), (5, 2, 400), (6, 2, 300),
                   (7, 2, 200)]
    for N_sites, N_fock, Ntr in MCWF_sizes:
        F_mcwf, wall = run_route_B(N_sites, N_fock, Delta, U, eps, J, kappa,
                                     Phi_vals, dPhi_MC, T_eq, Ntr)
        results[f"MCWF_N{N_sites}_F{N_fock}"] = dict(
            method="MCWF", N_sites=N_sites, N_fock=N_fock, N_traj=Ntr,
            T_eq=T_eq, wall=wall, F_mcwf=F_mcwf.tolist())
        print(f"  F_mcwf (N={N_sites}) = {np.array2string(F_mcwf, precision=3)}")

    # ---------------------- Plot ---------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    markers = ['o', 's', '^', 'd']
    for i, (N_sites, N_fock) in enumerate(ED_sizes):
        r = results[f"ED_N{N_sites}_F{N_fock}"]
        ax.plot(Phi_vals, r['F_cl'], '-' + markers[i], ms=7, lw=1.8,
                label=fr'ED $N={N_sites}$, $N_F={N_fock}$')
    ax.set_xlabel(r'Sagnac Peierls flux $\Phi$', fontsize=11)
    ax.set_ylabel(r'$F_{\rm cl}(\Phi)$ (PNR, exact)', fontsize=11)
    ax.set_title('(a) Classical Fisher info $F_{\\rm cl}$\n'
                 'from exact $\\rho_{ss}(\\Phi)$, small $N$', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    for i, (N_sites, N_fock) in enumerate(ED_sizes):
        r = results[f"ED_N{N_sites}_F{N_fock}"]
        ax.plot(Phi_vals, r['F_Q'], '-' + markers[i], ms=7, lw=1.8,
                label=fr'ED $N={N_sites}$, $N_F={N_fock}$')
    ax.set_xlabel(r'$\Phi$', fontsize=11)
    ax.set_ylabel(r'$F_Q(\Phi)$ (QFI, Uhlmann)', fontsize=11)
    ax.set_title('(b) Quantum Fisher info $F_Q$\n'
                 'best any POVM could achieve', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    # ED vs MCWF at overlapping N (N=3 N_fock=3; N=4 N_fock=2)
    # Plot both
    for idx_ed, (N_sites, N_fock) in enumerate(ED_sizes):
        r_ed = results.get(f"ED_N{N_sites}_F{N_fock}")
        r_mc = results.get(f"MCWF_N{N_sites}_F{N_fock}")
        if r_ed and r_mc:
            ax.plot(Phi_vals, r_ed['F_cl'], '-' + markers[idx_ed],
                     ms=8, lw=1.8,
                     label=fr'ED $N={N_sites}$')
            ax.plot(Phi_vals, r_mc['F_mcwf'], '--' + markers[idx_ed],
                     ms=6, lw=1.5, alpha=0.7,
                     label=fr'MCWF $N={N_sites}$')
    ax.set_xlabel(r'$\Phi$', fontsize=11)
    ax.set_ylabel(r'$F_{\rm cl}(\Phi)$', fontsize=11)
    ax.set_title(r'(c) MCWF-sampled FI vs ED (cross-check)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    # Wall-time scaling with N
    N_all = []
    wall_ED = []
    for N_sites, N_fock in ED_sizes:
        r = results[f"ED_N{N_sites}_F{N_fock}"]
        N_all.append(N_sites)
        wall_ED.append(r['wall'])
    wall_MC = []
    N_MC = []
    for N_sites, N_fock, Ntr in MCWF_sizes:
        r = results[f"MCWF_N{N_sites}_F{N_fock}"]
        N_MC.append(N_sites)
        wall_MC.append(r['wall'])

    ax.plot(N_all, wall_ED, 'o-', ms=10, lw=2,
            label=r'ED wall time', color='tab:blue')
    ax.plot(N_MC, wall_MC, 's-', ms=10, lw=2,
            label=r'MCWF wall time', color='tab:orange')
    ax.set_xlabel('Number of ring modes $N$', fontsize=11)
    ax.set_ylabel('Wall time (s)', fontsize=11)
    ax.set_title(f'(d) Scaling: ED blows up as $D^4 \\propto N_F^{{4N}}$\n'
                 f'MCWF scales as $D \\propto N_F^N$ × $N_{{\\rm traj}}$',
                 fontsize=10)
    ax.set_yscale('log')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')

    fig.suptitle(r'End-to-end Keldysh Fisher information for a Sagnac '
                 r'$N$-mode Kerr ring'
                 '\nED up to $N{=}4$ (blockade, $U{=}15\\kappa$); MCWF'
                 ' sampling for larger $N$. No linearization.',
                 fontsize=11, y=1.00)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sagnac_ring_multimode_FI.pdf",
                 dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_sagnac_ring_multimode_FI.png",
                 dpi=150, bbox_inches='tight')
    print(f"\nSaved {FIG_DIR}/fig_sagnac_ring_multimode_FI.pdf")

    with open(DATA_DIR / "sagnac_ring_multimode_FI.json", "w") as f:
        json.dump(dict(
            kappa=kappa, Delta=Delta, U=U, eps=eps, J=J,
            dPhi=dPhi, Phi_vals=Phi_vals.tolist(),
            results=results,
        ), f, indent=2)


if __name__ == "__main__":
    main()
