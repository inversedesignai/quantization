"""
Route (a+): Pathwise-differentiable MCWF gradients via Common Random Numbers (CRN).

==========================================================================
Pathwise gradient via common-seeded trajectories
==========================================================================

The naive finite-difference gradient for MCWF:

    dX/dtheta ~ (X_theta+delta - X_theta)/delta

is high-variance because each X comes from independent random noise.  The
Common Random Numbers (CRN) trick: use the SAME random seeds for the two
MCWF runs at theta and theta+delta.  Then the random parts nearly cancel
trajectory-by-trajectory, leaving only the parameter-dependent part.

For a smooth parameter dependence and small delta, the CRN gradient
converges to the pathwise gradient d X/d theta at a rate much faster than
the independent-seed estimator.  This is a first step toward full
pathwise differentiation of the MCWF SDE (adjoint quantum optimal control).

Result: low-variance gradient of photon-count observables w.r.t. the
design parameter.  Demonstrated here for the Sagnac flux Phi; the same
approach extends to ANY design parameter entering H_eff or the collapse
operators.
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


def build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi):
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


def mcwf_expect_at_T(H, c_ops, a_list, psi0, T, N_traj, seeds):
    """Compute site-photon-number expectations at time T via MCWF with
    specified trajectory seeds."""
    tlist = [0.0, T]
    n_ops = [a.dag() * a for a in a_list]
    result = qt.mcsolve(H, psi0, tlist, c_ops, n_ops, ntraj=N_traj,
                        options=dict(nsteps=50000, store_states=False),
                        seeds=seeds)
    # result.expect[j] is array of length 2 (t=0 and t=T)
    return np.array([result.expect[j][-1] for j in range(len(a_list))])


def pathwise_gradient_CRN(N_sites, N_fock, J, U, E_drive, kappa, Phi, dPhi,
                           T, N_traj, common_seed):
    """Pathwise-ish gradient d<n>/dPhi using common random numbers.

    Returns the N-site photon-number gradient vector and the pathwise
    variance estimate.
    """
    psi0 = qt.tensor(*[qt.basis(N_fock, 0) for _ in range(N_sites)])
    H_p, c_p, a_p = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi + dPhi/2)
    H_m, c_m, a_m = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi - dPhi/2)

    # CRN seeds
    seeds_CRN = common_seed
    n_p_CRN = mcwf_expect_at_T(H_p, c_p, a_p, psi0, T, N_traj, seeds_CRN)
    n_m_CRN = mcwf_expect_at_T(H_m, c_m, a_m, psi0, T, N_traj, seeds_CRN)
    grad_CRN = (n_p_CRN - n_m_CRN) / dPhi

    # Independent seeds
    n_p_ind = mcwf_expect_at_T(H_p, c_p, a_p, psi0, T, N_traj,
                                common_seed + 10000)
    n_m_ind = mcwf_expect_at_T(H_m, c_m, a_m, psi0, T, N_traj,
                                common_seed + 20000)
    grad_ind = (n_p_ind - n_m_ind) / dPhi
    return grad_CRN, grad_ind


def main():
    N_sites = 3
    N_fock = 2
    J = 1.0
    U = 40.0
    kappa = 1.0
    E_drive = 0.4
    Phi = 2.5
    dPhi = 0.05
    T = 20.0

    # Compare CRN vs independent-seeds gradients for increasing N_traj
    N_traj_list = [50, 100, 200, 500, 1000]
    N_resamples = 8

    rows = []
    for N_traj in N_traj_list:
        CRN_grads = []
        IND_grads = []
        for r in range(N_resamples):
            common_seed = 1000 * (r + 1)
            gC, gI = pathwise_gradient_CRN(N_sites, N_fock, J, U, E_drive, kappa,
                                             Phi, dPhi, T, N_traj, common_seed)
            CRN_grads.append(gC[0])    # site 0
            IND_grads.append(gI[0])
        CRN_grads = np.array(CRN_grads)
        IND_grads = np.array(IND_grads)
        print(f"  N_traj={N_traj}  CRN: mean={CRN_grads.mean():.4e} std={CRN_grads.std():.4e}"
              f"  | IND: mean={IND_grads.mean():.4e} std={IND_grads.std():.4e}")
        rows.append(dict(N_traj=N_traj,
                         CRN_mean=float(CRN_grads.mean()),
                         CRN_std=float(CRN_grads.std()),
                         IND_mean=float(IND_grads.mean()),
                         IND_std=float(IND_grads.std())))

    # Reference: ED
    from qutip import steadystate
    H_p, c_p, a_p = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi + dPhi/2)
    H_m, c_m, a_m = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi - dPhi/2)
    rho_p = steadystate(H_p, c_p)
    rho_m = steadystate(H_m, c_m)
    n_p_ED = [float(qt.expect(a.dag()*a, rho_p).real) for a in a_p]
    n_m_ED = [float(qt.expect(a.dag()*a, rho_m).real) for a in a_m]
    grad_ED = (np.array(n_p_ED) - np.array(n_m_ED)) / dPhi
    print(f"\nReference ED gradient: {grad_ED}")

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    Ntraj = np.array([r['N_traj'] for r in rows])
    ax.errorbar(Ntraj, [r['CRN_mean'] for r in rows],
                yerr=[r['CRN_std'] for r in rows], fmt='o-', ms=8,
                label='CRN (pathwise-like)', capsize=4, color='tab:blue')
    ax.errorbar(Ntraj, [r['IND_mean'] for r in rows],
                yerr=[r['IND_std'] for r in rows], fmt='s--', ms=8,
                label='independent seeds', capsize=4, color='tab:red')
    ax.axhline(grad_ED[0], color='black', ls=':', lw=1.5,
               label=fr'exact ED = {grad_ED[0]:.3f}')
    ax.set_xscale('log')
    ax.set_xlabel(r'MCWF trajectories $N_{\rm traj}$', fontsize=11)
    ax.set_ylabel(r'$d\langle n_0\rangle/d\Phi$', fontsize=11)
    ax.set_title('Pathwise gradient via Common Random Numbers: lower variance than independent seeds',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_blockade_MCWF_pathwise.pdf", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR/"fig_blockade_MCWF_pathwise.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved {FIG_DIR}/fig_blockade_MCWF_pathwise.pdf")

    # Variance reduction ratio
    for r in rows:
        ratio = r['IND_std'] / max(r['CRN_std'], 1e-10)
        print(f"  N_traj={r['N_traj']}: variance reduction CRN vs IND = {ratio:.1f}x")

    with open(DATA_DIR/"blockade_MCWF_pathwise.json", "w") as f:
        json.dump(dict(rows=rows, grad_ED=grad_ED.tolist()), f, indent=2)


if __name__ == "__main__":
    main()
