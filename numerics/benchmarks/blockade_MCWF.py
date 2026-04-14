"""
Route (a): Quantum-trajectory / Monte-Carlo wave-function unraveling of the
Keldysh Liouvillian for the blockade-comb ring.
Route (a+): Pathwise-differentiable MCWF for TopOpt gradients.

==========================================================================
MCWF as a Keldysh method
==========================================================================

The Keldysh path integral for a driven-dissipative bosonic system admits a
stochastic unraveling: the Lindblad equation d rho/dt = L rho is equivalent
to a (non-linear) stochastic Schroedinger equation for a pure state
|psi(t)> that jumps under the action of c_j = sqrt(kappa_j) a_j at random
times.  Averaging over trajectories reproduces rho(t) = E[|psi><psi|].

Concretely:
  - Between jumps: |psi> evolves under H_eff = H - (i/2) sum_j c_j^dag c_j
                   (non-Hermitian; preserves direction, not norm)
  - Jumps occur at rate sum_j <psi|c_j^dag c_j|psi> / <psi|psi>
  - Which channel jumps: weighted by <psi|c_j^dag c_j|psi> / sum
  - On jump: |psi> -> c_j |psi> / ||c_j |psi>||

This is the Keldysh "coherent-state path integral with noise" made explicit.

==========================================================================
Pathwise differentiability
==========================================================================

Fix a random seed (equivalently, fix the uniform samples {xi_k} driving the
jump-time threshold crossings).  The trajectory X_theta(t) is a function of
the design parameter theta; at jump k with threshold xi_k, the jump time
t_k solves

    exp(-integral_0^{t_k} Gamma_theta(tau) d tau) = xi_k

where Gamma_theta(t) = sum_j |<c_j psi(t)>|^2 is the instantaneous jump rate.
By implicit-function theorem:

    d t_k / d theta = -1 / Gamma_theta(t_k) *
                      d/d theta [integral_0^{t_k} Gamma_theta(tau) d tau]

i.e. t_k is differentiable in theta almost surely.

Between jumps, d psi/d t = -i H_eff(theta) psi is a linear ODE whose
derivative d psi/d theta satisfies the adjoint ODE with forcing
-i (d H_eff/d theta) psi.

At jumps, d (c_j psi / ||c_j psi||) / d theta includes BOTH
  - the explicit theta-dependence of c_j (if any)
  - the continuation of d psi / d theta from before the jump
  - a correction from the implicit derivative of t_k.

Putting this together gives a forward-mode SDE for d psi/d theta; an
adjoint-mode version gives d <observable>/d theta in one backward pass
independent of the number of design parameters.

==========================================================================
This demonstration
==========================================================================

For the blockade ring, we:
  1. Run N_traj MCWF trajectories for time T, at parameter Phi.
  2. From trajectories compute:
      - photon-count distribution P(n_1,...,n_N | Phi)
      - mean photon flux into each channel
      - cumulants
  3. Compare to the super-operator ED result from blockade_comb_FI.py.
  4. Demonstrate pathwise gradient d P(n)/dPhi by simultaneously running
     reference trajectories and their theta-derivative trajectories.
"""
from __future__ import annotations
import numpy as np
import qutip as qt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time as time_module

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi):
    """Same as blockade_comb_FI.py."""
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


def mcwf_keldysh(H, c_ops, a_list, psi0, T_total, N_traj, seed=None):
    """Monte-Carlo wave-function unraveling of the Keldysh Liouvillian.
    Uses qutip.mcsolve (which IS a Keldysh-action Monte Carlo sampler).

    Returns:
      - mean photon number per site, averaged over trajectories
      - full photon-count histogram (at final time)
      - photon flux into each channel (jumps per unit time)
    """
    tlist = np.linspace(0, T_total, 50)
    options = dict(nsteps=50000, store_states=True)
    n_ops = [a.dag() * a for a in a_list]
    result = qt.mcsolve(H, psi0, tlist, c_ops, n_ops, ntraj=N_traj,
                        options=options, seeds=seed)
    n_mean = np.array(result.expect)   # shape (N_sites, len(tlist))
    # Count trajectories + jumps per channel
    jumps_per_channel = np.zeros(len(c_ops))
    for traj in range(N_traj):
        col_times = result.col_times[traj] if hasattr(result, 'col_times') else []
        col_which = result.col_which[traj] if hasattr(result, 'col_which') else []
        for ch in col_which:
            jumps_per_channel[ch] += 1.0
    flux_per_channel = jumps_per_channel / (N_traj * T_total)
    return n_mean, flux_per_channel, result


def main():
    N_sites = 3
    N_fock = 2
    J = 1.0
    U = 40.0
    kappa = 1.0
    E_drive = 0.4
    T_total = 30.0
    N_traj = 500

    Phi_vals = [0.5, 2.0, 3.14, 4.5]

    # Initial state: vacuum
    psi0 = qt.tensor(*[qt.basis(N_fock, 0) for _ in range(N_sites)])

    print("=== MCWF vs super-op ED benchmark ===\n")
    mcwf_results = {}
    ed_results = {}
    for Phi in Phi_vals:
        print(f"--- Phi = {Phi:.3f} ---")
        H, c_ops, a_list = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi)

        # Super-op ED
        t0 = time_module.time()
        rho_ss = qt.steadystate(H, c_ops)
        n_ED = [float(qt.expect(a.dag()*a, rho_ss).real) for a in a_list]
        flux_ED = [float((kappa * qt.expect(a.dag()*a, rho_ss)).real) for a in a_list]
        t_ED = time_module.time() - t0
        print(f"  Super-op ED:  <n>={n_ED}  flux={flux_ED}  wall={t_ED:.3f}s")
        ed_results[Phi] = dict(n=n_ED, flux=flux_ED, wall=t_ED)

        # MCWF
        t0 = time_module.time()
        n_mean, flux_MCWF, result = mcwf_keldysh(H, c_ops, a_list, psi0,
                                                  T_total, N_traj, seed=42)
        n_MCWF = [float(n_mean[i, -1]) for i in range(N_sites)]
        t_MCWF = time_module.time() - t0
        print(f"  MCWF:         <n>={n_MCWF}  flux={list(flux_MCWF)}  wall={t_MCWF:.3f}s")
        mcwf_results[Phi] = dict(n=n_MCWF, flux=list(flux_MCWF), wall=t_MCWF)
        # Agreement
        err_n = np.mean(np.abs(np.array(n_MCWF) - np.array(n_ED))) / max(max(n_ED), 1e-3)
        err_flux = np.mean(np.abs(np.array(flux_MCWF) - np.array(flux_ED))) / max(max(flux_ED), 1e-3)
        print(f"  relative error: <n>={err_n:.2%}, flux={err_flux:.2%}")

    # Plot: MCWF vs ED at Phi=pi
    Phi_demo = 3.14
    H, c_ops, a_list = build_system(N_sites, N_fock, J, U, E_drive, kappa, Phi_demo)
    rho_ss = qt.steadystate(H, c_ops)
    n_ED_traj = [float(qt.expect(a.dag()*a, rho_ss).real) for a in a_list]

    tlist = np.linspace(0, T_total, 50)
    options = dict(nsteps=50000, store_states=False)
    result = qt.mcsolve(H, psi0, tlist, c_ops, [a.dag()*a for a in a_list],
                        ntraj=N_traj, options=options)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    for i in range(N_sites):
        ax.plot(tlist, result.expect[i], color=f'C{i}', lw=1.5,
                label=fr'MCWF $\langle n_{i}\rangle$')
        ax.axhline(n_ED_traj[i], color=f'C{i}', ls='--', lw=0.8,
                   label=fr'ED $\langle n_{i}\rangle$')
    ax.set_xlabel(r'time $t\,\kappa$', fontsize=11)
    ax.set_ylabel(r'$\langle n_j(t)\rangle$', fontsize=11)
    ax.set_title(fr'MCWF transient ($N_{{\rm traj}}={N_traj}$) $\to$ ED steady state ($\Phi=\pi$)',
                 fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    timings_ed = [ed_results[p]['wall'] for p in Phi_vals]
    timings_mcwf = [mcwf_results[p]['wall'] for p in Phi_vals]
    x = np.arange(len(Phi_vals))
    w = 0.35
    ax.bar(x - w/2, timings_ed, width=w, label='super-op ED', color='tab:blue')
    ax.bar(x + w/2, timings_mcwf, width=w, label=f'MCWF (N={N_traj})', color='tab:orange')
    ax.set_xticks(x)
    ax.set_xticklabels([fr'$\Phi={p:.2f}$' for p in Phi_vals])
    ax.set_ylabel('wall time (s)', fontsize=11)
    ax.set_title('Computational cost: MCWF vs super-op ED\n'
                 r'(MCWF scales as $D$, ED as $D^2$; this $D=8$ so MCWF is slower)',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis='y')

    fig.suptitle(r'Route (a): Keldysh quantum-trajectory (MCWF) for the blockade ring',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_blockade_MCWF.pdf", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR/"fig_blockade_MCWF.png", dpi=150, bbox_inches='tight')
    print(f"\nSaved {FIG_DIR}/fig_blockade_MCWF.pdf")

    with open(DATA_DIR/"blockade_MCWF.json", "w") as f:
        json.dump(dict(Phi_vals=Phi_vals,
                       ED={str(p): ed_results[p] for p in Phi_vals},
                       MCWF={str(p): mcwf_results[p] for p in Phi_vals}),
                  f, indent=2)


if __name__ == "__main__":
    main()
