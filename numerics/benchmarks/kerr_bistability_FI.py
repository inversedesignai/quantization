"""
End-to-end Fisher information from the Keldysh/Lindblad generator, in a
regime where NO linearization, NO Gaussian approximation, and NO closed-form
likelihood is available.

Scenario: driven Kerr resonator at bistability.
------------------------------------------------
    H(Delta) = -Delta a^dag a + (U/2) a^dag a^dag a a + epsilon (a + a^dag)
    L_a = sqrt(kappa) a                              (photon loss)

Parameters chosen so that the classical mean-field equation

    -i Delta alpha + i U |alpha|^2 alpha - kappa alpha / 2 - i epsilon = 0

has THREE solutions across a range of Delta (upper branch, lower branch,
unstable middle).  This is the canonical driven-Kerr bistability;
Gaussian/Bogoliubov linearization around EITHER branch gives a wrong
answer for the steady state because the true quantum state is a
non-Gaussian MIXTURE of the two branches (Drummond-Walls 1980; Carmichael
1985).

Goal: estimate the detuning Delta (playing the role of a Sagnac shift),
using photon-number-resolving (PNR) detection of the cavity output.

Pipeline (no linearization):
  1.  From the Lindblad generator L(Delta), solve L[rho] = 0 for the
      steady state rho_ss(Delta) by exact super-operator diagonalization
      (qutip.steadystate).  This is literally the Keldysh saddle equation.
  2.  Read out P(n|Delta) = <n|rho_ss(Delta)|n> in the Fock basis.
      This is the Mandel distribution -- not closed-form, non-Gaussian.
  3.  Compute classical FI
          F_cl(Delta) = sum_n (1/P)(dP/dDelta)^2
      by finite-difference in Delta (central difference on rho_ss).
  4.  Compute quantum FI (the measurement-independent upper bound)
          F_Q(Delta) = 8 (1 - F(rho(Delta), rho(Delta+d)))/d^2
      using the Uhlmann fidelity.  This is the best any POVM could
      achieve.
  5.  For contrast, compute the "naive Gaussian" FI one would predict
      by linearizing around the upper-branch mean field.  It is
      systematically wrong inside the bistable window.

This demonstrates that the Keldysh generator, used as a black-box
numerical engine, gives F(theta) without requiring any specific
likelihood shape.
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

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_lindblad(N_fock, Delta, U, epsilon, kappa):
    """Driven Kerr cavity Lindblad (Keldysh saddle) generator."""
    a = qt.destroy(N_fock)
    n = a.dag() * a
    H = -Delta * n + (U / 2.0) * a.dag() * a.dag() * a * a \
        + epsilon * (a + a.dag())
    c_ops = [np.sqrt(kappa) * a]
    return H, c_ops, a


def keldysh_steady_state(N_fock, Delta, U, epsilon, kappa):
    """Solve L[rho] = 0 for rho_ss (exact super-operator saddle)."""
    H, c_ops, a = build_lindblad(N_fock, Delta, U, epsilon, kappa)
    rho_ss = qt.steadystate(H, c_ops)
    return rho_ss, a


def classical_mean_field(Delta, U, epsilon, kappa):
    """Roots of the classical Kerr cubic:
           [(U|alpha|^2 - Delta) + i kappa/2] alpha = epsilon.
    Parametrize by n = |alpha|^2 and solve
           n [ (U n - Delta)^2 + (kappa/2)^2 ] = epsilon^2.
    """
    # Cubic in n (with U, Delta, kappa, epsilon real)
    a3 = U**2
    a2 = -2 * U * Delta
    a1 = Delta**2 + (kappa / 2.0)**2
    a0 = -epsilon**2
    roots = np.roots([a3, a2, a1, a0])
    real_roots = sorted(np.real(r) for r in roots if abs(r.imag) < 1e-8 and r.real > 0)
    return real_roots  # photon number on each classical branch


def gaussian_linearized_FI(Delta, U, epsilon, kappa, dDelta=1e-3):
    """Gaussian/Bogoliubov Fisher info for PNR of a displaced Gaussian state
    linearized around the UPPER mean-field branch.  This is what Section 6
    of qnm_inertial.tex would predict if one naively applied it.

    Treats the cavity state as coherent |alpha_ss> with shot noise.
    F = |d<n>/dDelta|^2 / Var(n) = |d|alpha|^2/dDelta|^2 / |alpha|^2
      = 4 |alpha|^2 (d arg alpha / dDelta)^2  (for coherent)
    """
    def n_upper(Delta_):
        roots = classical_mean_field(Delta_, U, epsilon, kappa)
        return max(roots) if roots else np.nan
    n_p = n_upper(Delta + dDelta)
    n_m = n_upper(Delta - dDelta)
    n_0 = n_upper(Delta)
    if np.isnan(n_0) or n_0 < 1e-6:
        return 0.0
    dndD = (n_p - n_m) / (2 * dDelta)
    # For coherent state: F = (dn/dDelta)^2 / Var(n) with Var(n)=n.
    # This is the "Gaussian-linearized" PNR FI prediction.
    return dndD**2 / n_0


def Pn_from_rho(rho, N_fock):
    """Diagonal photon-number probability distribution P(n)."""
    diag = np.real(np.diag(rho.full()))
    return np.clip(diag, 0, 1)


def FI_classical_PNR(rho_p, rho_0, rho_m, dDelta, N_fock):
    """Classical FI for PNR outcomes from finite-difference of rho_ss."""
    P_p = Pn_from_rho(rho_p, N_fock)
    P_m = Pn_from_rho(rho_m, N_fock)
    P_0 = Pn_from_rho(rho_0, N_fock)
    dP = (P_p - P_m) / (2 * dDelta)
    mask = P_0 > 1e-12
    return float(np.sum((dP[mask]**2) / P_0[mask]))


def quantum_FI(rho_p, rho_m, dDelta):
    """Quantum Fisher information via the Uhlmann-fidelity formula:
        F_Q = 8 (1 - sqrt(F(rho(theta), rho(theta+d)))) / d^2
    where F(rho, sigma) = (Tr sqrt(sqrt(rho) sigma sqrt(rho)))^2.
    For a 2-step finite difference with step d = 2*dDelta:
        F_Q = 8 (1 - Fid) / (2 dDelta)^2.
    """
    # Symmetrize: use rho_m as reference, rho_p as shifted by 2*dDelta.
    sqrt_m = (rho_m).sqrtm()
    inner = sqrt_m * rho_p * sqrt_m
    Fid = (inner.sqrtm()).tr()
    Fid = float(np.real(Fid))**2  # squared fidelity
    F_Q = 8.0 * (1.0 - np.sqrt(max(Fid, 0.0))) / (2 * dDelta)**2
    return F_Q


def wigner_function(rho, xvec):
    return qt.wigner(rho, xvec, xvec)


def main():
    # Physical parameters (units of kappa)
    kappa = 1.0
    U = 0.15            # Kerr nonlinearity (weak but finite)
    epsilon = 1.3       # drive strength -- tuned to put us in bistable window
    N_fock = 60         # truncation

    # Scan detuning across the bistable window
    Delta_vals = np.linspace(0.2, 1.5, 27)
    dDelta = 5e-3

    rhos = {}
    FI_cl = []
    FI_Q = []
    FI_gauss = []
    n_branches_list = []
    n_mean = []
    n_var = []
    t0 = time.time()
    print("=== Keldysh FI without linearization (bistable Kerr) ===\n")
    print(f"{'Delta':>7} {'#br':>4} {'<n>_Q':>8} {'Var(n)':>8} "
          f"{'F_cl (PNR)':>12} {'F_Q':>12} {'F_gauss':>10}")

    for D in Delta_vals:
        rho_p, a = keldysh_steady_state(N_fock, D + dDelta, U, epsilon, kappa)
        rho_m, _ = keldysh_steady_state(N_fock, D - dDelta, U, epsilon, kappa)
        rho_0, _ = keldysh_steady_state(N_fock, D, U, epsilon, kappa)
        rhos[float(D)] = rho_0
        n_expect = float(qt.expect(a.dag()*a, rho_0).real)
        n2 = float(qt.expect((a.dag()*a)**2, rho_0).real)
        n_mean.append(n_expect)
        n_var.append(n2 - n_expect**2)
        F_cl = FI_classical_PNR(rho_p, rho_0, rho_m, dDelta, N_fock)
        FI_cl.append(F_cl)
        try:
            F_Q = quantum_FI(rho_p, rho_m, dDelta)
        except Exception as exc:
            F_Q = np.nan
        FI_Q.append(F_Q)
        F_g = gaussian_linearized_FI(D, U, epsilon, kappa)
        FI_gauss.append(F_g)
        branches = classical_mean_field(D, U, epsilon, kappa)
        n_branches_list.append(branches)
        print(f"{D:>7.3f} {len(branches):>4d} {n_expect:>8.3f} "
              f"{n2 - n_expect**2:>8.3f} {F_cl:>12.3e} {F_Q:>12.3e} "
              f"{F_g:>10.3e}")

    t1 = time.time()
    print(f"\nWall time: {t1-t0:.1f} s for {len(Delta_vals)} detuning points\n")

    FI_cl = np.array(FI_cl)
    FI_Q = np.array(FI_Q)
    FI_gauss = np.array(FI_gauss)
    n_mean = np.array(n_mean)
    n_var = np.array(n_var)

    # --- figure -------------------------------------------------------
    fig = plt.figure(figsize=(13, 9.5))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    # (a) classical mean-field branches vs Delta: bistability diagram
    ax = fig.add_subplot(gs[0, 0])
    for D, br in zip(Delta_vals, n_branches_list):
        for nb in br:
            color = 'r' if (len(br) == 3 and nb > 0.3 * br[2]) else 'b'
            ax.plot(D, nb, 'o', ms=3, color=color)
    ax.plot(Delta_vals, n_mean, 'k-', lw=2, label=r'quantum $\langle n\rangle$')
    ax.set_xlabel(r'$\Delta/\kappa$', fontsize=11)
    ax.set_ylabel(r'$|\alpha|^2$ (classical) / $\langle n\rangle$ (quantum)',
                  fontsize=11)
    ax.set_title('(a) Bistability: 3 classical branches\nquantum $\\rho_{ss}$'
                 ' smears them', fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    # (b) Fisher information: classical PNR, quantum, Gaussian-linearized
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(Delta_vals, FI_cl, 'b-', lw=2, label=r'$F_{\rm cl}$ (PNR, end-to-end)')
    ax.plot(Delta_vals, FI_Q, 'g--', lw=2, label=r'$F_Q$ (QFI, upper bound)')
    ax.plot(Delta_vals, FI_gauss, 'r:', lw=2,
            label=r'$F_{\rm Gauss}$ (linearized upper branch)')
    ax.set_xlabel(r'$\Delta/\kappa$', fontsize=11)
    ax.set_ylabel(r'$F(\Delta)$ [per unit $\kappa^{-2}$]', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('(b) Fisher information\nGaussian fails inside bistable window',
                 fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, which='both')

    # (c) variance /mean (Mandel Q factor)
    ax = fig.add_subplot(gs[0, 2])
    Q_mandel = (n_var - n_mean) / np.maximum(n_mean, 1e-6)
    ax.plot(Delta_vals, Q_mandel, 'k-', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.6,
               label='Poisson (coherent)')
    ax.set_xlabel(r'$\Delta/\kappa$', fontsize=11)
    ax.set_ylabel(r'Mandel $Q = {\rm Var}(n)/\langle n\rangle - 1$', fontsize=11)
    ax.set_title('(c) Super-Poissonian noise signatures\nbistable mixture',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (d-f) P(n|Delta) at three detunings: outside, inside, across the window
    highlight = [0.3, 0.75, 1.3]
    for i, D_sel in enumerate(highlight):
        j = int(np.argmin(np.abs(Delta_vals - D_sel)))
        D_actual = Delta_vals[j]
        rho_sel = rhos[float(D_actual)]
        P = Pn_from_rho(rho_sel, N_fock)
        ax = fig.add_subplot(gs[1, i])
        ax.bar(np.arange(N_fock), P, color='tab:purple', alpha=0.8)
        ax.set_xlim(0, min(N_fock, 30))
        ax.set_xlabel(r'photon number $n$', fontsize=11)
        ax.set_ylabel(r'$P(n|\Delta)$', fontsize=11)
        ax.set_title(f'(d{i+1}) $P(n)$ at $\\Delta/\\kappa={D_actual:.2f}$\n'
                     f'{"below" if i==0 else ("in bistable" if i==1 else "above")} window',
                     fontsize=10)
        ax.grid(alpha=0.3, axis='y')

    # (g-i) Wigner functions at the same three detunings
    xvec = np.linspace(-5, 6, 80)
    for i, D_sel in enumerate(highlight):
        j = int(np.argmin(np.abs(Delta_vals - D_sel)))
        D_actual = Delta_vals[j]
        rho_sel = rhos[float(D_actual)]
        W = wigner_function(rho_sel, xvec)
        ax = fig.add_subplot(gs[2, i])
        vmax = np.abs(W).max()
        im = ax.pcolormesh(xvec, xvec, W, cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                           shading='auto')
        ax.set_aspect('equal')
        ax.set_xlabel(r'$x$', fontsize=11)
        ax.set_ylabel(r'$p$', fontsize=11)
        ax.set_title(f'(e{i+1}) Wigner $W(x,p)$ at $\\Delta/\\kappa='
                     f'{D_actual:.2f}$', fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.046)

    fig.suptitle(r'End-to-end Keldysh Fisher information without linearization'
                 '\n(bistable driven Kerr cavity -- Gaussian approximation'
                 ' provably fails)',
                 fontsize=12, y=0.995)
    fig.savefig(FIG_DIR / "fig_kerr_bistability_FI.pdf", dpi=200, bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_kerr_bistability_FI.png", dpi=150, bbox_inches='tight')
    print(f"Saved {FIG_DIR}/fig_kerr_bistability_FI.pdf")

    # Save data
    with open(DATA_DIR / "kerr_bistability_FI.json", "w") as f:
        json.dump(dict(
            kappa=kappa, U=U, epsilon=epsilon, N_fock=N_fock,
            Delta=Delta_vals.tolist(),
            n_mean=n_mean.tolist(), n_var=n_var.tolist(),
            FI_cl=FI_cl.tolist(), FI_Q=FI_Q.tolist(),
            FI_gauss=FI_gauss.tolist(),
            branches={float(D): br for D, br in zip(Delta_vals, n_branches_list)},
        ), f, indent=2, default=lambda x: list(x))


if __name__ == "__main__":
    main()
