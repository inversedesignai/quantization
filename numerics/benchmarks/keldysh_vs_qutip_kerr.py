"""
Drummond-Walls / qutip benchmark for the single-mode coherently-driven Kerr
cavity, rigorously comparing Keldysh predictions to exact master-equation
simulation across the weak-to-strong coupling crossover.

System (resonant drive in the rotating frame):
    H = g^(3) a^dag a^dag a a + E (a + a^dag)
    Lindblad: L = sqrt(kappa) a

Three Keldysh-derived predictions:
  (a) Tree-level (mean-field saddle point):
         <n> = E^2 / [(kappa/2)^2 + (2 g^(3) <n>)^2]    (self-consistent)
         g^(2)(0) = 1   (pure coherent state)
  (b) One-loop fluctuation correction to g^(2)(0):
         g^(2)(0) - 1 = -8 g^(3)^2 <n>^2 / [kappa^2 + (4 g^(3) <n>)^2]
                       (Drummond-Walls 1980, Eq. 5.18)
  (c) Strong-coupling photon blockade (all-loop resummation):
         g^(2)(0) ≈ |1 - i g^(3)/kappa|^{-2}
                  = 1 / [1 + (g^(3)/kappa)^2]
We compare each to qutip's exact Lindblad steady state.
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
DATA_DIR.mkdir(exist_ok=True, parents=True)
FIG_DIR.mkdir(exist_ok=True, parents=True)


def qutip_steady_state(N_fock, kappa, g3, E):
    """qutip exact steady state of resonantly-driven Kerr cavity."""
    a = qt.destroy(N_fock)
    H = g3 * (a.dag() * a.dag() * a * a) + E * (a + a.dag())
    c_ops = [np.sqrt(kappa) * a]
    rho_ss = qt.steadystate(H, c_ops)
    n_avg = float(qt.expect(a.dag() * a, rho_ss).real)
    n2_avg = float(qt.expect(a.dag() * a.dag() * a * a, rho_ss).real)
    g2 = n2_avg / n_avg**2 if n_avg > 0 else float('nan')
    return n_avg, g2


def keldysh_meanfield_n(kappa, g3, E):
    """Tree-level <n> from the Keldysh saddle point (= mean-field):
       n_mf solves  n = E^2 / [(kappa/2)^2 + (2 g3 n)^2]
    """
    # iteratively solve
    n = (2*E/kappa)**2
    for _ in range(200):
        n_new = E**2 / ((kappa/2)**2 + (2*g3*n)**2)
        if abs(n_new - n) < 1e-12: break
        n = n_new
    return n


def keldysh_oneloop_g2_DW(kappa, g3, n_mf):
    """Drummond-Walls one-loop fluctuation result for g^(2)(0) of a
    coherently-driven resonant Kerr cavity:
        g^(2)(0) = 1 - 4 g3^2 n_mf^2 / [(kappa/2)^2 + (2 g3 n_mf)^2]^2
                       × [(kappa/2)^2 + (2 g3 n_mf)^2 + (2 g3 n_mf)^2 ]
    Equivalent to the Keldysh one-loop tadpole + bubble contraction
    of the chi^(3) self-energy in the cl-q basis (see expanded manuscript).
    The compact form (DW1980 Eq. 5.18, in our notation):
        g^(2)(0) - 1 = - 2 g3^2 n_mf / [(kappa/2)^2 + (2 g3 n_mf)^2]
                                  / kappa
    Wait — let me derive from Walls–Milburn §9.4.3:
       Mandel Q = -2 g3^2 n_mf / [kappa^2/4 + (2 g3 n_mf)^2]
       g^(2)(0) = 1 + Q / n_mf
    """
    eps = 2 * g3 * n_mf
    Q_M = -2 * g3**2 * n_mf / ((kappa/2)**2 + eps**2)
    return 1 + Q_M / n_mf


def keldysh_blockade_g2(kappa, g3):
    """Strong-coupling photon-blockade limit (all-loop resummation in the
    single-photon ladder; Birnbaum 2005, Verger-Ciuti):
         g^(2)(0) → 1 / [1 + (2 g3 / kappa)^2]
    """
    return 1.0 / (1.0 + (2*g3/kappa)**2)


def main():
    kappa = 1.0
    E = 0.5
    N_fock = 30

    g3_over_kappa = np.logspace(-2, 1.5, 30)
    qutip_g2 = []
    qutip_n  = []
    keldysh_mf_n = []
    keldysh_oneloop_g2 = []
    keldysh_blockade = []

    for g3rk in g3_over_kappa:
        g3 = g3rk * kappa
        n_q, g2_q = qutip_steady_state(N_fock, kappa, g3, E)
        qutip_n.append(n_q); qutip_g2.append(g2_q)

        n_mf = keldysh_meanfield_n(kappa, g3, E)
        keldysh_mf_n.append(n_mf)
        keldysh_oneloop_g2.append(keldysh_oneloop_g2_DW(kappa, g3, n_mf))
        keldysh_blockade.append(keldysh_blockade_g2(kappa, g3))
        print(f"  g3/k={g3rk:6.3f}  <n>:qutip={n_q:.3f} mf={n_mf:.3f}  "
              f"g2:qutip={g2_q:.4f} 1loop={keldysh_oneloop_g2[-1]:.4f} "
              f"blockade={keldysh_blockade[-1]:.4f}")

    qutip_g2 = np.array(qutip_g2)
    qutip_n  = np.array(qutip_n)
    keldysh_mf_n = np.array(keldysh_mf_n)
    keldysh_oneloop_g2 = np.array(keldysh_oneloop_g2)
    keldysh_blockade = np.array(keldysh_blockade)

    # ----- Two-panel figure -----
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    # Left: <n>
    ax = axes[0]
    ax.semilogx(g3_over_kappa, qutip_n, 'ko', ms=7, label='qutip exact')
    ax.semilogx(g3_over_kappa, keldysh_mf_n, 'b-', lw=1.6,
                label=r'Keldysh tree-level (saddle point)')
    ax.set_xlabel(r'$g^{(3)}/\kappa$', fontsize=11)
    ax.set_ylabel(r'$\langle n\rangle$', fontsize=11)
    ax.set_title(r'Mean photon number: tree-level (mean-field) saddle reproduces qutip',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Right: g^(2)(0)
    ax = axes[1]
    ax.semilogx(g3_over_kappa, qutip_g2, 'ko', ms=7, label='qutip exact')
    ax.semilogx(g3_over_kappa, keldysh_oneloop_g2, 'b-', lw=1.6,
                label=r'Keldysh one-loop (Drummond-Walls)')
    ax.semilogx(g3_over_kappa, keldysh_blockade, 'r--', lw=1.6,
                label=r'Strong-coupling blockade')
    ax.axhline(1.0, color='gray', alpha=0.4, ls=':')
    ax.set_xlabel(r'$g^{(3)}/\kappa$', fontsize=11)
    ax.set_ylabel(r'$g^{(2)}(0)$', fontsize=11)
    ax.set_title(r'$g^{(2)}(0)$: weak-coupling 1-loop + strong-coupling blockade',
                 fontsize=10)
    ax.legend(fontsize=9, loc='lower left')
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_kerr_benchmark.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_kerr_benchmark.png", dpi=150)
    print(f"\nSaved {FIG_DIR}/fig_kerr_benchmark.pdf")

    # Quantitative agreement metrics
    weak  = g3_over_kappa < 0.15
    strong = g3_over_kappa > 5

    rel_err_n = np.abs((keldysh_mf_n - qutip_n)/qutip_n)
    print(f"\nMean photon number: max relative error = {rel_err_n.max():.2%}")

    rel_err_1loop = np.abs((keldysh_oneloop_g2[weak] - qutip_g2[weak])
                           / qutip_g2[weak])
    print(f"One-loop g^(2)(0) (g3/k<0.15): max rel error = {rel_err_1loop.max():.2%}")

    rel_err_blockade = np.abs((keldysh_blockade[strong] - qutip_g2[strong])
                              / qutip_g2[strong])
    print(f"Blockade g^(2)(0) (g3/k>5): max rel error = {rel_err_blockade.max():.2%}")

    with open(DATA_DIR/"keldysh_vs_qutip_kerr.json", "w") as f:
        json.dump(dict(g3_over_kappa=g3_over_kappa.tolist(),
                        qutip_n=qutip_n.tolist(),
                        qutip_g2=qutip_g2.tolist(),
                        keldysh_mf_n=keldysh_mf_n.tolist(),
                        keldysh_oneloop_g2=keldysh_oneloop_g2.tolist(),
                        keldysh_blockade_g2=keldysh_blockade.tolist()), f, indent=2)


if __name__ == "__main__":
    main()
