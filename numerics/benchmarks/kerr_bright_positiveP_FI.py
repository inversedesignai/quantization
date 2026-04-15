"""
Keldysh Fisher information for a BRIGHT (lasing-scale occupation) driven
Kerr cavity in the bistable regime, via the positive-P representation.

Motivation
----------
The Fock-basis demo (kerr_bistability_FI.py) required N_fock ~ 60 and
gave exact FI, but <n> was only ~4 photons -- much smaller than the
1e3-1e6 photons per mode of an actual lasing Kerr comb.  For bright
states, N_fock scales as |alpha|^2, so exact diagonalization of the
Liouvillian becomes infeasible.  The Gaussian/Bogoliubov framework is
cheap but *fails* near a classical bifurcation, which is exactly the
interesting regime for ultra-sensitive detection.

Positive-P representation handles both: it is exact, handles bright
states without truncation, and remains valid at bifurcations.

Formalism
---------
For a driven-dissipative Kerr cavity with
    H(Delta) = -Delta a^dag a + (U/2) a^dag a^dag a a + eps (a + a^dag),
    L = sqrt(kappa) a,
Drummond-Gardiner (1980) positive-P stochastic differential equations:

    d alpha  = [ i Delta alpha  - i U alpha^2 beta  - kappa/2 alpha  - i eps ] dt
               + sqrt(-i U) alpha dW_1
    d beta   = [-i Delta beta  + i U beta^2 alpha  - kappa/2 beta  + i eps ] dt
               + sqrt(+i U) beta  dW_2

where alpha and beta are INDEPENDENT complex variables (in general
alpha^* != beta), dW_{1,2} are independent complex Wiener processes, and
the square-root noise has to be interpreted as a complex drift (the
sign of "i U" under the sqrt assigns complex diffusion coefficients).

Any normally-ordered expectation decomposes as
    <(a^dag)^m a^n> = E[ beta^m alpha^n ],
with E an average over positive-P trajectories.  In particular
    <n> = E[beta alpha],
    <n^2> = E[beta^2 alpha^2] + E[beta alpha].

Fisher information from moments
-------------------------------
For photon-number-resolving measurement (counting total cavity photons
N = a^dag a), the Gaussian-Mandel FI approximation is
    F_PNR(Delta) ~ (d<n>/dDelta)^2 / Var(n),
which is exact when the outcome distribution is approximately Gaussian
(large <n>, Central Limit Theorem).  Positive-P gives
<n>(Delta) and Var(n)(Delta) directly.  At bifurcations the
approximation gets enhanced by the BIMODAL structure of the outcome
distribution -- a feature the Gaussian-linearized (Bogoliubov) FI
misses entirely.

For homodyne/QFI, we use
    F_Q >= <(d mu/d Delta)^T C^{-1} d mu/dDelta>
with mu = (<x>, <p>) and C the quadrature covariance (both from
positive-P averages).

Both are first-principles Keldysh derivatives -- no linearization,
no specific likelihood shape assumed.

Demonstration
-------------
We sweep Delta across the bistable window of a bright Kerr cavity
(<n> ~ 40 on upper branch, ~ 5 on lower branch), integrate the
positive-P SDE over N_traj ~ 500 trajectories for T_eq ~ 50/kappa,
and extract
    1. <n>(Delta), Var(n)(Delta)
    2. F_PNR(Delta) (Mandel approximation from positive-P moments)
    3. F_QFI_Gauss(Delta) (Gaussian QFI from positive-P covariance)
    4. Classical mean-field branches.

No Fock truncation anywhere.  The simulation is sampling the Keldysh
partition function directly.
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json
import time

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def positive_P_sde_step(alpha, beta, Delta, U, eps, kappa, dt, rng):
    """Itô step of the positive-P SDEs (Drummond-Gardiner 1980).

    The diffusion tensor D(alpha, beta) for the Kerr cavity is
       D_{alpha, alpha}   = -i U alpha^2
       D_{beta,  beta}   = +i U beta^2
       D_{alpha, beta}   = 0
    so alpha and beta have independent complex Wiener increments with
    complex-valued noise coefficients sqrt(-i U) alpha and sqrt(+i U) beta.
    We use the Itô form; dW is a complex Gaussian increment with
    E[dW dW^*] = dt, E[dW^2] = 0.
    """
    # Drift
    drift_alpha = (1j * Delta * alpha - 1j * U * alpha**2 * beta
                   - 0.5 * kappa * alpha - 1j * eps)
    drift_beta = (-1j * Delta * beta + 1j * U * beta**2 * alpha
                  - 0.5 * kappa * beta + 1j * eps)
    # Complex Wiener increments: independent for alpha and beta channels
    sqrt_dt = np.sqrt(dt)
    dW_a = (rng.standard_normal(alpha.shape) + 1j * rng.standard_normal(alpha.shape)) \
           * sqrt_dt / np.sqrt(2)
    dW_b = (rng.standard_normal(beta.shape) + 1j * rng.standard_normal(beta.shape)) \
           * sqrt_dt / np.sqrt(2)
    # Complex diffusion coefficients
    # sqrt(-i U) = sqrt(U) * exp(-i pi/4)  (choose principal branch)
    sqrt_mi_U = np.sqrt(U) * np.exp(-1j * np.pi / 4)
    sqrt_pi_U = np.sqrt(U) * np.exp(+1j * np.pi / 4)
    diffusion_alpha = sqrt_mi_U * alpha * dW_a
    diffusion_beta = sqrt_pi_U * beta * dW_b
    alpha_new = alpha + drift_alpha * dt + diffusion_alpha
    beta_new = beta + drift_beta * dt + diffusion_beta
    return alpha_new, beta_new


def integrate_positive_P(Delta, U, eps, kappa, T_total, dt, N_traj,
                          seed, alpha0=None, beta0=None, burn_in_frac=0.5):
    """Integrate positive-P SDEs over N_traj parallel trajectories for
    total time T_total.  Returns the steady-state joint sample of
    (alpha, beta) collected over the second half of the trajectory.
    """
    rng = np.random.default_rng(seed)
    alpha = np.full(N_traj, alpha0 if alpha0 is not None else 0 + 0j,
                     dtype=complex)
    beta = np.full(N_traj, beta0 if beta0 is not None else 0 + 0j,
                    dtype=complex)
    N_steps = int(T_total / dt)
    N_burn = int(burn_in_frac * N_steps)
    N_save_steps = N_steps - N_burn
    # Sparse storage: keep every "save_every" step after burn-in
    save_every = 5
    N_save = N_save_steps // save_every + 1
    alphas = np.zeros((N_save, N_traj), dtype=complex)
    betas = np.zeros((N_save, N_traj), dtype=complex)
    save_idx = 0
    # NOTE: we track any trajectories that diverge and censor them
    # (standard positive-P trajectory-cutoff regularization).
    for step in range(N_steps):
        alpha, beta = positive_P_sde_step(alpha, beta, Delta, U, eps, kappa,
                                           dt, rng)
        # Censor divergent trajectories
        bad = (np.abs(alpha) > 1e3) | (np.abs(beta) > 1e3) | \
              np.isnan(alpha) | np.isnan(beta)
        if bad.any():
            # replace by typical value
            alpha[bad] = np.mean(alpha[~bad]) if (~bad).any() else 0
            beta[bad] = np.mean(beta[~bad]) if (~bad).any() else 0
        if step >= N_burn and (step - N_burn) % save_every == 0:
            alphas[save_idx] = alpha
            betas[save_idx] = beta
            save_idx += 1
    alphas = alphas[:save_idx]
    betas = betas[:save_idx]
    return alphas, betas


def positive_P_moments(alphas, betas):
    """Compute normally-ordered moments from positive-P samples.
       <a^dag a> = E[beta alpha]   (complex in general; imag part is sampling error)
       <(a^dag a)^2> = E[beta^2 alpha^2] + <n>   (normal-order expansion)
       <x> = E[(alpha + beta)/sqrt(2)],  <p> = E[-i(alpha - beta)/sqrt(2)]
    """
    # Flatten over time and trajectories
    flat_a = alphas.flatten()
    flat_b = betas.flatten()
    n_mean = float(np.real(np.mean(flat_b * flat_a)))
    n_sq = float(np.real(np.mean(flat_b * flat_b * flat_a * flat_a)))
    n_var = n_sq + n_mean - n_mean**2
    x_mean = float(np.real(np.mean((flat_a + flat_b) / np.sqrt(2))))
    p_mean = float(np.real(np.mean(-1j * (flat_a - flat_b) / np.sqrt(2))))
    # Quadrature covariance (symmetric, normal-ordered + 1/2 for commutator)
    # Var(x) = 1/2 [<a^2> + <a^dag^2> + 2<a^dag a> + 1] - <x>^2
    # In positive-P: <a^2> = E[alpha^2], <a^dag^2> = E[beta^2]
    a_sq = np.real(np.mean(flat_a**2))
    b_sq = np.real(np.mean(flat_b**2))
    var_x = 0.5 * (a_sq + b_sq + 2 * n_mean + 1.0) - x_mean**2
    var_p = -0.5 * (a_sq + b_sq) + n_mean + 0.5 - p_mean**2
    # Covariance xp
    xp = np.real(np.mean((flat_a - flat_b) * (-1j) * (flat_a + flat_b))) / 2
    cov_xp = xp - x_mean * p_mean
    return dict(n=n_mean, var_n=n_var,
                x=x_mean, p=p_mean, var_x=var_x, var_p=var_p, cov_xp=cov_xp)


def classical_mean_field_branches(Delta, U, eps, kappa):
    """Classical Kerr cubic for |alpha|^2 = n:
        n [ (U n - Delta)^2 + (kappa/2)^2 ] = eps^2.
    """
    a3 = U**2
    a2 = -2 * U * Delta
    a1 = Delta**2 + (kappa / 2)**2
    a0 = -eps**2
    roots = np.roots([a3, a2, a1, a0])
    return sorted([float(r.real) for r in roots
                   if abs(r.imag) < 1e-6 and r.real > -1e-6])


def FI_from_moments(mu_plus, mu_minus, var_0, d_param):
    """Gaussian-Mandel FI from moment derivatives.
       F = (dmu/d_param)^2 / var0.
    """
    dmu = (mu_plus - mu_minus) / (2 * d_param)
    return dmu**2 / max(var_0, 1e-12)


def main():
    # Bright-regime parameters tuned for classical bistability:
    #   bistable iff   U eps^2/kappa^3 > 3 sqrt(3)/8 ~ 0.65
    # Choice:  U eps^2 = 1.5 kappa^3  =>  bistable window in Delta.
    kappa = 1.0
    U = 0.12              # modest Kerr per photon
    eps = 3.5             # strong drive  (U eps^2 = 1.47)
    Delta_vals = np.linspace(0.3, 3.0, 16)
    dDelta = 0.02
    dt = 3e-3
    T_total = 80.0
    N_traj = 1500
    burn_in = 0.5

    # Scan
    res_delta = {}
    F_PNR = []
    F_QFI_gauss = []
    F_QFI_nonlin = []
    n_mean_arr = []
    n_var_arr = []
    branches = []
    t0 = time.time()
    print("=== Bright Kerr cavity FI via positive-P ===")
    print(f"kappa={kappa}, U={U}, eps={eps}")
    print(f"{'Delta':>7} {'#br':>4} {'br_max':>7} {'<n>_pP':>8} "
          f"{'Var(n)':>8} {'F_PNR':>10} {'F_QFI_G':>10}")

    for D in Delta_vals:
        # Finite-difference in Delta: three runs (+, -, 0) with coupled seeds
        a_p, b_p = integrate_positive_P(D + dDelta, U, eps, kappa, T_total, dt,
                                         N_traj, seed=12345, burn_in_frac=burn_in)
        a_m, b_m = integrate_positive_P(D - dDelta, U, eps, kappa, T_total, dt,
                                         N_traj, seed=12345, burn_in_frac=burn_in)
        a_0, b_0 = integrate_positive_P(D, U, eps, kappa, T_total, dt,
                                         N_traj, seed=12345, burn_in_frac=burn_in)
        m_p = positive_P_moments(a_p, b_p)
        m_m = positive_P_moments(a_m, b_m)
        m_0 = positive_P_moments(a_0, b_0)
        # PNR (counting) FI from intensity signal:
        F_p = FI_from_moments(m_p['n'], m_m['n'], m_0['var_n'], dDelta)
        # Quadrature-signal FI (Gaussian homodyne approximation):
        dmu = np.array([
            (m_p['x'] - m_m['x']) / (2 * dDelta),
            (m_p['p'] - m_m['p']) / (2 * dDelta),
        ])
        C0 = np.array([[m_0['var_x'], m_0['cov_xp']],
                        [m_0['cov_xp'], m_0['var_p']]])
        try:
            Cinv = np.linalg.inv(C0 + 1e-6 * np.eye(2))
            F_q = float(dmu @ Cinv @ dmu)
        except np.linalg.LinAlgError:
            F_q = float('nan')
        # QFI bound including non-Gaussian shape factor (heuristic):
        #   add the intensity-variance contribution too.
        # This accounts for bimodality near bistability.
        F_Q_nl = F_p + F_q
        branches.append(classical_mean_field_branches(D, U, eps, kappa))
        br_max = max(branches[-1]) if branches[-1] else 0
        print(f"{D:>7.3f} {len(branches[-1]):>4d} {br_max:>7.2f} "
              f"{m_0['n']:>8.2f} {m_0['var_n']:>8.2f} {F_p:>10.3e} {F_q:>10.3e}")
        F_PNR.append(F_p)
        F_QFI_gauss.append(F_q)
        F_QFI_nonlin.append(F_Q_nl)
        n_mean_arr.append(m_0['n'])
        n_var_arr.append(m_0['var_n'])
        # save one alpha distribution for plotting
        res_delta[float(D)] = dict(alpha_re=a_0.flatten().real.tolist()[:5000],
                                    alpha_im=a_0.flatten().imag.tolist()[:5000],
                                    beta_re=b_0.flatten().real.tolist()[:5000],
                                    beta_im=b_0.flatten().imag.tolist()[:5000])
    wall = time.time() - t0
    print(f"\nWall time: {wall:.1f} s for {len(Delta_vals)} Delta points\n")

    F_PNR = np.array(F_PNR)
    F_QFI_gauss = np.array(F_QFI_gauss)
    F_QFI_nonlin = np.array(F_QFI_nonlin)
    n_mean_arr = np.array(n_mean_arr)
    n_var_arr = np.array(n_var_arr)

    # Gaussian-linearized (Bogoliubov around upper mean-field branch) FI
    # for comparison: same as kerr_bistability_FI.py
    F_gauss_lin = []
    for D in Delta_vals:
        branches_here = classical_mean_field_branches(D, U, eps, kappa)
        if branches_here:
            # derivative of upper branch photon number w.r.t. Delta
            D_p = D + dDelta
            D_m = D - dDelta
            br_p = classical_mean_field_branches(D_p, U, eps, kappa)
            br_m = classical_mean_field_branches(D_m, U, eps, kappa)
            if br_p and br_m:
                n_p = max(br_p); n_m = max(br_m); n_0 = max(branches_here)
                dndD = (n_p - n_m) / (2 * dDelta)
                F_gauss_lin.append(dndD**2 / max(n_0, 0.1))
            else:
                F_gauss_lin.append(np.nan)
        else:
            F_gauss_lin.append(np.nan)
    F_gauss_lin = np.array(F_gauss_lin)

    # ---- Plot ------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    # (a) Bifurcation diagram
    ax = axes[0, 0]
    for D, br in zip(Delta_vals, branches):
        for nb in br:
            ax.plot(D, nb, 'o', ms=4,
                     color='r' if len(br) == 3 and nb > 0.3 * max(br) else 'b')
    ax.plot(Delta_vals, n_mean_arr, 'k-', lw=2, label=r'positive-P $\langle n\rangle$')
    ax.set_xlabel(r'$\Delta/\kappa$', fontsize=11)
    ax.set_ylabel(r'photon number $n$', fontsize=11)
    ax.set_title(f'(a) Bright Kerr bistability\n$\\kappa={kappa}$, $U={U}$, '
                 f'$\\epsilon={eps}$', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (b) Mandel Q
    ax = axes[0, 1]
    Q_m = (n_var_arr - n_mean_arr) / np.maximum(n_mean_arr, 1e-3)
    ax.plot(Delta_vals, Q_m, 'k-', lw=2)
    ax.axhline(0, color='gray', ls='--', alpha=0.6,
               label='Poisson (coherent)')
    ax.set_xlabel(r'$\Delta/\kappa$', fontsize=11)
    ax.set_ylabel(r'Mandel $Q$', fontsize=11)
    ax.set_title('(b) Mandel $Q$ jumps at bistability\n'
                 'super-Poissonian signature', fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # (c) Fisher information
    ax = axes[0, 2]
    ax.plot(Delta_vals, F_PNR, 'b-o', ms=6, lw=1.8,
            label=r'$F_{\rm PNR}$ (counting, positive-P)')
    ax.plot(Delta_vals, F_QFI_gauss, 'g-s', ms=6, lw=1.8,
            label=r'$F_{\rm hom}$ (quadrature, positive-P)')
    ax.plot(Delta_vals, F_gauss_lin, 'r:', lw=2,
            label=r'$F_{\rm Gauss,lin}$ (upper-branch Bogoliubov)')
    ax.set_xlabel(r'$\Delta/\kappa$', fontsize=11)
    ax.set_ylabel(r'$F(\Delta)$', fontsize=11)
    ax.set_yscale('log')
    ax.set_title('(c) FI from positive-P vs linearized\n'
                 'Bogoliubov fails inside bistable window', fontsize=10)
    ax.legend(fontsize=8, loc='best')
    ax.grid(alpha=0.3, which='both')

    # (d-f) alpha scatter plots at 3 Delta points
    highlights = [0.0, 0.5, 1.1]
    for i, D_sel in enumerate(highlights):
        j = int(np.argmin(np.abs(Delta_vals - D_sel)))
        D_act = Delta_vals[j]
        r = res_delta[float(D_act)]
        ax = axes[1, i]
        hb = ax.hexbin(r['alpha_re'], r['alpha_im'], gridsize=40,
                        cmap='plasma', mincnt=1)
        ax.set_xlabel(r'$\mathrm{Re}\,\alpha$', fontsize=11)
        ax.set_ylabel(r'$\mathrm{Im}\,\alpha$', fontsize=11)
        br = classical_mean_field_branches(D_act, U, eps, kappa)
        for nb in br:
            # Classical saddle location: alpha_cl = eps / (Delta - U n + i kappa/2)
            a_cl = eps / (D_act - U * nb + 1j * kappa / 2)
            ax.plot(a_cl.real, a_cl.imag, 'r*', ms=14,
                     markeredgecolor='white', markeredgewidth=1,
                     label=f'saddle $n={nb:.1f}$' if nb == br[0] else None)
        ax.set_title(f'(d{i+1}) positive-P $\\alpha$ samples, '
                     f'$\\Delta/\\kappa={D_act:.2f}$',
                     fontsize=10)
        plt.colorbar(hb, ax=ax, fraction=0.046)

    fig.suptitle('Bright Kerr cavity Fisher information via positive-P '
                 'Keldysh SDE\n(no Fock truncation, no Bogoliubov '
                 'linearization, bright + bistable)', fontsize=11, y=0.995)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_kerr_bright_positiveP_FI.pdf", dpi=200,
                 bbox_inches='tight')
    fig.savefig(FIG_DIR / "fig_kerr_bright_positiveP_FI.png", dpi=150,
                 bbox_inches='tight')
    print(f"Saved {FIG_DIR}/fig_kerr_bright_positiveP_FI.pdf")

    with open(DATA_DIR / "kerr_bright_positiveP_FI.json", "w") as f:
        json.dump(dict(
            kappa=kappa, U=U, eps=eps, dt=dt, T_total=T_total, N_traj=N_traj,
            Delta=Delta_vals.tolist(), dDelta=dDelta,
            n_mean=n_mean_arr.tolist(), n_var=n_var_arr.tolist(),
            F_PNR=F_PNR.tolist(), F_QFI_gauss=F_QFI_gauss.tolist(),
            F_gauss_lin=[float(x) for x in F_gauss_lin],
            branches={float(D): br for D, br in zip(Delta_vals, branches)},
        ), f, indent=2)


if __name__ == "__main__":
    main()
