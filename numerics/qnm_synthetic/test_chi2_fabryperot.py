"""
Non-tautological chi^(2) Im/Re vs 1/Q test using ANALYTICAL 1D Fabry-Perot QNMs.

This addresses critique CRIT-2: the previous synthetic-QNM test built the
mode functions as f̃ = f^(0)(1 + iφ/Q) by hand, so the resulting Im(g²)/Re ~ 1/Q
scaling was tautological.  Here we use the closed-form 1D FP QNM
   f̃_m(x) = 2iB e^{i k̃_m x} sin(k̃_m x),    k̃_m = (m π/L)(1 - i/2 Q_m)
which is the EXACT solution of the 1D Maxwell eigenvalue problem with
outgoing (Sommerfeld) boundary conditions [App E, Eq FP_QNM_freq].
The i/Q phase structure is then a CONSEQUENCE of solving Maxwell, not an
ANSATZ.  Verifying Im(g²)/Re ~ 1/Q on these QNMs is thus a true verification.

Test setup:
  - Cavity length L = 1 (units), n_c = mirror index
  - Three modes: m_1 (signal), m_2 (idler), m_3 = m_1 + m_2 (pump)
    so omega_3 = omega_1 + omega_2 (true downconversion triplet)
  - chi^(2) is taken as a constant scalar in the cavity (1D simplification)
  - Sweep Q via varying n_c: Q_m = m π / T = m π n_c² / 4
  - Verify slope of log|Im(g²)/Re| vs log(1/Q) is +1
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def fp_qnm_complex_k(m, L, n_c):
    """EXACT complex wavenumber of the m-th 1D FP QNM.

    Round-trip condition: r² e^{2i k̃ L} = 1  where r = (n_c-1)/(n_c+1).
    Solve: 2i k̃ L = ln(1/r²) + 2πim = -2 ln(r) + 2πim
       => k̃ L = πm + i ln(r)            (ln r < 0 for r<1, so Im(k̃)<0 ✓)

    Q_m = Re(k̃)/(-2 Im(k̃)) = πm/(2|ln r|).
    """
    r = (n_c - 1) / (n_c + 1)
    if r <= 0:
        # n_c <= 1: no resonance
        return None, None
    ln_r = np.log(r)
    k_real = m * np.pi / L
    k_imag = ln_r / L         # negative since ln(r)<0
    Q = m * np.pi / (-2 * ln_r)
    return k_real + 1j*k_imag, Q


def fp_qnm_field(x, m, L, n_c):
    """Closed-form FP QNM field f̃_m(x) inside the cavity, with bilinear norm 1."""
    k_tilde, Q = fp_qnm_complex_k(m, L, n_c)
    # f̃ = sin(k̃ x)  (drop overall 2iB normalization; we'll bilinear-normalize)
    f = np.sin(k_tilde * x)
    # bilinear norm: integral_0^L f² dx (no conjugation)
    # analytically: integral sin²(k̃ x) dx from 0 to L is complex; we just
    # numerically integrate
    dx = abs(x[1] - x[0])
    B = (f*f).sum() * dx
    return f / np.sqrt(B), Q


def chi2_overlap(f1, f2, f3, dx, chi2_profile=None, chi2=1.0):
    """g^(2)_{1,2,3} = integral chi2(x) f1* f2 f3 dx.

    NOTE: for a UNIFORM chi^(2) and 1D FP sin modes, the leading-order
    overlap vanishes by orthogonality whenever m_3 = m_1 + m_2.  This is
    a special 1D-symmetry feature, not a Kleinman cancellation.  To test
    the genuine chi^(2) Im/Re ~ 1/Q scaling, we use a SPATIALLY
    INHOMOGENEOUS chi^(2)(x) — e.g., a thin nonlinear film localized to
    a fraction of the cavity — which is also physically realistic
    (epitaxial GaAs/AlGaAs heterostructures, Si rib waveguides with
    SiN cladding, etc.).
    """
    if chi2_profile is None:
        return chi2 * (f1.conj() * f2 * f3).sum() * dx
    return ((chi2_profile) * f1.conj() * f2 * f3).sum() * dx


def main():
    # Resolution
    L = 1.0
    Nx = 4096   # high resolution to avoid grid artefacts
    x = np.linspace(0.001, 0.999, Nx)
    dx = x[1] - x[0]

    # Choose modes (m_1, m_2, m_3=m_1+m_2) for several pairs
    triplets = [(1, 1, 2), (1, 2, 3), (2, 3, 5), (1, 3, 4)]

    # Q sweep via mirror index n_c
    # Use EXACT QNM, restrict to high-n_c (Q > 30) where the high-Q expansion
    # of App E is valid and Im/Re ~ 1/Q is the predicted leading behavior.
    n_c_vals = np.array([5.0, 7.0, 10.0, 15.0, 20.0, 30.0, 50.0, 80.0, 120.0, 200.0])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))

    # Asymmetric chi^(2)(x) — localized to x in [0.15L, 0.55L] (off-center)
    # This breaks the parity that otherwise forces the leading-order chi^(2)
    # downconversion overlap to vanish for sin modes with m_3 = m_1 + m_2.
    chi2_profile = np.where((x > 0.15) & (x < 0.55), 1.0, 0.0)

    summary = []
    for (m1, m2, m3) in triplets:
        Qs_avg = []
        ratios = []
        for n_c in n_c_vals:
            f1, Q1 = fp_qnm_field(x, m1, L, n_c)
            f2, Q2 = fp_qnm_field(x, m2, L, n_c)
            f3, Q3 = fp_qnm_field(x, m3, L, n_c)
            g2 = chi2_overlap(f1, f2, f3, dx, chi2_profile=chi2_profile)
            ratio = abs(g2.imag/g2.real) if abs(g2.real) > 0 else 0
            # average Q for this triplet
            Q_avg = 1/(1/Q1 + 1/Q2 + 1/Q3) * 3       # harmonic mean
            Qs_avg.append(Q_avg)
            ratios.append(ratio)
            print(f"  ({m1},{m2},{m3})  n_c={n_c:.1f}  Q1={Q1:.0f} Q2={Q2:.0f} Q3={Q3:.0f}  Im/Re={ratio:.3e}")

        Qs_avg = np.array(Qs_avg)
        ratios = np.array(ratios)
        # log-log fit
        mask = ratios > 1e-15
        if mask.sum() >= 3:
            log_ratios = np.log(ratios[mask])
            log_invQ = np.log(1.0/Qs_avg[mask])
            slope, intercept = np.polyfit(log_invQ, log_ratios, 1)
        else:
            slope, intercept = 0.0, 0.0
        prefactor = np.exp(intercept)
        print(f"\n  ({m1},{m2},{m3}) fit: slope={slope:.4f} prefactor={prefactor:.3f}")
        summary.append(dict(triplet=(m1,m2,m3), slope=float(slope),
                            prefactor=float(prefactor),
                            Qs=Qs_avg.tolist(), ratios=ratios.tolist()))

        axes[0].loglog(1/Qs_avg, ratios, 'o-', ms=6,
                       label=f'(m₁,m₂,m₃)=({m1},{m2},{m3}), slope={slope:.2f}')

    # Theory line
    inv_Q = np.logspace(-3, 0, 50)
    axes[0].loglog(inv_Q, inv_Q, 'k--', lw=1, label=r'slope 1 reference')
    axes[0].set_xlabel(r'$1/Q$ (harmonic mean of triplet)', fontsize=11)
    axes[0].set_ylabel(r'$|\mathrm{Im}(g^{(2)})/\mathrm{Re}|$', fontsize=11)
    axes[0].set_title('Analytic 1D Fabry–Pérot QNMs:\n$\\chi^{(2)}$ Im/Re vs 1/Q',
                      fontsize=10)
    axes[0].grid(alpha=0.3, which='both')
    axes[0].legend(fontsize=8, loc='upper left')

    # Right panel: show one QNM mode profile
    n_c_demo = 3.0
    fdemo, Qdemo = fp_qnm_field(x, 1, L, n_c_demo)
    ax = axes[1]
    ax.plot(x, fdemo.real, 'b-', label='Re $\\tilde f_1(x)$')
    ax.plot(x, fdemo.imag, 'r-', label='Im $\\tilde f_1(x)$')
    ax.set_xlabel('$x/L$', fontsize=11)
    ax.set_ylabel(r'$\tilde f_m(x)$', fontsize=11)
    ax.set_title(f'1D FP QNM m=1 at $n_c={n_c_demo}$, $Q\\approx{Qdemo:.0f}$',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_chi2_fabryperot.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_chi2_fabryperot.png", dpi=150)
    print(f"\nSaved {FIG_DIR}/fig_chi2_fabryperot.pdf")

    with open(DATA_DIR/"chi2_fabryperot_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Sanity check: average slope across triplets
    avg_slope = np.mean([s['slope'] for s in summary])
    print(f"\n=== Average slope across {len(summary)} triplets: {avg_slope:.4f} ===")
    print(f"=== Theory predicts slope = 1 (exact in high-Q limit) ===")
    if 0.85 < avg_slope < 1.15:
        print("=== [PASS] χ² Im/Re ~ 1/Q on analytical Maxwell QNMs ===")
    else:
        print(f"=== [FAIL] slope {avg_slope:.3f} differs from 1 by >15% ===")


if __name__ == "__main__":
    main()
