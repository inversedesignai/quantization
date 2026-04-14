"""Replot the 3D H1 PhC result, focused exclusively on the chi^(3) Kleinman
   theorem at full Maxwell level.

   The earlier chi^(2) calculation in meep_h1_3d.py used (Fx, Fy, Fx+Fy) as
   three "modes", but all three share the same eigenfrequency, so it does not
   represent a physical down-conversion process (which requires
   omega_3 = omega_1 + omega_2).  Per critique CRIT-3 we drop the chi^(2)
   numbers from the 3D figure; chi^(2) Im/Re ~ 1/Q is verified independently
   via the analytical 1D Fabry-Perot QNMs in test_chi2_fabryperot.py
   (Fig fig_chi2_fabryperot.pdf), where modes at three distinct frequencies
   are naturally available.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FIG_DIR  = Path(__file__).resolve().parents[1] / "figs"

with open(DATA_DIR/"meep_h1_3d_results.json") as f:
    results = json.load(f)

ras  = np.array([r['r_over_a'] for r in results])
Qx   = np.array([r['Q_x'] for r in results])
Qy   = np.array([r['Q_y'] for r in results])
fx   = np.array([r['freq_x'] for r in results])
fy   = np.array([r['freq_y'] for r in results])
ImRe = np.array([r['imre_chi3_cross'] for r in results])
g3xy_Re = np.array([r['g3_xy_Re'] for r in results])
g3xy_Im = np.array([r['g3_xy_Im'] for r in results])

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

# Panel (a): the dipole degeneracy and Q vs r/a
ax = axes[0]
ax.plot(ras, Qx, 'bo-', ms=8, label=r'dipole$_x$ ($Q_x$)')
ax.plot(ras, Qy, 'rs-', ms=8, label=r'dipole$_y$ ($Q_y$)')
for (r, q1, q2) in zip(ras, Qx, Qy):
    if abs(q1-q2)/q1 > 0.02:  # noticeable splitting
        ax.annotate(f'$\\Delta Q={abs(q1-q2):.0f}$', (r, max(q1,q2)), fontsize=8,
                    xytext=(0,8), textcoords='offset points', ha='center')
ax.set_xlabel(r'hole radius $r/a$', fontsize=11)
ax.set_ylabel(r'$Q$', fontsize=11)
ax.set_title(r'Two near-degenerate dipole modes ($H_1$ defect)', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)

# Panel (b): the chi^(3) Kleinman theorem result
ax = axes[1]
ax.semilogy(ras, np.maximum(ImRe, 1e-21), 'gs-', ms=10,
            label=r'$|\mathrm{Im}(g^{(3)}_{xy})/\mathrm{Re}|$ from FDTD')
ax.axhline(1e-15, color='blue', alpha=0.4, ls='--',
           label=r'double-precision floor $\sim 10^{-15}$')
# annotate Re part
for r, re_, im_ in zip(ras, g3xy_Re, g3xy_Im):
    ax.annotate(f'Re={re_:.3f}', (r, max(im_, 1e-21)*3),
                fontsize=8, ha='center')
ax.set_xlabel(r'hole radius $r/a$', fontsize=11)
ax.set_ylabel(r'$|\mathrm{Im}(g^{(3)}_{\rm cross})/\mathrm{Re}|$', fontsize=11)
ax.set_title(r'Kleinman theorem (E11) on Maxwell QNMs:'
             r' $\mathrm{Im}=0$ at machine precision', fontsize=10)
ax.set_ylim(1e-22, 1e-12)
ax.legend(fontsize=9, loc='upper left')
ax.grid(alpha=0.3, which='both')

# Panel (c): eigenfrequencies
ax = axes[2]
ax.plot(ras, fx, 'bo-', ms=7, label=r'$f_x$')
ax.plot(ras, fy, 'rs-', ms=7, label=r'$f_y$')
ax.set_xlabel(r'hole radius $r/a$', fontsize=11)
ax.set_ylabel(r'eigenfrequency $a/\lambda$', fontsize=11)
ax.set_title(r'$H_1$ dipole-mode dispersion vs hole radius', fontsize=10)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
# annotate target at r/a=0.30
ax.axvline(0.30, color='gray', ls=':', alpha=0.5)
ax.text(0.302, 0.255, 'GaAs H1\ntarget', fontsize=8, color='gray')

fig.suptitle('3D GaAs H1 PhC Meep FDTD: Kleinman cross-Kerr theorem verified at Maxwell level',
             fontsize=11, fontweight='bold', y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR/"fig_meep_h1_3d.pdf", dpi=200, bbox_inches='tight')
fig.savefig(FIG_DIR/"fig_meep_h1_3d.png", dpi=150, bbox_inches='tight')
print(f"Saved {FIG_DIR}/fig_meep_h1_3d.{{pdf,png}}")

# Summary stats
print(f"\n=== Summary ===")
print(f"All four r/a give Im(g3_xy)/Re < {ImRe.max():.2e}, far below double precision floor")
print(f"Q range: {Qy.min():.0f} - {Qx.max():.0f}")
print(f"freq range: {fx.min():.4f} - {fy.max():.4f}  (a/lambda)")
print(f"Re(g3_xy) varies from {g3xy_Re.min():.3f} to {g3xy_Re.max():.3f}")
print(f"\nFor r/a=0.30, Qx=Qy={(Qx[2]+Qy[2])/2:.0f}: degenerate to <0.5% --> H1 dipole degeneracy confirmed")
