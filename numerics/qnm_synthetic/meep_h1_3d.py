"""
3D GaAs H1 photonic-crystal cavity: Meep FDTD extraction of dipole QNMs.

Geometry (CLAUDE.md target):
  triangular lattice of air holes, lattice constant a = 240 nm
  hole radius r/a = 0.30,  slab thickness t = 200 nm = 0.833 a
  GaAs n = 3.5,  central hole removed (H1 defect)
  two degenerate dipole modes expected near a/λ = 0.24 (λ ~ 1000 nm)

Working units: lengths in units of a;  freq in units of c/a.

The degenerate pair (dipole_x, dipole_y) at the same frequency is exactly
the setting for the chi^(3) cross-Kerr theorem E11.

Resolution budget: this runs at res=16 for debug (~2 min), res=20 for
production (~15 min), res=24 for publication (~1 hour).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys, time, json
sys.path.insert(0, "/home/zlin/miniforge/envs/meep/lib/python3.11/site-packages")
import meep as mp

FIG_DIR  = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)


def _pr(*args, **kwargs):
    if mp.am_master():
        import builtins
        builtins.print(*args, **kwargs, flush=True)


def build_h1_geometry(r_over_a=0.30, n_cells=5, t_slab=0.833, n_GaAs=3.5):
    """Build the H1 defect photonic crystal slab.

    A triangular (hexagonal) lattice of air holes of radius r*a, with the
    central hole removed.  Supercell: n_cells rings of holes around the defect.
    """
    eps_slab = n_GaAs**2
    eps_air  = 1.0
    # slab block
    geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_slab),
                     center=mp.Vector3(0, 0, 0),
                     material=mp.Medium(epsilon=eps_slab))]
    # triangular lattice basis vectors (in plane, xy)
    a1 = mp.Vector3(1.0, 0.0, 0.0)
    a2 = mp.Vector3(0.5, np.sqrt(3)/2, 0.0)
    # place holes at every lattice point within n_cells, EXCEPT the origin
    for i in range(-n_cells, n_cells+1):
        for j in range(-n_cells, n_cells+1):
            if i == 0 and j == 0: continue
            center = i*a1 + j*a2
            # limit to hexagonal region
            if abs(center.x) > n_cells + 0.1 or abs(center.y) > n_cells*np.sqrt(3)/2 + 0.1:
                continue
            geom.append(mp.Cylinder(radius=r_over_a,
                                    height=t_slab + 0.01,
                                    center=center,
                                    material=mp.Medium(epsilon=eps_air)))
    return geom


def run_h1_qnm(r_over_a=0.30, t_slab=0.833, n_cells=4,
               resolution=16, run_time=600,
               target_freq=0.255, df=0.07,
               source_comp=mp.Ex):
    """Run H1 PhC FDTD, extract fundamental dipole QNM via Harminv.

    n_cells controls the supercell size (tradeoff: larger -> higher accuracy
    for low-radiating modes, but quadratically slower).
    target_freq: a/lambda ~ 0.24 for H1 at r=0.3a.
    """
    geom = build_h1_geometry(r_over_a=r_over_a, n_cells=n_cells, t_slab=t_slab)
    # supercell: hexagonal region with n_cells hole rings; cell ~ 2(n_cells+0.5)a
    Lxy = 2*(n_cells + 1.5)
    Lz  = t_slab + 2.0    # 1 a of padding each side + PML
    pml = 1.0
    cell = mp.Vector3(Lxy, Lxy, Lz + 2*pml)
    pml_layers = [mp.PML(pml, direction=mp.Z),
                  mp.PML(pml*0.5, direction=mp.X),
                  mp.PML(pml*0.5, direction=mp.Y)]

    # source at the centre of the defect, polarised along source_comp
    src = mp.Source(mp.GaussianSource(target_freq, fwidth=df),
                    component=source_comp,
                    center=mp.Vector3(0.1, 0.05, 0))   # slightly off-center
    sim = mp.Simulation(cell_size=cell, geometry=geom,
                        boundary_layers=pml_layers,
                        sources=[src], resolution=resolution,
                        force_complex_fields=True,
                        symmetries=[])    # keep all modes

    h = mp.Harminv(source_comp, mp.Vector3(0.15, 0.1, 0), target_freq, df)
    t0 = time.time()
    sim.run(mp.after_sources(h), until_after_sources=run_time)
    wall_harminv = time.time() - t0
    if not h.modes:
        _pr("  no modes found by Harminv")
        return None

    # pick the mode closest to target freq with highest Q
    cand = [m for m in h.modes if abs(m.freq - target_freq) < df/2]
    if not cand:
        cand = h.modes
    best = max(cand, key=lambda m: m.Q)
    freq, Q = best.freq, best.Q
    omega_tilde = 2*np.pi*(freq - 1j*best.decay)
    _pr(f"  mode: f={freq:.4f}  Q={Q:.1f}  decay={best.decay:.2e}   "
          f"Harminv wall={wall_harminv:.1f}s")

    # Now DFT the full vector field at the resonance
    sim.reset_meep()
    sim = mp.Simulation(cell_size=cell, geometry=geom,
                        boundary_layers=pml_layers,
                        sources=[src], resolution=resolution,
                        force_complex_fields=True,
                        symmetries=[])
    # central slab cross-section (z=0) DFT for all 3 E components
    # Use a slab of thickness t_slab (full volume where mode lives)
    dft_center = mp.Vector3(0, 0, 0)
    dft_size   = mp.Vector3(Lxy*0.85, Lxy*0.85, t_slab*0.95)
    dft = sim.add_dft_fields([mp.Ex, mp.Ey, mp.Ez],
                              freq, 0, 1,
                              center=dft_center, size=dft_size)
    sim.run(until_after_sources=run_time)
    Ex = sim.get_dft_array(dft, mp.Ex, 0)
    Ey = sim.get_dft_array(dft, mp.Ey, 0)
    Ez = sim.get_dft_array(dft, mp.Ez, 0)
    # stack into single array
    shape = Ex.shape
    F = np.stack([Ex, Ey, Ez], axis=0)       # (3, Nx, Ny, Nz)
    # bilinear norm will be applied externally
    # grid spacings
    dx = dft_size.x / shape[0]
    dy = dft_size.y / shape[1]
    dz = dft_size.z / shape[2]
    return dict(freq=freq, Q=Q, decay=best.decay,
                omega_tilde=complex(omega_tilde),
                F=F, shape=shape, dx=dx, dy=dy, dz=dz,
                dft_size=(dft_size.x, dft_size.y, dft_size.z),
                resolution=resolution, r_over_a=r_over_a, t_slab=t_slab)


def bilinear_norm_3d(F, eps_array, dV):
    """Enforce integral eps F.F d^3r = 1 (unconjugated)."""
    integrand = eps_array * np.einsum('ixyz,ixyz->xyz', F, F)
    B = integrand.sum() * dV
    if abs(B) < 1e-30:
        return F, 0
    return F / np.sqrt(B), B


def overlap_g3_cross(fA, fB, chi3, dV):
    """Kleinman-isotropic chi^(3) cross-Kerr coupling."""
    mA2  = np.einsum('ixyz,ixyz->xyz', fA.conj(), fA)
    mB2  = np.einsum('ixyz,ixyz->xyz', fB.conj(), fB)
    AcBc = np.einsum('ixyz,ixyz->xyz', fA.conj(), fB.conj())
    BA   = np.einsum('ixyz,ixyz->xyz', fB, fA)
    AcB  = np.einsum('ixyz,ixyz->xyz', fA.conj(), fB)
    BcA  = np.einsum('ixyz,ixyz->xyz', fB.conj(), fA)
    return chi3 * (mA2*mB2 + AcBc*BA + AcB*BcA).sum() * dV


def overlap_g3_self(fA, chi3, dV):
    """Single-mode self-Kerr (should be exactly real, positive)."""
    mod2 = np.einsum('ixyz,ixyz->xyz', fA.conj(), fA).real
    return 3.0 * chi3 * (mod2**2).sum() * dV


def overlap_g2_Td(fA, fB, fC, chi2, dV):
    """Td Kleinman chi^(2) (GaAs-like d14).  chi^2_{xyz} only, all 6 permutations equal."""
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    integ = np.zeros(fA.shape[1:], dtype=complex)
    for (i,j,k) in perms:
        integ += fA[i].conj() * fB[j] * fC[k]
    return chi2 * integ.sum() * dV


def main():
    _pr("=== 3D GaAs H1 PhC Meep FDTD ===\n")

    # Debug-resolution sweep across hole-radius to tune Q.
    # At r/a = 0.30, Q is moderate (~10^3-10^4).  Vary r to sweep Q.
    # Also vary slab thickness to get degeneracy-breaking for diagnostics.

    # Q-tuning via hole radius.  Larger r -> more scattering -> lower Q.
    # Smaller r -> smaller modal confinement bandgap -> lower Q in a different way.
    # Expected Q sweep: 100 - 3000 at n_cells=4, res=14.
    runs = [
        dict(r_over_a=0.26, t_slab=0.833, n_cells=4, resolution=14,
             target_freq=0.235, df=0.08, run_time=500),
        dict(r_over_a=0.28, t_slab=0.833, n_cells=4, resolution=14,
             target_freq=0.250, df=0.08, run_time=500),
        dict(r_over_a=0.30, t_slab=0.833, n_cells=4, resolution=14,
             target_freq=0.265, df=0.08, run_time=500),
        dict(r_over_a=0.32, t_slab=0.833, n_cells=4, resolution=14,
             target_freq=0.280, df=0.08, run_time=500),
    ]

    # First pass: find modes along Ex (dipole_x) and Ey (dipole_y) for each config
    sweep_results = []
    for cfg in runs:
        _pr(f"--- r/a={cfg['r_over_a']:.2f}, t={cfg['t_slab']:.3f}, "
              f"cells={cfg['n_cells']}, res={cfg['resolution']} ---")
        # Ex dipole
        _pr(" dipole_x (source Ex)")
        rx = run_h1_qnm(source_comp=mp.Ex, **cfg)
        # Ey dipole
        _pr(" dipole_y (source Ey)")
        ry = run_h1_qnm(source_comp=mp.Ey, **cfg)
        if rx is None or ry is None:
            _pr("  skip (no mode)")
            continue
        sweep_results.append(dict(cfg=cfg, rx=rx, ry=ry))

    if not sweep_results:
        _pr("\nNO MODES FOUND IN ANY SWEEP POINT")
        return

    # Compute overlaps
    results = []
    for s in sweep_results:
        rx, ry = s['rx'], s['ry']
        # both modes share the same grid (same cfg)
        dV = rx['dx']*rx['dy']*rx['dz']
        Fx = rx['F']; Fy = ry['F']
        # Normalize bilinearly (use eps=1 approx — chi3 integrand is what matters)
        Fx_n, _ = bilinear_norm_3d(Fx, eps_array=1.0, dV=dV)
        Fy_n, _ = bilinear_norm_3d(Fy, eps_array=1.0, dV=dV)

        chi3 = 1.0
        chi2 = 1.0
        g3_xx = overlap_g3_self(Fx_n, chi3, dV)
        g3_yy = overlap_g3_self(Fy_n, chi3, dV)
        g3_xy = overlap_g3_cross(Fx_n, Fy_n, chi3, dV)
        # chi^(2): need three modes.  Use Fx, Fy, and (Fx+Fy) as third
        Fz = (Fx_n + Fy_n)/np.sqrt(2)
        Fz, _ = bilinear_norm_3d(Fz, eps_array=1.0, dV=dV)
        g2 = overlap_g2_Td(Fx_n, Fy_n, Fz, chi2, dV)

        imre_cross = abs(g3_xy.imag/g3_xy.real) if abs(g3_xy.real)>0 else 0
        imre_g2 = abs(g2.imag/g2.real) if abs(g2.real)>0 else 0
        Q_avg = (rx['Q'] + ry['Q'])/2
        _pr(f"\nr/a={s['cfg']['r_over_a']:.2f}  "
              f"Qx={rx['Q']:.1f}  Qy={ry['Q']:.1f}  "
              f"Im(g3_xy)/Re={imre_cross:.3e}  "
              f"Im(g2)/Re={imre_g2:.3e}")
        results.append(dict(r_over_a=s['cfg']['r_over_a'],
                            Q_x=float(rx['Q']), Q_y=float(ry['Q']),
                            freq_x=float(rx['freq']), freq_y=float(ry['freq']),
                            g3_xx_Re=float(g3_xx.real), g3_xx_Im=float(g3_xx.imag),
                            g3_yy_Re=float(g3_yy.real), g3_yy_Im=float(g3_yy.imag),
                            g3_xy_Re=float(g3_xy.real), g3_xy_Im=float(g3_xy.imag),
                            g2_Re=float(g2.real),     g2_Im=float(g2.imag),
                            imre_chi3_cross=float(imre_cross),
                            imre_chi2=float(imre_g2),
                            Q_avg=float(Q_avg)))

    with open(DATA_DIR/"meep_h1_3d_results.json", "w") as f:
        json.dump(results, f, indent=2)
    _pr(f"\nWrote {DATA_DIR}/meep_h1_3d_results.json")

    if len(results) < 2:
        _pr("Not enough data points for scaling plot.")
        return

    # Plot Im/Re vs 1/Q for both chi^(3) and chi^(2)
    Qs = np.array([r['Q_avg'] for r in results])
    r3 = np.array([r['imre_chi3_cross'] for r in results])
    r2 = np.array([r['imre_chi2'] for r in results])

    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    ax.loglog(1/Qs, np.clip(r3, 1e-18, None), 'bs', ms=10,
              label=r'$\chi^{(3)}$ Kleinman cross-Kerr')
    ax.loglog(1/Qs, r2, 'ro', ms=10,
              label=r'$\chi^{(2)}$ Kleinman $T_d$ (dipole$_x$, dipole$_y$, sum)')
    xref = np.logspace(-5, -1, 40)
    ax.loglog(xref, 1.5*xref, 'k--', lw=1.0, alpha=0.6,
              label=r'theory $(3/2)/Q$ for $\chi^{(2)}$')
    ax.axhline(1e-15, color='blue', alpha=0.3, ls=':',
               label='double-precision floor')
    ax.set_xlabel(r'$1/Q$', fontsize=11)
    ax.set_ylabel(r'$|\mathrm{Im}/\mathrm{Re}|$', fontsize=11)
    ax.set_title(r'3D GaAs H1 PhC Meep FDTD: Maxwell-extracted dipole QNMs',
                 fontsize=10)
    ax.legend(fontsize=9, loc='best')
    ax.grid(alpha=0.3, which='both')
    for r_ in results:
        ax.annotate(f"r/a={r_['r_over_a']:.2f}", (1/r_['Q_avg'], r_['imre_chi2']),
                    xytext=(5,0), textcoords='offset points', fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_meep_h1_3d.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_meep_h1_3d.png", dpi=150)
    _pr(f"Saved {FIG_DIR}/fig_meep_h1_3d.pdf")


if __name__ == "__main__":
    main()
