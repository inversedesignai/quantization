"""
1D Bragg-stack cavity: QNM with tunable Q = 10^2 - 10^5 via number of Bragg periods.

Geometry: [PML][air][N pairs of (nH, nL)][cavity, n=1, length L0][N pairs (nH, nL)][air][PML]
Each quarter-wave pair forms a Bragg mirror; more pairs -> higher Q.

Extract the fundamental QNM, compute chi^(3) and chi^(2) Im/Re ratios using
three different cavity modes (m=1, m=2, m=3 longitudinal orders).
"""
from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import sys
sys.path.insert(0, "/home/zlin/miniforge/envs/meep/lib/python3.11/site-packages")
import meep as mp

FIG_DIR = Path(__file__).resolve().parents[1] / "figs"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def build_bragg(N_periods, nH=2.5, nL=1.0, L_cavity=2.0, pml_thick=1.0):
    # quarter-wave stack at design wavelength lambda0 = 2 (in a.u.); freq=0.5
    lam0 = 2.0
    dH = lam0/(4*nH)
    dL = lam0/(4*nL)
    period = dH + dL

    # Build symmetric mirror region of N_periods on each side
    zcenter_mirror_R_start = L_cavity/2
    zcenter_mirror_L_start = -L_cavity/2
    geom = []
    for i in range(N_periods):
        # right mirror
        zcH = zcenter_mirror_R_start + i*period + dH/2
        zcL = zcenter_mirror_R_start + i*period + dH + dL/2
        geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, dH),
                             center=mp.Vector3(0, 0, zcH),
                             material=mp.Medium(epsilon=nH**2)))
        geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, dL),
                             center=mp.Vector3(0, 0, zcL),
                             material=mp.Medium(epsilon=nL**2)))
        # left mirror
        zcH = zcenter_mirror_L_start - i*period - dH/2
        zcL = zcenter_mirror_L_start - i*period - dH - dL/2
        geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, dH),
                             center=mp.Vector3(0, 0, zcH),
                             material=mp.Medium(epsilon=nH**2)))
        geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, dL),
                             center=mp.Vector3(0, 0, zcL),
                             material=mp.Medium(epsilon=nL**2)))
    # total length
    total_mirror = N_periods*period
    pad = 1.0
    half = pad + total_mirror + L_cavity/2
    cell = mp.Vector3(0, 0, 2*(half + pml_thick))
    pml = [mp.PML(pml_thick, direction=mp.Z)]
    return cell, pml, geom


def extract_fundamental(N_periods, resolution=64, run_time=3000, nH=2.5, L_cavity=2.0):
    cell, pml, geom = build_bragg(N_periods, nH=nH, nL=1.0, L_cavity=L_cavity)
    fcen = 0.5
    df = 0.35
    src = mp.Source(mp.GaussianSource(fcen, fwidth=df),
                    component=mp.Ex,
                    center=mp.Vector3(0, 0, 0.1))
    sim = mp.Simulation(cell_size=cell, geometry=geom, boundary_layers=pml,
                        sources=[src], resolution=resolution, dimensions=1,
                        force_complex_fields=True)
    h = mp.Harminv(mp.Ex, mp.Vector3(0, 0, 0.23), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=run_time)
    if not h.modes:
        return None
    cand = [m for m in h.modes if abs(m.freq-fcen) < df/2]
    if not cand: return None
    best = max(cand, key=lambda m: m.Q)

    sim.reset_meep()
    sim = mp.Simulation(cell_size=cell, geometry=geom, boundary_layers=pml,
                        sources=[src], resolution=resolution, dimensions=1,
                        force_complex_fields=True)
    dft = sim.add_dft_fields([mp.Ex], best.freq, 0, 1,
                              center=mp.Vector3(0,0,0),
                              size=mp.Vector3(0, 0, cell.z*0.95))
    sim.run(until_after_sources=run_time)
    Ex = sim.get_dft_array(dft, mp.Ex, 0).flatten()
    z = np.linspace(-cell.z*0.475, cell.z*0.475, Ex.size)
    mask = np.abs(z) < L_cavity/2
    return dict(Q=best.Q, freq=best.freq, decay=best.decay,
                z=z, Ex=Ex, mask=mask, N=N_periods, L_cav=L_cavity)


def extract_multiple_orders(N_periods, resolution=64, run_time=3000, nH=2.5, L_cavity=4.0):
    """Extract m=1, m=2, m=3 longitudinal modes.  Use L_cavity=4 so modes fit at
    fcen=0.25, 0.5, 0.75.
    """
    modes_out = []
    for target_f, Lc in [(0.25, L_cavity), (0.5, L_cavity), (0.75, L_cavity)]:
        cell, pml, geom = build_bragg(N_periods, nH=nH, nL=1.0, L_cavity=Lc)
        src = mp.Source(mp.GaussianSource(target_f, fwidth=0.08),
                        component=mp.Ex, center=mp.Vector3(0,0, 0.1))
        sim = mp.Simulation(cell_size=cell, geometry=geom, boundary_layers=pml,
                            sources=[src], resolution=resolution, dimensions=1,
                            force_complex_fields=True)
        h = mp.Harminv(mp.Ex, mp.Vector3(0,0,0.23), target_f, 0.08)
        sim.run(mp.after_sources(h), until_after_sources=run_time)
        if not h.modes: continue
        best = max(h.modes, key=lambda m: m.Q)

        sim.reset_meep()
        sim = mp.Simulation(cell_size=cell, geometry=geom, boundary_layers=pml,
                            sources=[src], resolution=resolution, dimensions=1,
                            force_complex_fields=True)
        dft = sim.add_dft_fields([mp.Ex], best.freq, 0, 1,
                                  center=mp.Vector3(0,0,0),
                                  size=mp.Vector3(0, 0, cell.z*0.95))
        sim.run(until_after_sources=run_time)
        Ex = sim.get_dft_array(dft, mp.Ex, 0).flatten()
        z = np.linspace(-cell.z*0.475, cell.z*0.475, Ex.size)
        mask = np.abs(z) < Lc/2
        modes_out.append(dict(Q=best.Q, freq=best.freq, decay=best.decay,
                              z=z, Ex=Ex, mask=mask, order=len(modes_out)+1))
    return modes_out


def bilinear_normalize(E, dz):
    B = (E*E).sum()*dz
    return E/np.sqrt(B) if abs(B) > 0 else E


def overlaps_from_modes(modes, dz):
    """Take 3 scalar QNM fields, assign 3 orthonormal polarisations, compute
    chi^(3) cross-Kerr and chi^(2) Td Kleinman overlaps."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((3,3))
    P, _ = np.linalg.qr(M)

    # Interpolate all modes onto the same z grid (use first mode's grid)
    z0 = modes[0]['z']
    mask0 = modes[0]['mask']
    Es_on_z0 = []
    for m in modes:
        if np.array_equal(m['z'], z0):
            E = np.where(mask0, m['Ex'], 0+0j)
        else:
            E_on_z0 = np.interp(z0, m['z'], m['Ex'].real) + 1j*np.interp(z0, m['z'], m['Ex'].imag)
            E = np.where(mask0, E_on_z0, 0+0j)
        Es_on_z0.append(bilinear_normalize(E, dz))

    # build 3D vector modes
    Fs = [np.outer(P[i], Es_on_z0[i]) for i in range(3)]

    # chi^(3) cross-Kerr (passive Kleinman isotropic chi^3) between modes 0 and 1
    A, B, C = Fs[0], Fs[1], Fs[2]
    mA2 = np.einsum('iz,iz->z', A.conj(), A)
    mB2 = np.einsum('iz,iz->z', B.conj(), B)
    AcBc = np.einsum('iz,iz->z', A.conj(), B.conj())
    BA   = np.einsum('iz,iz->z', B, A)
    AcB  = np.einsum('iz,iz->z', A.conj(), B)
    BcA  = np.einsum('iz,iz->z', B.conj(), A)
    g3 = (mA2*mB2 + AcBc*BA + AcB*BcA).sum()*dz

    # chi^(2) Td: (A*, B, C)_ijk fully symmetric
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    integ2 = np.zeros(A.shape[1], dtype=complex)
    for (i,j,k) in perms:
        integ2 += A[i].conj()*B[j]*C[k]
    g2 = integ2.sum()*dz

    return g3, g2


def main():
    N_list = [2, 3, 4, 5, 6, 7, 8]      # sweep Q by number of Bragg periods
    results = []
    for N in N_list:
        print(f"\n=== N_periods = {N} ===")
        try:
            modes = extract_multiple_orders(N, resolution=64, run_time=2500,
                                             nH=2.5, L_cavity=4.0)
        except Exception as e:
            print(f"  FAIL: {e}")
            continue
        if len(modes) < 3:
            print(f"  only got {len(modes)} modes, skipping")
            continue
        Qs_local = [m['Q'] for m in modes]
        print(f"  Qs = {Qs_local}")
        dz = abs(modes[0]['z'][1]-modes[0]['z'][0])
        g3, g2 = overlaps_from_modes(modes, dz)
        imre3 = abs(g3.imag/g3.real) if abs(g3.real) > 0 else 0
        imre2 = abs(g2.imag/g2.real) if abs(g2.real) > 0 else 0
        Q_typical = np.mean(Qs_local)
        print(f"  <Q> = {Q_typical:.1f}   |Im(g3)/Re| = {imre3:.3e}   |Im(g2)/Re| = {imre2:.3e}")
        results.append(dict(N=N, Q=Q_typical, imre_chi3=imre3, imre_chi2=imre2))

    if not results:
        print("NO RESULTS")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    Qs = np.array([r['Q'] for r in results])
    r3 = np.array([r['imre_chi3'] for r in results])
    r2 = np.array([r['imre_chi2'] for r in results])
    ax.loglog(1/Qs, np.clip(r3, 1e-18, None), 'bs', ms=8,
              label=r'$\chi^{(3)}$ Kleinman (FDTD QNM, should be ~0)')
    ax.loglog(1/Qs, r2, 'ro-', ms=8,
              label=r'$\chi^{(2)}$ Kleinman $T_d$ (FDTD QNM)')
    xref = np.logspace(-5, -1, 50)
    ax.loglog(xref, 1.5*xref, 'k--', lw=1.0, label=r'theory $(3/2)/Q$')
    ax.set_xlabel(r'$1/Q$', fontsize=11)
    ax.set_ylabel(r'$|\mathrm{Im}/\mathrm{Re}|$', fontsize=11)
    ax.set_title('Meep 1D Bragg-stack FDTD: Maxwell-extracted QNMs',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_meep_bragg.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_meep_bragg.png", dpi=150)
    print(f"\nSaved {FIG_DIR}/fig_meep_bragg.pdf")

    import json
    with open(DATA_DIR/"meep_bragg_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
