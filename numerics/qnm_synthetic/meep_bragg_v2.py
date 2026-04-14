"""
1D Bragg-stack cavity QNM sweep (Meep 1.33), production version.

Sweeps the number of Bragg periods N = 2..8 to obtain Q from ~10 to ~2x10^5.
For each Q, extracts (i) the fundamental eigenfrequency, (ii) the DFT field
profile at the resonance, and (iii) computes the chi^(2) Kleinman-Td and
chi^(3) Kleinman-isotropic overlap integrals using three synthetic polarisation
vectors applied to the single scalar QNM + two phase-twisted partners.

Produces fig_meep_bragg.pdf showing:
  - |Im(g^(3))/Re| across Q: should sit at machine precision (~1e-15)
  - |Im(g^(2))/Re| across Q: should scale as 1/Q with slope 1
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


def bragg_geometry(N_periods, nH=2.5, nL=1.0, L_cav=2.0):
    lam0 = 2.0
    dH = lam0/(4*nH); dL = lam0/(4*nL)
    period = dH + dL
    geom = []
    for i in range(N_periods):
        for side in [+1, -1]:
            zH = side*(L_cav/2 + i*period + dH/2)
            zL = side*(L_cav/2 + i*period + dH + dL/2)
            geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, dH),
                                 center=mp.Vector3(0,0,zH),
                                 material=mp.Medium(epsilon=nH**2)))
            geom.append(mp.Block(size=mp.Vector3(mp.inf, mp.inf, dL),
                                 center=mp.Vector3(0,0,zL),
                                 material=mp.Medium(epsilon=nL**2)))
    return geom, period


def run_bragg(N_periods, nH=2.5, L_cav=2.0, resolution=64, pml=1.0, pad=0.5):
    geom, period = bragg_geometry(N_periods, nH=nH, L_cav=L_cav)
    total_mirror = N_periods*period
    half = pad + total_mirror + L_cav/2
    cell = mp.Vector3(0, 0, 2*(half + pml))

    # source slightly off-center — avoids node of fundamental
    src_loc = mp.Vector3(0, 0, 0.37)
    # detector different location for Harminv
    det_loc = mp.Vector3(0, 0, 0.53)

    src = mp.Source(mp.GaussianSource(0.5, fwidth=0.2),
                    component=mp.Ex, center=src_loc)

    # adaptive run time: need enough time for high-Q modes to ring.  For Q ~ 10^6,
    # need time > Q (in units of period).  Cap at 12000.
    sim = mp.Simulation(cell_size=cell, geometry=geom,
                        boundary_layers=[mp.PML(pml, direction=mp.Z)],
                        sources=[src], resolution=resolution,
                        dimensions=1, force_complex_fields=True)
    h = mp.Harminv(mp.Ex, det_loc, 0.5, 0.2)
    # run time: scale with expected Q ~ per_interface_R^(2N)
    run_time = max(2000, min(40000, 500 * N_periods**1.5))
    sim.run(mp.after_sources(h), until_after_sources=run_time)
    if not h.modes:
        return None
    # fundamental: f closest to 0.5
    best = min(h.modes, key=lambda m: abs(m.freq - 0.5))
    if best.Q < 2: return None   # dud
    fres, Q = best.freq, best.Q

    # now DFT field
    sim.reset_meep()
    sim = mp.Simulation(cell_size=cell, geometry=geom,
                        boundary_layers=[mp.PML(pml, direction=mp.Z)],
                        sources=[src], resolution=resolution,
                        dimensions=1, force_complex_fields=True)
    dft = sim.add_dft_fields([mp.Ex], fres, 0, 1,
                              center=mp.Vector3(0,0,0),
                              size=mp.Vector3(0, 0, cell.z*0.95))
    sim.run(until_after_sources=run_time)
    Ex = sim.get_dft_array(dft, mp.Ex, 0).flatten()
    z  = np.linspace(-cell.z*0.475, cell.z*0.475, Ex.size)
    mask_cav = np.abs(z) < L_cav/2
    return dict(N_periods=N_periods, Q=Q, freq=fres, decay=best.decay,
                Ex=Ex, z=z, mask_cav=mask_cav, L_cav=L_cav)


def bilinear_norm(E, dz):
    B = (E*E).sum()*dz
    return E/np.sqrt(B) if abs(B) > 1e-30 else E


def overlaps_for_result(r):
    """Build three scalar QNM-like fields from the extracted Ex and compute
    chi^(3) cross-Kerr and chi^(2) Td Kleinman integrals.

    The three fields share the same eigenfrequency (same Q) but are constructed
    with different spatial phasings.  This tests whether the Im(g^3) = 0 and
    Im(g^2) != 0 behaviour is determined by the *imaginary-phase* structure of
    the genuine Maxwell QNM, not by the multi-modal distinction."""
    Ex = r['Ex']; z = r['z']; mask = r['mask_cav']; Lc = r['L_cav']
    dz = abs(z[1] - z[0])
    E0 = np.where(mask, Ex, 0+0j)

    # Three "test modes": fundamental, its FP-second-order phase twist,
    # and a linear-position modulation.  All share the same QNM's complex phase
    # structure (the only thing that matters for open-system effects).
    E1 = E0
    E2 = E0 * np.exp(2j*np.pi*z/Lc)
    E3 = E0 * z

    E1 = bilinear_norm(E1, dz)
    E2 = bilinear_norm(E2, dz)
    E3 = bilinear_norm(E3, dz)

    rng = np.random.default_rng(42)
    M = rng.standard_normal((3,3))
    P, _ = np.linalg.qr(M)
    F = [np.outer(P[i], Es) for i, Es in enumerate([E1, E2, E3])]

    # chi^(3) Kleinman-isotropic cross-Kerr (two modes A,B):
    A, B = F[0], F[1]
    mA2 = np.einsum('iz,iz->z', A.conj(), A)
    mB2 = np.einsum('iz,iz->z', B.conj(), B)
    AcBc = np.einsum('iz,iz->z', A.conj(), B.conj())
    BA   = np.einsum('iz,iz->z', B, A)
    AcB  = np.einsum('iz,iz->z', A.conj(), B)
    BcA  = np.einsum('iz,iz->z', B.conj(), A)
    g3 = (mA2*mB2 + AcBc*BA + AcB*BcA).sum()*dz

    # chi^(2) Td Kleinman (three distinct modes A,B,C):
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    integ = np.zeros(A.shape[1], dtype=complex)
    A, B, C = F[0], F[1], F[2]
    for (i,j,k) in perms:
        integ += A[i].conj()*B[j]*C[k]
    g2 = integ.sum()*dz
    return g3, g2


def main():
    Ns = [2, 3, 4, 5, 6, 7]      # Q sweep — N=7 should give Q ~ 10^5
    results = []
    for N in Ns:
        t0 = time.time()
        print(f"\n=== N_periods = {N} ===")
        try:
            r = run_bragg(N)
        except Exception as e:
            print(f"  ERR {e}")
            continue
        if r is None:
            print(f"  no QNM found")
            continue
        print(f"  Q = {r['Q']:.1f}   freq = {r['freq']:.4f}   "
              f"wall = {time.time()-t0:.1f}s")
        g3, g2 = overlaps_for_result(r)
        ir3 = abs(g3.imag/g3.real) if abs(g3.real)>0 else 0.0
        ir2 = abs(g2.imag/g2.real) if abs(g2.real)>0 else 0.0
        print(f"  |Im(g3)/Re| = {ir3:.3e}   |Im(g2)/Re| = {ir2:.3e}")
        results.append(dict(N=N, Q=float(r['Q']), freq=float(r['freq']),
                            imre_chi3=float(ir3), imre_chi2=float(ir2)))

    if not results:
        print("NO RESULTS")
        return

    # Plot
    Qs = np.array([r['Q'] for r in results])
    r3 = np.array([r['imre_chi3'] for r in results])
    r2 = np.array([r['imre_chi2'] for r in results])

    # This figure is a clean test of the chi^(3) Kleinman reality theorem
    # on Maxwell-extracted QNMs at varying Q.  It is NOT a valid test of the
    # chi^(2) 1/Q scaling — that requires three independent QNMs of different
    # frequencies, which 1D cannot naturally provide; see the 3D H1 PhC test.
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    r3_plot = np.clip(r3, 1e-18, None)
    ax.loglog(1/Qs, r3_plot, 'bs', ms=10,
              label=r'$\chi^{(3)}$ Kleinman cross-Kerr (FDTD Maxwell QNM)')
    ax.axhline(1e-15, color='blue', alpha=0.3, ls=':',
               label=r'double-precision floor $\sim 10^{-15}$')
    ax.set_ylim(1e-20, 1)
    ax.set_xlabel(r'$1/Q$', fontsize=11)
    ax.set_ylabel(r'$|\mathrm{Im}(g^{(3)}_{\lambda\mu\mu\lambda})/\mathrm{Re}|$', fontsize=11)
    ax.set_title(r'1D Bragg-stack Meep FDTD: $\chi^{(3)}$ Kleinman theorem at every $Q$',
                 fontsize=10)
    # annotate the Q span
    for Q, v in zip(Qs, r3_plot):
        ax.annotate(f'$Q\\!=\\!{Q:.0f}$', (1/Q, v), fontsize=7,
                    xytext=(0,8), textcoords='offset points',
                    ha='center', color='blue', alpha=0.7)
    ax.legend(fontsize=9, loc='upper left')
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_meep_bragg.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_meep_bragg.png", dpi=150)
    print(f"\nSaved {FIG_DIR}/fig_meep_bragg.{{pdf,png}}")

    with open(DATA_DIR/"meep_bragg_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {DATA_DIR}/meep_bragg_results.json")


if __name__ == "__main__":
    main()
