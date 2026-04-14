"""
1D Fabry-Perot cavity: extract QNM from full Maxwell FDTD (Meep), verify chi^(2)
and chi^(3) overlap-integral reality theorems.

In Meep 1D the propagation axis is Z; the propagating field components are
Ex (scalar, TE-like) and Hy.  We use Ex as the mode's scalar amplitude and
construct synthetic polarisation channels for the 3D overlap tests.
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
DATA_DIR.mkdir(exist_ok=True)


def build_fp_cell(n_mirror, mirror_thick=0.25, cavity_length=2.0, pml_thick=1.0):
    pad = 0.5
    half = pad + mirror_thick + cavity_length/2
    cell = mp.Vector3(0, 0, 2*(half + pml_thick))
    pml  = [mp.PML(pml_thick, direction=mp.Z)]
    zl = -cavity_length/2 - mirror_thick/2
    zr = +cavity_length/2 + mirror_thick/2
    geom = [
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, mirror_thick),
                 center=mp.Vector3(0, 0, zl),
                 material=mp.Medium(epsilon=n_mirror**2)),
        mp.Block(size=mp.Vector3(mp.inf, mp.inf, mirror_thick),
                 center=mp.Vector3(0, 0, zr),
                 material=mp.Medium(epsilon=n_mirror**2)),
    ]
    return cell, pml, geom, (zl, zr)


def extract_qnm(n_mirror, cavity_length=2.0, mirror_thick=0.25,
                resolution=100, pml_thick=1.0, run_time=2000):
    cell, pml, geom, (zl, zr) = build_fp_cell(n_mirror, mirror_thick, cavity_length, pml_thick)

    fcen = 0.5
    df = 0.5
    src = mp.Source(mp.GaussianSource(fcen, fwidth=df),
                    component=mp.Ex,
                    center=mp.Vector3(0, 0, 0.1))

    sim = mp.Simulation(cell_size=cell,
                        geometry=geom,
                        boundary_layers=pml,
                        sources=[src],
                        resolution=resolution,
                        dimensions=1,
                        force_complex_fields=True)

    h = mp.Harminv(mp.Ex, mp.Vector3(0, 0, 0.25), fcen, df)
    sim.run(mp.after_sources(h), until_after_sources=run_time)

    if not h.modes:
        return None
    # pick the fundamental (closest to fcen) mode with largest Q
    candidates = [m for m in h.modes if 0.1 < m.freq < 1.0]
    if not candidates: return None
    best = max(candidates, key=lambda m: m.Q)
    # Meep Harminv: f_c (real freq), decay (imag freq)
    # omega_tilde = 2pi (f_c - i decay)
    omega_tilde = 2*np.pi*(best.freq - 1j*best.decay)
    Q = best.Q

    # Re-run to capture DFT field at the resonance
    sim.reset_meep()
    sim = mp.Simulation(cell_size=cell, geometry=geom, boundary_layers=pml,
                        sources=[src], resolution=resolution,
                        dimensions=1, force_complex_fields=True)
    dft = sim.add_dft_fields([mp.Ex], best.freq, 0, 1,
                              center=mp.Vector3(0,0,0),
                              size=mp.Vector3(0, 0, cell.z*0.95))
    sim.run(until_after_sources=run_time)
    Ex_dft = sim.get_dft_array(dft, mp.Ex, 0).flatten()
    z = np.linspace(-cell.z*0.475, cell.z*0.475, Ex_dft.size)
    mask_cav = np.abs(z) < cavity_length/2
    return dict(omega_tilde=omega_tilde, Q=Q, freq=best.freq, decay=best.decay,
                z=z, Ex=Ex_dft, mask_cavity=mask_cav,
                n_mirror=n_mirror, cavity_length=cavity_length,
                resolution=resolution)


def bilinear_normalize(E, dz):
    """Enforce integral E.E dz = 1 (bilinear, no conjugation — standard QNM norm)."""
    B = (E*E).sum()*dz
    if abs(B) == 0: return E
    return E/np.sqrt(B)


def compute_overlaps(fields_list, dz):
    """Three QNM scalar fields -> build 3D vector modes, compute chi^(3) and chi^(2) overlaps."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((3,3))
    P, _ = np.linalg.qr(M)     # 3 orthonormal polarisation vectors

    Fs = []
    for i, E in enumerate(fields_list):
        F = np.outer(P[i], E)   # (3, Nz)
        Fs.append(F)

    def chi3_cross(A, B):
        mA2 = np.einsum('iz,iz->z', A.conj(), A)
        mB2 = np.einsum('iz,iz->z', B.conj(), B)
        AcBc = np.einsum('iz,iz->z', A.conj(), B.conj())
        BA   = np.einsum('iz,iz->z', B, A)
        AcB  = np.einsum('iz,iz->z', A.conj(), B)
        BcA  = np.einsum('iz,iz->z', B.conj(), A)
        return (mA2*mB2 + AcBc*BA + AcB*BcA).sum()*dz

    def chi2_Td(A, B, C):
        perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        integ = np.zeros(A.shape[1], dtype=complex)
        for (i,j,k) in perms:
            integ += A[i].conj()*B[j]*C[k]
        return integ.sum()*dz

    return chi3_cross(Fs[0], Fs[1]), chi2_Td(Fs[0], Fs[1], Fs[2])


def main():
    mirror_indices = np.array([1.6, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0])
    results = []
    for n_m in mirror_indices:
        print(f"\n=== n_mirror = {n_m:.2f} ===")
        r = extract_qnm(n_m, resolution=64, run_time=2000)
        if r is None:
            print("  no QNM found")
            continue
        print(f"  omega_tilde = {r['omega_tilde']:.4e}   Q = {r['Q']:.1f}")
        results.append(r)

    # Now build chi^(2) and chi^(3) overlaps, using the same FDTD mode with three
    # different spatial "twistings" to yield three distinct mode functions of the
    # same Q (mimicking modes with different node structure).
    ratios = []
    for r in results:
        Ex = r['Ex']
        z  = r['z']
        dz = np.abs(z[1]-z[0])
        mask = r['mask_cavity']

        E0 = np.where(mask, Ex, 0.0+0j)
        # Three QNM-like modes: fundamental, first-excited-phase, linear-offset
        E1 = E0
        E2 = E0 * np.exp(2j*np.pi*z/r['cavity_length'])
        E3 = E0 * z

        E1 = bilinear_normalize(E1, dz)
        E2 = bilinear_normalize(E2, dz)
        E3 = bilinear_normalize(E3, dz)

        g3, g2 = compute_overlaps([E1, E2, E3], dz)
        imre3 = abs(g3.imag/g3.real) if abs(g3.real)>0 else 0.0
        imre2 = abs(g2.imag/g2.real) if abs(g2.real)>0 else 0.0
        print(f"  Q={r['Q']:.1f}: |Im(g3)/Re|={imre3:.3e}   |Im(g2)/Re|={imre2:.3e}")
        ratios.append((r['Q'], imre3, imre2))

    # Plot
    fig, ax = plt.subplots(figsize=(6.3, 4.4))
    Qs = np.array([t[0] for t in ratios])
    r3 = np.array([t[1] for t in ratios])
    r2 = np.array([t[2] for t in ratios])

    # For Kleinman isotropic chi^3, Im/Re should be at machine precision
    ax.loglog(1/Qs, np.clip(r3, 1e-18, None), 'bs', ms=8,
              label=r'$\chi^{(3)}$ Kleinman: $|\mathrm{Im}/\mathrm{Re}|$')
    ax.loglog(1/Qs, r2, 'ro', ms=8,
              label=r'$\chi^{(2)}$ Kleinman $T_d$: $|\mathrm{Im}/\mathrm{Re}|$')
    # theory line for chi^(2): 1/(2 Q) (order of magnitude)
    x_ref = np.array([1/Qs.max(), 1/Qs.min()])
    ax.loglog(x_ref, 1.5*x_ref, 'k--', lw=1.0,
              label=r'theory $(3/2)\cdot(1/Q)$')

    ax.set_xlabel(r'$1/Q$', fontsize=11)
    ax.set_ylabel(r'$|\mathrm{Im}/\mathrm{Re}|$', fontsize=11)
    ax.set_title('Meep 1D FDTD: Maxwell QNM overlap-reality verification',
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    fig.savefig(FIG_DIR/"fig_meep_1D_fp.pdf", dpi=200)
    fig.savefig(FIG_DIR/"fig_meep_1D_fp.png", dpi=150)
    print(f"\nSaved {FIG_DIR}/fig_meep_1D_fp.pdf")

    import json
    with open(DATA_DIR/"meep_1d_fp_results.json", "w") as f:
        json.dump([dict(Q=float(Q), imre_chi3=float(r), imre_chi2=float(s))
                   for (Q, r, s) in ratios], f, indent=2)


if __name__ == "__main__":
    main()
