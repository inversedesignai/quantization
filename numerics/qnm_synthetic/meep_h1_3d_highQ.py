"""
Push 3D H1 PhC Q higher by enlarging the supercell and modestly raising
resolution.  Two-point sweep at the GaAs target (r/a=0.30) with n_cells in
{4, 5, 6} to extrapolate Q.

This is the production run that addresses critique CRIT-5: the previous
n_cells=4, res=14 run gave Q~228, far below the GaAs H1 design Q~10^4.

The hypothesis is that radiation loss out the lateral PML, which is the
dominant Q-limiter at small n_cells, decreases roughly exponentially with
n_cells (each ring of holes provides additional bandgap confinement).
We test this by computing Q at three n_cells values.
"""
from __future__ import annotations
import numpy as np
import sys, time, json
sys.path.insert(0, "/home/zlin/miniforge/envs/meep/lib/python3.11/site-packages")
import meep as mp
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _pr(*args, **kwargs):
    if mp.am_master():
        import builtins
        builtins.print(*args, **kwargs, flush=True)


def build_h1_geometry(r_over_a=0.30, n_cells=5, t_slab=0.833, n_GaAs=3.5):
    eps_slab = n_GaAs**2
    geom = [mp.Block(size=mp.Vector3(mp.inf, mp.inf, t_slab),
                     center=mp.Vector3(),
                     material=mp.Medium(epsilon=eps_slab))]
    a1 = mp.Vector3(1.0, 0.0, 0.0)
    a2 = mp.Vector3(0.5, np.sqrt(3)/2, 0.0)
    for i in range(-n_cells, n_cells+1):
        for j in range(-n_cells, n_cells+1):
            if i == 0 and j == 0: continue
            c = i*a1 + j*a2
            if abs(c.x) > n_cells + 0.1: continue
            if abs(c.y) > n_cells*np.sqrt(3)/2 + 0.1: continue
            geom.append(mp.Cylinder(radius=r_over_a, height=t_slab+0.01,
                                    center=c,
                                    material=mp.Medium(epsilon=1.0)))
    return geom


def run(n_cells, resolution=16, run_time=None):
    r_over_a = 0.30
    t_slab = 0.833
    pml = 1.0
    geom = build_h1_geometry(r_over_a=r_over_a, n_cells=n_cells, t_slab=t_slab)
    Lxy = 2*(n_cells + 1.5)
    Lz = t_slab + 2.0
    cell = mp.Vector3(Lxy, Lxy, Lz + 2*pml)
    pml_layers = [mp.PML(pml, direction=mp.Z),
                  mp.PML(pml*0.5, direction=mp.X),
                  mp.PML(pml*0.5, direction=mp.Y)]
    target_freq = 0.27
    df = 0.07
    src = mp.Source(mp.GaussianSource(target_freq, fwidth=df),
                    component=mp.Ex,
                    center=mp.Vector3(0.10, 0.05, 0))
    sim = mp.Simulation(cell_size=cell, geometry=geom,
                        boundary_layers=pml_layers,
                        sources=[src], resolution=resolution,
                        force_complex_fields=True, symmetries=[])
    if run_time is None:
        # need run_time > Q (in units of period); estimate Q ~ 10^(n_cells*0.7)
        # so run_time scales appropriately; cap at 8000
        Q_est = 10**(n_cells*0.7)
        run_time = max(1000, min(8000, int(2 * Q_est / target_freq)))
    _pr(f"  N={n_cells}, res={resolution}, run_time={run_time}, "
        f"cell={Lxy:.1f}x{Lxy:.1f}x{Lz+2*pml:.2f}")
    h = mp.Harminv(mp.Ex, mp.Vector3(0.15, 0.10, 0), target_freq, df)
    t0 = time.time()
    sim.run(mp.after_sources(h), until_after_sources=run_time)
    wall = time.time() - t0
    if not h.modes:
        _pr(f"  N={n_cells}: no mode after {wall:.0f}s")
        return None
    cand = [m for m in h.modes if abs(m.freq - target_freq) < df/2]
    if not cand: cand = h.modes
    best = max(cand, key=lambda m: m.Q)
    _pr(f"  N={n_cells}: f={best.freq:.4f}  Q={best.Q:.1f}  wall={wall:.0f}s")
    return dict(n_cells=n_cells, resolution=resolution, run_time=run_time,
                freq=float(best.freq), Q=float(best.Q),
                decay=float(best.decay), wall=wall)


def main():
    _pr("=== 3D H1 PhC Q-vs-n_cells sweep ===\n")
    results = []
    # n_cells=4 baseline (matches earlier run); n_cells=5,6 to extrapolate
    for n_cells in [4, 5, 6]:
        r = run(n_cells, resolution=16)
        if r is not None:
            results.append(r)

    if not results:
        _pr("NO MODES")
        return

    Qs = np.array([r['Q'] for r in results])
    Ns = np.array([r['n_cells'] for r in results])
    # log-Q vs N gives slope = log10(per-ring Q multiplier)
    if len(Qs) >= 2:
        slope, intercept = np.polyfit(Ns, np.log10(Qs), 1)
        Q_at_8 = 10**(slope*8 + intercept)
        Q_at_10 = 10**(slope*10 + intercept)
        _pr(f"\nQ extrapolation: log10 Q = {slope:.2f}·N + {intercept:.2f}")
        _pr(f"  predicted Q(N=8) = {Q_at_8:.0f}")
        _pr(f"  predicted Q(N=10) = {Q_at_10:.0f}")
        _pr(f"  measured Q at N=4,5,6: {Qs}")

    with open(DATA_DIR/"meep_h1_q_sweep.json", "w") as f:
        json.dump(dict(results=results,
                       Q_extrapolation_slope=float(slope) if len(Qs)>=2 else None,
                       Q_extrapolation_intercept=float(intercept) if len(Qs)>=2 else None),
                   f, indent=2)
    _pr(f"\nWrote {DATA_DIR}/meep_h1_q_sweep.json")


if __name__ == "__main__":
    main()
