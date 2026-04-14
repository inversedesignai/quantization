"""Run all QNM Keldysh numerical verification scripts.

Sanity-check for PRX Quantum submission:
  1. Platform BSS / Mandel / Henry values (verify_platforms.py)
  2. Kleinman chi^(3) reality theorem over 2000 random synthetic QNM pairs
  3. chi^(2) Im/Re ~ 1/Q flagship prediction scaling (synthetic 3D QNMs)
  4. chi^(3) cross-Kerr vs 1/Q comparison (passive vs TPA-broken)
  5. Active-cavity cross-mode dephasing (dressed propagator)
  6. All 7 analytic tier-2/3 figures

Maxwell FDTD verifications (require Meep conda env):
  7. 1D Bragg-stack QNM, Q sweep over four decades   (meep_bragg_v2.py)
  8. 3D GaAs H1 PhC, hole-radius sweep at GaAs lattice (meep_h1_3d.py)

The Meep tests are run separately via:
    source /home/zlin/miniforge/etc/profile.d/conda.sh && conda activate meep
    cd qnm_synthetic && python meep_bragg_v2.py            (~2 min)
    mpirun -n 16 python meep_h1_3d.py                       (~15 min)
    python replot_h1_3d.py
"""
from __future__ import annotations
import subprocess, sys, os, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)

TESTS = [
    # Platform values + symmetry-dichotomy theorems
    ("verify_platforms.py",                           HERE),
    ("qnm_synthetic/test_kleinman_real.py",            HERE/"qnm_synthetic"),
    ("qnm_synthetic/test_chi2_scaling.py",             HERE/"qnm_synthetic"),
    ("qnm_synthetic/test_chi3_cross_kerr.py",          HERE/"qnm_synthetic"),
    ("qnm_synthetic/test_cross_dephasing.py",          HERE/"qnm_synthetic"),
    ("qnm_synthetic/test_chi2_fabryperot.py",          HERE/"qnm_synthetic"),
    # Reproductions of standard results & comparisons
    ("benchmarks/keldysh_vs_qutip_kerr.py",            HERE/"benchmarks"),
    ("benchmarks/cholesky_vs_FH.py",                   HERE/"benchmarks"),
    ("benchmarks/lau_clerk_EP.py",                     HERE/"benchmarks"),
    ("benchmarks/photon_blockade.py",                  HERE/"benchmarks"),
    ("benchmarks/two_mode_squeezing.py",               HERE/"benchmarks"),
    # Analytic figures
    ("analytic_figures/all_figures.py",                HERE/"analytic_figures"),
]

def main():
    t0 = time.time()
    results = []
    for path, cwd in TESTS:
        print(f"\n{'='*70}\n  RUN  {path}\n{'='*70}")
        t = time.time()
        p = subprocess.run([sys.executable, Path(path).name], cwd=cwd,
                           capture_output=True, text=True)
        dt = time.time() - t
        status = "OK" if p.returncode == 0 else "FAIL"
        results.append((path, status, dt))
        # print last ~15 lines of stdout to confirm success pattern
        tail = "\n".join(p.stdout.splitlines()[-15:])
        print(tail)
        if p.returncode != 0:
            print("STDERR:", p.stderr[-500:])

    print(f"\n\n{'='*70}\n  SUMMARY  (total {time.time()-t0:.1f} s)\n{'='*70}")
    for path, status, dt in results:
        print(f"  [{status}] ({dt:.1f}s)  {path}")

if __name__ == "__main__":
    main()
