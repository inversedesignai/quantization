"""
CORE TEST — E11 verification.

For real Kleinman-symmetric chi^(3), the cross-Kerr coupling
g^(3)_{lambda mu mu lambda} is EXACTLY real for any complex mode functions
(passive cavities).  Test over 2000 random mode pairs with the full high-Q
structure f_tilde = f^(0)(1 + i phi/Q).
"""
import numpy as np
from synthetic_modes import build_vector_qnm, g3_cross_kleinman, g3_self_kleinman

def main():
    rng = np.random.default_rng(seed=20260414)
    Nx = Ny = Nz = 16           # moderate resolution
    shape = (Nx, Ny, Nz)
    dV = 1.0/(Nx*Ny*Nz)
    chi3 = 1.0

    N_pairs = 2000
    ratios = np.zeros(N_pairs)
    selfs  = np.zeros(N_pairs)

    # Sample Q values spanning the paper's regimes
    Qs = rng.uniform(1e2, 1e6, size=N_pairs)

    for idx in range(N_pairs):
        Q = Qs[idx]
        fA, f0A, phiA = build_vector_qnm(shape, rng, Q=Q, n_cells=2)
        fB, f0B, phiB = build_vector_qnm(shape, rng, Q=Q, n_cells=2)
        g3x = g3_cross_kleinman(fA, fB, chi3, dV)
        g3s = g3_self_kleinman(fA, chi3, dV)
        denom = max(abs(g3x.real), 1e-300)
        ratios[idx] = abs(g3x.imag/denom)
        selfs[idx]  = abs(g3s.imag)/max(abs(g3s.real), 1e-300)

    print(f"\n=== E11 CROSS-KERR REALITY THEOREM TEST ===")
    print(f"Number of random QNM pairs: {N_pairs}")
    print(f"Q range: [{Qs.min():.0f}, {Qs.max():.0f}]")
    print(f"\nCross-Kerr g^(3)_{{lambda mu mu lambda}} |Im/Re|:")
    print(f"  max  = {ratios.max():.3e}")
    print(f"  mean = {ratios.mean():.3e}")
    print(f"  99th percentile = {np.percentile(ratios,99):.3e}")
    print(f"\nSelf-Kerr g^(3)_{{lambda lambda lambda lambda}} |Im/Re|:")
    print(f"  max  = {selfs.max():.3e}")
    print(f"  mean = {selfs.mean():.3e}")

    # Assertion
    assert ratios.max() < 1e-12, \
        f"FAIL: Im(g3_cross)/Re = {ratios.max():.3e} > 1e-12 -- Kleinman theorem broken?"
    print("\n[PASS] Im(g^(3)_cross) = 0 at double-precision machine epsilon.")
    print("       The Kleinman-reality theorem (E11) holds algebraically.")

    np.savez("kleinman_real_results.npz", ratios=ratios, selfs=selfs, Qs=Qs)
    print("\nSaved kleinman_real_results.npz")

if __name__ == "__main__":
    main()
