"""Recompute BSS table, self-Kerr per platform, verify all paper values (E25 etc.)."""
import numpy as np
from common import (PLATFORMS, omega_from_lam, kappa_from_Q, self_kerr_n2,
                    bss_shift, mode_volume_for)

def fmt(x, unit=""):
    if x == 0: return f"0 {unit}"
    e = int(np.floor(np.log10(abs(x))))
    m = x/10**e
    return f"{m:.2f}e{e:+d} {unit}".strip()

lines = []
lines.append(f"{'Platform':18s}  {'g^(3)/2pi':>12s}  {'kappa/2pi':>12s}  {'g^(3)/omega':>11s}  {'BSS/2pi':>12s}")
lines.append("-"*80)

for key, p in PLATFORMS.items():
    if key == "Circuit_QED":
        omega = p["omega"]
        kappa = kappa_from_Q(omega, p["Q"])
        # Circuit QED: g3 is given phenomenologically (Kerr anharmonicity)
        g3_lo, g3_hi = p["g3_Hz_range"]
        g3_ang_lo = 2*np.pi*g3_lo
        g3_ang_hi = 2*np.pi*g3_hi
        bss_lo = bss_shift(g3_ang_lo, omega, kappa)
        bss_hi = bss_shift(g3_ang_hi, omega, kappa)
        g3over_lo = g3_ang_lo/omega
        g3over_hi = g3_ang_hi/omega
        lines.append(f"{p['name']:18s}  "
                     f"{g3_lo/1e6:4.0f}-{g3_hi/1e6:3.0f} MHz  "
                     f"{kappa/(2*np.pi*1e6):7.3f} MHz  "
                     f"{g3over_lo:.1e}-{g3over_hi:.1e}  "
                     f"{bss_lo/(2*np.pi):6.3f}-{bss_hi/(2*np.pi):4.2f} Hz")
        continue

    omega = omega_from_lam(p["lam"])
    kappa = kappa_from_Q(omega, p["Q"])
    V = mode_volume_for(p)
    g3_ang = self_kerr_n2(p)          # g3 in rad/s (since formula uses hbar*omega)
    g3_Hz  = g3_ang/(2*np.pi)
    bss_ang = bss_shift(g3_ang, omega, kappa)
    bss_Hz = bss_ang/(2*np.pi)

    # human-readable units
    if g3_Hz >= 1e6:   g3str = f"{g3_Hz/1e6:.2f} MHz"
    elif g3_Hz >= 1e3: g3str = f"{g3_Hz/1e3:.0f} kHz"
    elif g3_Hz >= 1:   g3str = f"{g3_Hz:.1f} Hz"
    else:              g3str = f"{g3_Hz:.2e} Hz"

    if kappa/(2*np.pi) >= 1e9: kstr = f"{kappa/(2*np.pi)/1e9:.1f} GHz"
    elif kappa/(2*np.pi) >= 1e6: kstr = f"{kappa/(2*np.pi)/1e6:.1f} MHz"
    else: kstr = f"{kappa/(2*np.pi):.2f} Hz"

    lines.append(f"{p['name']:18s}  {g3str:>12s}  {kstr:>12s}  {g3_ang/omega:11.2e}  {fmt(bss_Hz,'Hz'):>12s}")

print("\nPlatform verification table (E25 formula: BSS = g^2 kappa / (8 omega^2)):\n")
print("\n".join(lines))

# Compare to CLAUDE.md claims
print("\n--- Comparison to CLAUDE.md claimed values ---")
claims = {
    "Si_microring":   dict(g3_Hz=294,    kappa_Hz=194e6,  bss_Hz=6e-17),
    "GaAs_H1_PhC":    dict(g3_Hz=235e3,  kappa_Hz=10e9,   bss_Hz=8e-10),
    "LiNbO3_WGM":     dict(g3_Hz=0.1,    kappa_Hz=2,      bss_Hz=1e-30),
}
for key, c in claims.items():
    p = PLATFORMS[key]
    omega = omega_from_lam(p["lam"])
    kappa = kappa_from_Q(omega, p["Q"])
    g3 = self_kerr_n2(p)
    bss = bss_shift(g3, omega, kappa)
    print(f"{p['name']:18s}  computed: g3={g3/(2*np.pi):.2e} Hz  kappa={kappa/(2*np.pi):.2e} Hz  BSS={bss/(2*np.pi):.2e} Hz")
    print(f"{'':18s}  claimed : g3={c['g3_Hz']:.2e} Hz  kappa={c['kappa_Hz']:.2e} Hz  BSS={c['bss_Hz']:.2e} Hz")

# Specifically confirm E25 factor of 2Q error-correction
print("\n--- E25 sanity: BSS should be (g3)^2 kappa/(8 omega^2), NOT g3^2/(4 omega) ---")
p = PLATFORMS["GaAs_H1_PhC"]
omega = omega_from_lam(p["lam"])
kappa = kappa_from_Q(omega, p["Q"])
g3 = self_kerr_n2(p)
wrong = g3**2 / (4*omega)
right = g3**2 * kappa/(8*omega**2)
print(f"Wrong formula  g3^2/(4 omega) = {wrong/(2*np.pi):.2e} Hz")
print(f"Right formula  g3^2 kappa/(8 omega^2) = {right/(2*np.pi):.2e} Hz")
print(f"Ratio wrong/right = 2Q = {wrong/right:.2e}  (should equal 2Q = {2*p['Q']:.2e})")
