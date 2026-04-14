"""Shared constants and physical parameters for QNM Keldysh numerics."""
import numpy as np

hbar = 1.054_571_817e-34     # J s
eps0 = 8.854_187_817e-12     # F/m
c    = 2.997_924_58e8        # m/s

PLATFORMS = {
    "Si_microring": dict(
        name = "Si microring",
        lam  = 1550e-9,
        n    = 3.48,
        n2   = 6e-18,           # m^2/W
        Aeff = 0.10e-12,        # 0.1 um^2
        R    = 5e-6,            # ring radius
        Q    = 1e6,
        chi2_pm_per_V = 0.0,    # centrosymmetric
    ),
    "GaAs_H1_PhC": dict(
        name = "GaAs H1 PhC",
        lam  = 1000e-9,
        n    = 3.5,
        n2   = 1.5e-17,         # m^2/W
        Veff = 0.023e-18,       # (lambda/n)^3 = 0.023 um^3
        Q    = 3e4,
        chi2_pm_per_V = 200.0,  # d14
    ),
    "LiNbO3_WGM": dict(
        name = "LiNbO3 WGM",
        lam  = 1550e-9,
        n    = 2.21,
        n2   = 2.5e-19,
        Veff = 1e4 * 1e-18,     # 10^4 um^3
        Q    = 1e8,
        chi2_pm_per_V = 30.0,   # d33
    ),
    "Circuit_QED": dict(
        name = "Circuit QED (transmon)",
        omega = 2*np.pi*5e9,    # 5 GHz
        Q    = 1e4,
        g3_Hz_range = (10e6, 100e6),   # explicit input; not computed from n2
    ),
}

def omega_from_lam(lam):
    return 2*np.pi*c/lam

def kappa_from_Q(omega, Q):
    return omega/Q

def mode_volume_for(p):
    if "Veff" in p: return p["Veff"]
    if "Aeff" in p and "R" in p:
        return p["Aeff"] * 2*np.pi*p["R"]
    raise KeyError("no Veff or ring geometry")

def self_kerr_n2(p):
    """g^(3) self-Kerr from n_2 formula:  g3 = hbar omega^2 c n_2 / (4 n^2 V_eff).
    See Eq. g3_n2_formula in the paper.  Returns g3 in rad/s (angular units).
    """
    omega = omega_from_lam(p["lam"])
    V = mode_volume_for(p)
    return hbar * omega**2 * c * p["n2"] / (4.0 * p["n"]**2 * V)

def bss_shift(g3, omega, kappa):
    """Bloch-Siegert shift: delta_omega_BSS ≈ (g3)^2 kappa / (8 omega^2)."""
    return g3**2 * kappa / (8.0 * omega**2)

def mandel_Q_laser(kappa, Nsp, g_minus_kappa):
    """Mandel Q = kappa Nsp / (2 (g - kappa))."""
    return kappa * Nsp / (2.0 * g_minus_kappa)

def henry_linewidth_ST(kappa, Nsp, g_minus_kappa, alphaH):
    """Schawlow-Townes Henry linewidth: delta omega = kappa^2 Nsp (1+alphaH^2)/(pi (g-kappa))."""
    return kappa**2 * Nsp * (1.0 + alphaH**2) / (np.pi * g_minus_kappa)
