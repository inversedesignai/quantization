"""
Synthetic QNM mode function constructor.

Theorem under test (App. E / Eq. highQ_expansion):
    f_tilde_lambda(r) = f_lambda^(0)(r) [ 1 + i/Q_lambda * phi_lambda(r) ]
                        + O(Q^-2)
where f^(0) is real and phi is real.

We build such fields on a 3D grid, with random real zeroth-order mode functions
and random real phase correction phi, then compute:
    g^(3)_{lambda mu mu lambda}  (Kleinman chi^3)
    g^(2)_{lambda mu nu}         (Kleinman chi^2, GaAs-like)
and check the ratio Im/Re vs 1/Q over a sweep of Q values.
"""
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray


def random_real_mode(shape: tuple[int,int,int], rng: np.random.Generator,
                     n_cells: int = 3) -> NDArray[np.float64]:
    """Random real scalar (to be multiplied by a polarisation vector).

    Built as a Fourier sum of modes with random real coefficients.
    Component-wise real => represents the closed-cavity standing-wave limit.
    """
    Nx, Ny, Nz = shape
    f = np.zeros(shape, dtype=np.float64)
    x = np.linspace(0, 1, Nx, endpoint=False)
    y = np.linspace(0, 1, Ny, endpoint=False)
    z = np.linspace(0, 1, Nz, endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    for kx in range(1, n_cells+1):
        for ky in range(1, n_cells+1):
            for kz in range(1, n_cells+1):
                cx = rng.standard_normal()
                cy = rng.standard_normal()
                cz = rng.standard_normal()
                # standing-wave basis: sin(k pi x) which vanishes on cavity boundary
                f += (cx*np.sin(kx*np.pi*X) *
                      np.sin(ky*np.pi*Y) *
                      np.sin(kz*np.pi*Z)) + \
                     (cy*np.cos(kx*np.pi*X) *
                      np.sin(ky*np.pi*Y) *
                      np.sin(kz*np.pi*Z)) * 0.5
                del cy, cz
    return f


def random_real_phase(shape: tuple[int,int,int], rng: np.random.Generator,
                      n_cells: int = 3, scale: float = 1.0
                      ) -> NDArray[np.float64]:
    """Random real-valued phase correction phi(r).  Smooth, not singular.

    The 1D Fabry-Perot result (Eq. FP_phase) gives phi_m(x)=-(m*pi*x/(2L)) cot(m*pi*x/L)
    which is O(1) but non-zero-mean.  For synthetic tests we use a generic
    random smooth real field with zero mean plus a non-zero-mean term to allow
    a nonzero eta.
    """
    x = np.linspace(0, 1, shape[0], endpoint=False)
    y = np.linspace(0, 1, shape[1], endpoint=False)
    z = np.linspace(0, 1, shape[2], endpoint=False)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    phi = 0.5 * rng.standard_normal()  # constant part
    for kx in range(n_cells+1):
        for ky in range(n_cells+1):
            for kz in range(n_cells+1):
                if kx==0 and ky==0 and kz==0: continue
                a = rng.standard_normal()
                b = rng.standard_normal()
                phi += a*np.cos(kx*np.pi*X)*np.cos(ky*np.pi*Y)*np.cos(kz*np.pi*Z)
                phi += b*np.sin(kx*np.pi*X)*np.sin(ky*np.pi*Y)*np.sin(kz*np.pi*Z)*0.3
    return scale * phi


def build_vector_qnm(shape, rng, Q, n_cells=3):
    """Construct a 3D vector QNM mode function f_tilde.

    Returns: f_tilde   (3, Nx, Ny, Nz) complex,  with f^(0) real and the
             exact structure f^(0)(1 + i phi/Q).
    """
    f0 = np.empty((3,)+shape, dtype=np.float64)
    phi = np.empty((3,)+shape, dtype=np.float64)
    for i in range(3):
        f0[i]  = random_real_mode(shape, rng, n_cells=n_cells)
        phi[i] = random_real_phase(shape, rng, n_cells=n_cells)
    ft = f0.astype(np.complex128) * (1.0 + 1j/Q * phi)
    return ft, f0, phi


def bilinear_norm(f_tilde, eps=1.0, dV=1.0):
    """B_ll = integral eps * f_tilde . f_tilde  d^3r   (NOTE: f.f not |f|^2)."""
    # sum over vector component i, then over space
    integrand = eps * np.einsum('ixyz,ixyz->xyz', f_tilde, f_tilde)
    return integrand.sum() * dV


def hermitian_overlap(f_tilde, eps=1.0, dV=1.0):
    """O_ll = integral eps |f_tilde|^2 d^3r."""
    integrand = eps * np.einsum('ixyz,ixyz->xyz', f_tilde.conj(), f_tilde)
    return integrand.sum().real * dV


# Isotropic Kleinman chi^(3): chi^3_{ijkl} = chi3 (d_ij d_kl + d_ik d_jl + d_il d_jk).
def g3_cross_kleinman(fA, fB, chi3, dV=1.0):
    """g^(3)_{lambda mu mu lambda} for two distinct modes A, B.

    g3 ~ (fA*)_i (fB*)_j (fB)_k (fA)_l * chi3_{ijkl}
       = chi3 [ (fA*.fA)(fB*.fB) + (fA*.fB*)(fB.fA) + (fA*.fB)(fB*.fA) ]
    """
    dot_AcAc = np.einsum('ixyz,ixyz->xyz', fA.conj(), fA.conj())       # (fA*.fA*)
    dot_AcA  = np.einsum('ixyz,ixyz->xyz', fA.conj(), fA)               # |fA|^2
    dot_BcB  = np.einsum('ixyz,ixyz->xyz', fB.conj(), fB)               # |fB|^2
    dot_AcBc = np.einsum('ixyz,ixyz->xyz', fA.conj(), fB.conj())        # fA*.fB*
    dot_BA   = np.einsum('ixyz,ixyz->xyz', fB, fA)                      # fB.fA
    dot_AcB  = np.einsum('ixyz,ixyz->xyz', fA.conj(), fB)               # fA*.fB
    dot_BcA  = np.einsum('ixyz,ixyz->xyz', fB.conj(), fA)               # fB*.fA

    integrand = chi3 * (dot_AcA*dot_BcB + dot_AcBc*dot_BA + dot_AcB*dot_BcA)
    return integrand.sum() * dV


def g3_self_kleinman(fA, chi3, dV=1.0):
    """Single-mode self-Kerr (must be real, positive)."""
    mod2 = np.einsum('ixyz,ixyz->xyz', fA.conj(), fA).real
    # chi^(3) Kleinman-isotropic: sum over i j k l chi^(3)[dij dkl + dik djl + dil djk]
    # fA*_i fA*_j fA_k fA_l => three contractions all reduce to |fA|^4.
    integrand = 3.0 * chi3 * mod2**2
    return integrand.sum() * dV


# Kleinman chi^(2) for T_d (GaAs-like): only nonzero chi^(2)_{xyz} and all perms, equal.
def g2_kleinman_Td(fA, fB, fC, chi2, dV=1.0):
    """g^(2)_{lambda mu nu} with full Kleinman-symmetric Td chi^(2) (single d14 coeff).

    chi2_{ijk} is real and fully symmetric; nonzero only when {i,j,k}={x,y,z}.
    6 distinct permutations of (i,j,k).

    The coupling integrand is chi2_{ijk} fA*_i fB_j fC_k.
    """
    # sum over 6 permutations of (x,y,z)
    perms = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
    integrand = np.zeros(fA.shape[1:], dtype=np.complex128)
    for (i,j,k) in perms:
        integrand += fA[i].conj() * fB[j] * fC[k]
    integrand *= chi2
    return integrand.sum() * dV


def g2_kleinman_isotropic(fA, fB, fC, chi2, dV=1.0):
    """Isotropic-like Kleinman chi^(2) for generic test.

    chi^(2)_{ijk} = chi2 (d_ij d_jk + ... ) -- we take the fully symmetric
    isotropic form chi^(2)_{ijk} = (chi2/6) [d_ij v_k + d_ik v_j + d_jk v_i]
    with v=(1,1,1)/sqrt(3).  For testing analytical structure only.
    """
    v = np.array([1,1,1])/np.sqrt(3)
    integrand = np.zeros(fA.shape[1:], dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                tensor = 0.0
                if i==j: tensor += v[k]
                if i==k: tensor += v[j]
                if j==k: tensor += v[i]
                integrand += chi2*(tensor/6.0) * fA[i].conj() * fB[j] * fC[k]
    return integrand.sum() * dV
