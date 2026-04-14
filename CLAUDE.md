# CLAUDE.md — QNM Keldysh Numerics Project

## Project Identity

**Paper:** "Keldysh Path Integral Quantization of Nonlinear Open Nanophotonic Structures via Quasi-Normal Modes"
**Manuscript:** `qnm_final.tex` — 85 pages, 6014 lines, 39 refs, self-contained
**Target journals:** Physical Review A (primary); PRX Quantum (after Tier 1 numerics)
**Server:** 388 cores (4 nodes × 97 cores), spec in §PRIMARY COMPUTATIONAL TARGET

---

## Manuscript Status (April 2026)

All 26 errors corrected (E1–E26). Three consecutive clean audit rounds (16–18).
Expert reviews passed: quantum optics (8 checks) and condensed-matter/Keldysh (13 checks).

**Ready for arXiv and PRA submission NOW.**
**PRX Quantum requires Tier 1 numerics first** — see §Submission Readiness.

---

## Critical Physics Corrections (E1–E26)

### ⚠️ C1 / E11: Im(g³_cross) = 0 for passive Kleinman χ³ [CRITICAL]

**Old (wrong) claim:** Im(g³_{λμμλ})/Re ~ η_{λμ}/(2Q). This predicted cross-mode
dephasing δκ_λ ≠ 0 in passive cavities.

**Correct result:** Im(g³_{λμμλ}) = 0 **exactly** for passive Kleinman χ³.
Proved algebraically (I* = I) and numerically to |Im/Re| < 1.92×10⁻¹⁶ (2000 mode pairs).

**Physical consequence:** Cross-mode dephasing is identically zero in all passive
photonic structures, regardless of mode-overlap or cavity Q. It only arises from
gain saturation (Im χ³_sat ≠ 0) in active structures.

**Contrast with χ²:** g²_{λμν} ∈ ℂ with Im/Re ~ (1/Q_λ+1/Q_μ+1/Q_ν)/2 even
for passive media. Parametric processes do not conserve photon-number parity,
breaking the time-reversal symmetry that forces g³_cross ∈ ℝ.

**Current framing in paper:** "QNM-Hamiltonian consistency check" — the Keldysh
bilinear framework agrees with Hermitian QM for Kerr, but produces genuinely
complex couplings for parametric processes. This is the defensible referee framing.
Do **not** revert to "we prove g³_cross ∈ ℝ" (sounds trivial).

### ⚠️ E23: sign(O_ll) = −sign(g−κ)

Above threshold (g > κ): O_ll < 0. Below threshold: O_ll > 0.
Proof requires Im(ε_eff) > 0 for gain medium. Earlier version had both a wrong
sign and a wrong intermediate step that cancelled — both fixed.

### ⚠️ E25: BSS formula δω_BSS ≈ g³²κ/(8ω₀²)

Old (wrong): g³²/(4ω₀) — inflated by factor 2Q ~ 60,000 for GaAs.
Correct: g³²κ/(8ω₀²). Corrected BSS values:

| Platform | g³/2π | κ/2π | BSS/2π |
|---|---|---|---|
| Si microring | 294 Hz | 194 MHz | ~6×10⁻¹⁷ Hz |
| GaAs H1 PhC | 235 kHz | 10 GHz | ~8×10⁻¹⁰ Hz |
| LiNbO₃ WGM | 0.1 Hz | 2 Hz | ~10⁻³⁰ Hz |
| Circuit QED | 10–100 MHz | 0.5 MHz | ~0.02–2 Hz |

RWA conclusion unchanged (BSS << κ for all optical platforms).

### ⚠️ E26: CK sensing Mechanism 1 is ACTIVE structures only

Mechanism 1 uses S_CK = 4·Im(g³_sat,λμμλ), not Im(g³_cross) which is zero.
Subsubsection title now reads "(active structures)". Do not remove this qualifier.

### Other corrections (complete list)

| # | Location | Correction |
|---|---|---|
| E1 | Eq. modal_gain | Removed duplicate d³r |
| E2 | Eq. ep_time | G^R_EP(t) = −iθ(t)e^{−iω̃t}(I−itM) [sign fixed] |
| E3 | Eq. variance_gaussian | Constant −n̄/2 → +\|α₀\|²+N' |
| E4 | Eq. N_fluct | Factor-2 arithmetic fixed |
| E5 | Eq. Mandel_Q_laser | Q = κN_sp/(2(g−κ)); spurious −1 removed |
| E6 | Eq. g2_combined | Spurious −1/n̄ and wrong coefficient removed |
| E7 | Eq. pt_transfer | Sign: J²−g²/4 → g²/4−J² |
| E8 | EP D^K | Consistent gain notation |
| E9–E10 | Eq. imcrosskerr | Duplicate d³r; missing factor 4 |
| E12 | g3_Kerr annotation | "complex cross" → "real (Kleinman)" |
| E13 | Eq. chi3_sat | Removed spurious i in numerator: A/(Δ−iγ) |
| E14 | Eq. chi3_detuned | Re and Im were swapped |
| E15 | Eq. alpha_H | α_H = Re(g³)/Im(g³) = Δ/γ_⊥ [was inverted] |
| E16 | Eq. photon_number_K | n̄ = (+iG^K(0)−1)/2 [sign fixed] |
| E17 | chi2_complex section | χ² coupling IS complex even for passive media |
| E18 | Eq. g2_tau_result | Decay rate (g−κ); coeff Q/n̄ [were both wrong] |
| E19 | Sensing table | "χ² passive dephasing" [was "cross-Kerr passive"] |
| E20 | App. A header | 1/Q scaling for χ², not g³_cross |
| E21 | BSS table | All values updated to corrected g³ and BSS formula |
| E22 | Abstract BSS | Updated scaling statement |
| E24 | Eq. comb_squeezing text | S→−1 at threshold (not S→0) |

---

## Connection to Paper Equations — CORRECTED

| Code object | Paper equation | Notes |
|---|---|---|
| `omega_tilde` | ω̃_λ = ω_λ − iκ_λ/2, Sec 3 | |
| `bilinear_norm` | B_λλ = 1, Eq. bilinear_norm | normalization convention |
| `hermitian_O` | O_λλ = ∫ε\|f̃\|²d³r, Eq. hermitian_overlap | |
| `compute_g3_self` | g³_λλλλ ∈ ℝ⁺, Sec 8.2 | integrand = \|f̃\|⁴, trivially real |
| `compute_g3_cross` | g³_λμμλ ∈ ℝ, Sec 8.3 | **real for passive Kleinman — was listed as complex** |
| `compute_g3_sat_cross` | g³_sat,λμμλ ∈ iℝ, Sec 13 | purely imaginary for resonant gain |
| `compute_g2_complex` | g²_λμν ∈ ℂ, Sec 8.4 | Im/Re ~ (1/Q_λ+1/Q_μ+1/Q_ν)/2 |
| `ratio_im_re_chi2` | Im(g²)/Re(g²) ~ 1/Q | measurable via two-tone spectroscopy |
| `ratio_im_re_chi3` | Im(g³_cross) = 0 exactly | **was wrongly listed as ~ 1/Q** |
| `fig3_scaling.py` | Figure 3: Im(g²)/Re vs 1/Q | χ² scaling (not χ³) |
| `fig4_dephasing.py` | Figure 4: δκ vs n̄_μ | active structures only |

---

## Key Equations (verified correct)

```
Mandel Q         = κ·N_sp / [2·(g−κ)]             Gardiner-Zoller (10.4.11)
g^(2)(0)         = 1 + Q/n̄
g^(2)(τ)         = 1 + (Q/n̄)·exp(−(g−κ)|τ|)
N_fluct          = κ·N_sp / [4·(g−κ)]
Var(δn)          = 2·n̄·N_fluct + n̄
ST linewidth      = κ²·N_sp·(1+α_H²) / [π·(g−κ)]
Henry factor      α_H = Re(g³)/Im(g³) = Δ/γ_⊥
EP linewidth      = K₀·κ₀²·N_sp / [π·(g₀−κ₀)]  → ∞ at EP
BSS shift         ≈ (g³)²·κ / (8·ω₀²)            high-Q limit
Comb threshold    |g³_FWM|·|α|² > (κ₊₁+κ₋₁)/4
Squeezing at threshold: S^sq → −1   (NOT S→0; E24 fix)
sign(O_ll)        = −sign(g−κ):  O>0 below threshold, O<0 above
χ³ cross-Kerr     Im(g³_λμμλ) = 0 exactly  (passive Kleinman)
χ² coupling       g²_λμν ∈ ℂ;  Im/Re ~ (1/Q_λ+1/Q_μ+1/Q_ν)/2
χ³_sat coupling   g³_sat,λμμλ ∈ iℝ  (active, resonant gain)
D^K_total         = −iκ(2n+1) + ig(2N_sp+1)  [OPPOSITE SIGNS]
G^R_EP(t)         = −iθ(t)·e^{−iω̃t}·(I − it·M)
K_Petermann       = O_ll / |B_ll|²,  K→∞ at EP,  K≥1 always
```

---

## g^(2)(0) Combined Formula — CORRECTED

```
g^(2)(0) = 1
  + Q_laser / n̄                    [spontaneous emission; Q = κN_sp/(2(g−κ))]
  − 2·g³_R / κ_eff                 [self-Kerr squeezing]
  − |F_si|² / n̄²                   [χ² parametric squeezing]
```

Old CLAUDE.md had wrong Mandel Q coefficient and wrong Kerr coefficient. Use formula above.

---

## Submission Readiness

### Physical Review A — READY NOW ✓

Recommended as primary submission target. Accepts 85-page theory papers.
No figures beyond schematics required for analytical framework papers.

### PRX Quantum — Requires Tier 1 numerics first

Minimum additions needed before PRX Quantum submission:
1. **Tier 1 FDTD**: Im(g²_cross)/Re for GaAs H1 PhC (not g³_cross — that's zero)
2. Lau-Clerk comparison ✓ (already added, EP sensing Remark)
3. Franke comparison table ✓ (already updated, three-result itemization)
4. Length: cut to ~25 pages; move tutorial + appendices to supplemental

---

## PRIMARY COMPUTATIONAL TARGET — UPDATED

### What changed

OLD target: compute Im(g³_cross)/Re ~ η/(2Q) for χ³.
**This is analytically proved zero. FDTD not needed for χ³ cross-Kerr.**

NEW primary target: **compute Im(g²_λμν)/Re for GaAs H1 PhC χ² coupling**,
validating prediction Im/Re ~ (1/Q_λ + 1/Q_μ + 1/Q_ν)/2.

Secondary target: compute g³_sat cross-coupling (purely imaginary at resonance)
to validate active-structure cross-mode dephasing prediction.

### Structure

```yaml
material: GaAs
wavelength_nm: 1000.0
lattice_constant_nm: 240.0
hole_radius_ratio: 0.30
slab_thickness_nm: 200.0
n_GaAs: 3.5
chi2_GaAs_pm_per_V: 200.0     # d_14 ~ 200 pm/V
chi3_GaAs_m2_per_W: 1.5e-17
num_unit_cells: 10
target_modes: ["dipole_x", "dipole_y"]
target_Q_values: [1e3, 3e3, 1e4, 3e4, 1e5]
```

### Step 1: QNM fields (Meep + Harminv, unchanged)

```python
import meep as mp, numpy as np, h5py

def compute_qnm_fields(params, mode_index=0, resolution=40):
    geometry = build_h1_phc(params)
    src = mp.Source(mp.GaussianSource(frequency=params['omega0']/(2*np.pi), fwidth=0.1),
                    component=mp.Ex, center=mp.Vector3(0,0,0))
    sim = mp.Simulation(cell_size=cell_size, geometry=geometry, sources=[src],
                        resolution=resolution, boundary_layers=[mp.PML(params['pml_t'])])
    sim.run(mp.after_sources(mp.Harminv(mp.Ex, mp.Vector3(0,0,0),
                                         params['omega0']/(2*np.pi), 0.3)),
            until_after_sources=500)
    omega_tilde = select_target_mode(sim.harminv_results, mode_index)
    fields = extract_complex_fields(sim, omega_tilde)
    normalize_bilinear(fields, params)  # enforce B_ll = 1
    return omega_tilde, fields
```

### Step 2: χ² overlap integral (primary new target)

```python
def compute_g2_complex(f_lambda, f_mu, f_nu, chi2_tensor, dV):
    """g²_λμν = integral χ²_ijk f*_λi f_μj f_νk d³r"""
    integrand = np.einsum('...ijk,...i,...j,...k->...',
                           chi2_tensor,
                           f_lambda.conj(),   # creation: f*_λ
                           f_mu,              # annihilation: f_μ
                           f_nu)              # annihilation: f_ν
    return np.sum(integrand * dV)
    # Expected: complex, Im/Re ~ 1/Q ~ 3e-5 for Q=3e4

def validate_chi2_prediction(Q, g2):
    predicted = 1.5 / Q  # (1/Q_l + 1/Q_m + 1/Q_n)/2 for degenerate triplet
    print(f"Q={Q:.0e}: Im/Re={abs(g2.imag/g2.real):.2e}, predicted={predicted:.2e}")
```

### Step 3: χ³ sanity checks (passive cavity)

```python
def sanity_check_passive(f_lambda, f_mu, chi3, dV):
    """For passive Kleinman cavity:
       Im(g³_self)/Re  <  1e-8   [manifestly real]
       Im(g³_cross)/Re < 1e-10   [Kleinman theorem; any larger = numerical noise]
    """
    g3_self  = compute_g3_self(f_lambda, chi3, dV)
    g3_cross = compute_g3_cross(f_lambda, f_mu, chi3, dV)
    assert abs(g3_self.imag / g3_self.real)  < 1e-8,  "Self-Kerr Im/Re too large"
    assert abs(g3_cross.imag / g3_cross.real) < 1e-10, "Cross-Kerr Im/Re nonzero (numerical artifact)"
    return g3_self, g3_cross
```

### Numerical pitfalls

1. **χ³ cross-Kerr Im/Re should be numerically zero** (< 10⁻¹⁰ for passive Kleinman).
   If you see Im/Re ~ 1/Q for g³_cross, it is a **numerical artifact**, not physics.

2. **χ² Im/Re should be ~ 1/Q** (~3×10⁻⁵ for Q=3×10⁴). This is the real signal.
   Resolving it to 10% requires Im(f̃) accurate to 0.003%. Use resolution ≥ 40.

3. **PML convergence critical.** PML thickness ≥ 1 lattice constant for bilinear integral.

4. **Resolution convergence for Im part.** Compare resolutions 20, 30, 40, 50.

5. **Dipole degeneracy.** Use separate Ex and Ey sources. Slight numerical splitting OK.

---

## Quick-Start Commands

```bash
# Analytic figures (Tiers 2–3, no server)
python analytic_figures/fig_g2_vs_pump.py
python analytic_figures/fig_Pn_distributions.py
python analytic_figures/fig_comb_squeezing.py      # S→−1 at threshold (not S→0)
python analytic_figures/fig_petermann_K.py
python analytic_figures/fig_henry_qcl.py
python analytic_figures/fig_active_qfi.py
python analytic_figures/fig_st_henry.py

# Small FDTD test (4 cores, low resolution)
mpirun -n 4 python qnm_solver/compute_qnm.py \
    --resolution 20 --target-Q 1e4 --output data/test_run.h5
python tests/test_normalization.py data/test_run.h5

# Chi2 overlap (after FDTD)
python overlap/compute_g2_complex.py \
    --mode1 data/qnm_fields/dipole_x_Q30000.h5 \
    --mode2 data/qnm_fields/dipole_y_Q30000.h5 \
    --pump  data/qnm_fields/pump_Q30000.h5 \
    --output data/g2_results/g2_Q30000.csv

# Chi3 sanity checks
python overlap/compute_g3_checks.py data/qnm_fields/dipole_x_Q30000.h5
# Expected: Im(g3_cross)/Re < 1e-10 for passive

# Full Q sweep (cluster, Tier 1)
sbatch slurm/submit_chi2_sweep.sh
```

---

## Analytic Figure Specs (Tiers 2–3)

### fig_g2_vs_pump.py
```
x: (g−κ)/κ from 0.01 to 9
y: g^(2)(0) = 1 + Q/n̄  where Q=κN_sp/(2(g−κ)), n̄=(g−κ)/(2g3_sat)
Curves: gain only; +χ³ (g³_R/κ = 0.1, 0.5); +χ² (r = 0.5, 0.9)
Mark: g^(2)=2 threshold; g^(2)=1 coherent; g^(2)<1 sub-Poissonian
```

### fig_Pn_distributions.py
```
Panels: thermal | Poissonian | squeezed laser | photon blockade (g³_R/κ=0.1,1,10)
x: n from 0 to 4n̄, all panels n̄=10
```

### fig_petermann_K.py
```
x: ε = J−g/2 from 0.001 to 1
Left: K_λ(ε) — verify K ~ 1/ε
Right: SNR_EP(δ,ε) for 3 values of δ — verify SNR independent of ε (Lau-Clerk)
```

### fig_henry_qcl.py
```
x: mode index m (−50 to +50), FSR=10 GHz, γ_⊥=1 THz
α_H(m) = Δ_m/γ_⊥; comb threshold vs m; Re/Im(g³_sat) vs m
```

### fig_comb_squeezing.py
```
S^sq(ω) = 1 − 4|g³_FWM|²|α|² / (|ω−ω_m|² + |g³_FWM·α|²)
x: (ω−ω_m)/κ_m from −3 to +3; curves r=0.3,0.6,0.9,0.99
Note: S→−1 at threshold r→1 (maximum squeezing; E24 fix — not S→0)
```

### fig_active_qfi.py
```
x: κ_eff from 0.01κ to κ
Show: frequency term ~ 1/κ_eff² (passive+active); linewidth term ~ 1/κ_eff² (active only)
Linewidth sensing dominates near threshold
```

---

## Software Dependencies

```bash
conda create -n qnm-numerics python=3.11
conda activate qnm-numerics
conda install -c conda-forge meep mpi4py h5py
conda install numpy scipy matplotlib
pip install tqdm pyyaml
```

---

## Key References Added (April 2026)

| Key | Reference | Location in paper |
|---|---|---|
| Lau2018 | Lau & Clerk, Nat. Commun. 9, 4320 (2018) | EP sensing paradox Remark |
| Schawlow1958 | Schawlow & Townes, Phys. Rev. 112, 1940 (1958) | ST linewidth |
| Lax1967 | Lax, IEEE J. QE 3, 37 (1967) | ST linewidth |
| GardinerZoller2004 | Gardiner & Zoller, Quantum Noise 3rd ed. (2004) | Mandel Q |
| Ozdemir2019 | Özdemiret al., Nat. Mater. 18, 783 (2019) | Jordan chain |
| Heiss2012 | Heiss, J. Phys. A 45, 444016 (2012) | Jordan chain |

Total bibliography: 39 entries, 0 missing, 0 uncited.
