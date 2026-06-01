# Hybrid CPA Amplifier — Complete Simulation Reference
## Physics, Architecture, Equations, and Debugging Guide

**Last updated:** 2026-05-21
**System:** Yb fiber pre-amp + free-space relay + Yb:YAG (or YLF/Nd:YAG) solid-state amplifier
**Backend:** Python/NumPy (CPU) + Taichi/CUDA (GPU hot path)

---

## Table of contents

1. System overview
2. Coordinate system and grids
3. Fiber pre-amplifier (Pass A/B)
4. Free-space beam propagation (ABCD)
5. Pump cavity dynamics
6. Solid-state amplifier — Pass A: pump inversion
7. Solid-state amplifier — Pass B: thermal solve
8. Solid-state amplifier — Pass C: signal BPM
9. Rate equations
10. Material parameters
11. Diagnostics — every variable to check
12. Bug history and fixes
13. Validation targets
14. References

---

## 1. System overview

```
FIBER PRE-AMPLIFIER
  Seed pulse (ns chirped burst)
       ↓
  Yb:glass fiber, cladding-pumped
  Two-pass solver (Pass A: pump; Pass B: signal)
  P_s(z, t, λ), N₀,N₂(z, t)
       ↓
FREE-SPACE OPTICS
  Collimator (f₁) → relay telescope (f₂:f₃) → beam expander
  ABCD Gaussian propagation, beam waist tracked at each element
       ↓
SOLID-STATE AMPLIFIER
  Pass A: Pump inversion build
    P_pump(x,y,z,t) → N₂(x,y,z,t) via QSS rate march
    Accumulate Q_heat(x,y,z) = absorbed pump × quantum defect
  Pass B: Thermal solve
    Q_heat → ΔT(x,y,z) via 2D FD heat equation
    dn_thermal(x,y,z) = (dn/dT) × ΔT
  Pass C: Signal BPM
    Causal (z, t) march:
      For iz: BPM angular-spectrum step → gain + Kerr + thermal phase → N₂ RK4 update
  Output: U(x,y), M², B-integral, f_thermal, energetics
       ↓
OUTPUT PULSE
  Characterized by: E_signal, spectrum, M², wavefront, B
```

---

## 2. Coordinate system and grids

### Spatial grids

```
z[0…n_z-1]       uniform, Δz = L_crystal / (n_z - 1)
x[0…n_x-1]       uniform, Δx = aperture_x / n_x   (centered on beam axis)
y[0…n_y-1]       uniform, Δy = aperture_y / n_y
```

Recommended aperture: 4 × beam_waist_1e2. Padding needed for BPM anti-aliasing:
at least 2× beam diameter before edge. Use `np.fft.fftfreq` for kx, ky grids.

### Time grid (non-uniform, same as fiber code)

```
t_coarse:  linspace(0, T_pump) with ~500 points for pump dynamics
t_fine:    linspace(burst_start - 5ns, burst_end + 5ns) with ~1000 points for signal
t_total:   np.concatenate([t_coarse, t_fine]) sorted + deduplicated
```

Signal threshold for "t has signal": P_s(t) > 1e-6 × P_s_peak

### Wavelength grid

```
λ[0…n_λ-1]  linspace(λ_min, λ_max) nm
             Recommended: 1010–1050 nm for Yb:YAG, 0.5 nm/point minimum
```

### Transit time

```
Δt_travel = Δz / v_g
v_g = C0 / n_group
```

For Yb:YAG: n_group = 1.82 → v_g ≈ 1.648 × 10⁸ m/s

---

## 3. Fiber pre-amplifier

Reference: `laser_sim/physics/fiber_cpa.py`, `SIMULATION_REFERENCE.txt`.

### 3.1 Geometry

```
Core diameter:    d_core (µm), A_core = π (d_core/2)² [m²]
Cladding diameter: d_clad (µm), A_pump = π (d_clad/2)²
Overlap factor:   Γ_p = (r_core/r_clad)²  [cladding-pumped]
                  Γ_s = 1.0  [signal fills core]
```

### 3.2 Dopant concentration from measured κ (dB/m)

```
α_np = κ [dB/m] / (10 × log₁₀ e) = κ / 4.3429…  [Np/m]
N_tot = α_np / (Γ_p × σ_abs(λ_pump))

Note: measured fiber κ already includes Γ_p. Do NOT apply Γ_p twice.
Small-signal check: when N₀ ≈ N_tot, α_p ≈ σ_p × N_tot = α_np  ✓
```

### 3.3 Pass A — pump march (time, fixed z)

For each z-slab iz, march forward in t using QSS on fast levels (τ₃₂ ≈ 1 ps ≪ Δt):

```
W_p,abs = Γ_p × σ_abs(λ_p) × I_p / (h × ν_p)
W_p,esa = Γ_p × σ_esa(λ_p) × I_p / (h × ν_p)

N₃ ≈ W_p,abs × N₀ × τ₃₂ / (1 + W_p,esa × τ₃₂)   [QSS]

N₂_ss = W_p,abs × N₀ × τ₂₁ / (1 + W_p,abs × τ₂₁)  ← SELF-CONSISTENT (see §12)

N₂(t + Δt) = N₂_ss - (N₂_ss - N₂(t)) × exp(-Δt / τ₂₁)

N₀ = N_tot - N₂ - N₃
```

Pump z-propagation (after time march at iz):
```
α_p(t) = σ_abs(λ_p) × N₀(iz, t)          [not datasheet κ]
P_p(iz+1, t) = P_p(iz, t) × exp(-α_p(t) × Δz)
```

### 3.4 Pass B — signal amplification (z-march, causal t)

For each iz, for each t (causal order):
```
g(t, λ) = Γ_s × [σ_e(λ) × N₂(iz, t) - σ_a(λ) × N₀(iz, t)]   [m⁻¹]
P_s(iz+1, t, λ) = P_s(iz, t, λ) × exp(g × Δz)

Then RK4 update N₀, N₂ over Δt_travel with signal-stimulated rates:
  W_se = Γ_s × Σ_λ σ_e(λ) × I_s(λ) / (h × ν(λ))
  W_abs = Γ_s × Σ_λ σ_a(λ) × I_s(λ) / (h × ν(λ))
```

Note: RK4 evolves only N₀, N₂. N₁ = 0 (τ₁₀ → 0). N₃ = 0 (signal pass).

---

## 4. Free-space beam propagation (ABCD)

Gaussian beam q-parameter through each optical element:

```
q = z - z_waist + i × z_R
z_R = π × w₀² × n / λ

Free space (distance d):       q' = q + d
Thin lens (focal length f):   q' = q / (1 - q/f)

Beam waist from q:
  w(z) = sqrt(-λ / (π × Im(1/q)))
```

### Elements and their ABCD matrices

| Element | M |
|---------|---|
| Free space, d | [[1,d],[0,1]] |
| Thin lens, f | [[1,0],[-1/f,1]] |
| Crystal, n_in, n_out | refraction at interface: [[1,0],[0,n_in/n_out]] |

Compute beam radius at collimator output, relay output, crystal face for
diagram annotation. Verify: beam fills crystal aperture without hitting edges.

### Diffraction loss (spatial filter)

If a spatial filter (pinhole) is included:
```
η_pinhole = 1 - exp(-2 × (r_pinhole/w_filter)²)   [fraction transmitted]
```

---

## 5. Pump cavity dynamics

### 5.1 Round-trip model

State variables at round-trip k:
```
E_intra(k)     intra-cavity energy [J]
n₂(k)          upper-state fraction (population inversion)
```

Round-trip time: `T_rt = 2 × L_cav / C0`

Round-trip gain:
```
g₀(k) = Γ_s × (σ_e × n₂(k) - σ_a × (1 - n₂(k))) × N_tot
G_rt(k) = exp(2 × g₀(k) × L_crystal)
```

Round-trip loss:
```
loss(k) = (1 - R_OC(k)) × (1 - R_HR) × (1 + L_int)²
```

where `R_OC(k)` = `R_OC` when Q-switch open, `R_OC × (1 - η_holdoff)` when closed.

Energy evolution:
```
E_intra(k+1) = E_intra(k) × G_rt(k) × (1 - loss(k)) + E_seed
E_out(k) = E_intra(k) × (1 - R_OC(k))
```

Inversion depletion by lasing:
```
ΔN₂ = (G_rt - 1) × E_intra / (h × ν × N_tot × V_crystal)   [fraction]
n₂(k+1) = max(0, n₂_from_pump(k) - ΔN₂)
```

Pump inversion between round-trips (same QSS formula):
```
n₂_ss = W_p × τ₂₁ / (1 + W_p × τ₂₁)
n₂(t + T_rt) = n₂_ss - (n₂_ss - n₂(t)) × exp(-T_rt / τ₂₁)
```

### 5.2 Lasing threshold condition

```
G_rt × (1 - loss) > 1
→ 2 × g₀ × L_crystal > -log(1 - loss)
```

Threshold inversion fraction:
```
n₂_threshold = [ln(1/((1-loss_threshold))) / (2 × L_crystal × N_tot) + σ_a] /
               (σ_e + σ_a)
```

### 5.3 Q-switch pulse energy (Frantz-Nodvik estimate)

```
E_sat = h × ν × A_mode / (σ_e + σ_a)  [J, saturation energy]
E_pulse,FN = E_sat × ln(1 + G₀ × (exp(E_in/E_sat) - 1))
```

For Q-switch: E_in = spontaneous seed, G₀ = exp(2 × g₀ × L). Compare
`run_cavity_simulation` output to FN estimate — must agree within 20%.

---

## 6. Solid-state — Pass A: pump inversion

### 6.1 2D pump beam profile

Gaussian pump at crystal face:
```
I_pump(x, y, t) = P_pump(t) / (π × w_p²) × exp(-2(x² + y²) / w_p²)
```

Pump z-propagation (2D Beer-Lambert):
```
α_p(x, y, iz, t) = σ_abs(λ_p) × N₀(x, y, iz, t)
P_pump(x, y, iz+1, t) = P_pump(x, y, iz, t) × exp(-α_p × Δz)
```

### 6.2 Heat source

Quantum defect heating fraction:
```
η_heat = 1 - λ_pump / λ_signal
Yb:YAG 940→1030 nm: η_heat = 0.0874 (8.74%)
```

Heat deposited per unit volume per unit time:
```
Q(x, y, iz, t) = α_p(x, y, iz, t) × I_pump(x, y, iz, t) × η_heat   [W/m³]
```

Accumulated heat per z-slab over pump duration T_pump:
```
Q_total(x, y, iz) = ∫₀^T_pump Q(x, y, iz, t) dt
```

---

## 7. Solid-state — Pass B: thermal solve

### 7.1 2D heat equation

```
∂T/∂t = (κ_T / (ρ × Cp)) × (∂²T/∂x² + ∂²T/∂y²) + Q(x,y,z,t) / (ρ × Cp)
```

Boundary conditions: Dirichlet T = T_ambient at crystal edge (coolant contact).

Finite-difference explicit scheme (stability: Δt < Δx² × ρCp / (2κ_T)):
```
T[i,j]^(n+1) = T[i,j]^n + Δt × {
    (κ_T/(ρCp)) × (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1] - 4T[i,j]) / Δx²
    + Q[i,j] / (ρCp)
}
```

For µs pump pulses: use quasi-steady approximation (ΔT ~ Q × τ_pump / (ρCp)).
Full time-marching needed only for ms+ pump durations.

### 7.2 Thermal lens focal length

Refractive index change:
```
Δn(x, y, z) = (dn/dT) × ΔT(x, y, z)
```

For a parabolic ΔT profile (Gaussian pump, steady-state):
```
ΔT(r) ≈ ΔT₀ × (1 - 2r²/w_p²)
Δn(r) ≈ (dn/dT) × ΔT₀ × (1 - 2r²/w_p²)
```

Equivalent thin lens from accumulated phase:
```
ΔΦ(r) = k₀ × Δn(r) × L_crystal
f_thermal = w_p² / (2 × L_crystal × (dn/dT) × ΔT₀)
```

Paraxial validity check: `f_thermal >> L_crystal`

Material values:
```
Yb:YAG:   dn/dT = +7.3×10⁻⁶ K⁻¹, κ_T = 6.8 W/(m·K)  → positive lens (focusing)
Yb:YLF:   dn/dT = -2.0×10⁻⁶ K⁻¹, κ_T = 6.0 W/(m·K)  → negative lens (defocusing)
Nd:YAG:   dn/dT = +7.3×10⁻⁶ K⁻¹, κ_T = 11.2 W/(m·K)
```

Yb:YLF thermal self-compensation: negative dn/dT partially cancels gain guiding.

---

## 8. Solid-state — Pass C: signal BPM

### 8.1 Angular spectrum propagator

For free-space step Δz:
```
k₀ = 2π n / λ
kx[m] = 2π × fftfreq(n_x, d=Δx)
ky[n] = 2π × fftfreq(n_y, d=Δy)
kz[m,n] = sqrt(max(k₀² - kx[m]² - ky[n]², 0))   # zero evanescent
H[m,n] = exp(1j × kz[m,n] × Δz)
```

Apply in Fourier domain:
```
U_k = fft2(U)
U_k_prop = U_k × H
U_prop = ifft2(U_k_prop)
```

Anti-aliasing: zero high-k components where kx² + ky² > k₀² (evanescent).

### 8.2 Phase and gain step

After free-space propagation:
```
g_amp(x, y) = 0.5 × Γ_s × [σ_e(λ) × N₂(x,y,iz) - σ_a(λ) × N₀(x,y,iz)] × Δz

Φ_kerr(x, y) = k₀ × n₂ × |U(x,y)|² × Δz
Φ_thermal(x, y) = k₀ × Δn_thermal(x,y,iz) × Δz
Φ_total = Φ_kerr + Φ_thermal

U_out = U_prop × exp(g_amp) × exp(1j × Φ_total)
```

For spectral signal, sum over wavelength channels:
```
For each λ: U_out(λ) = bpm_step(U_in(λ), H(λ), g(λ)/2, Φ(λ))
```

### 8.3 Causal population update

After advancing U at (iz, it), update N₀, N₂ at iz over dt = t[it+1] - t[it]:
```
W_se(x,y) = Γ_s × Σ_λ σ_e(λ) × |U(x,y,λ)|² / (A_core × h × ν(λ))
W_abs(x,y) = Γ_s × Σ_λ σ_a(λ) × |U(x,y,λ)|² / (A_core × h × ν(λ))

RK4 on (N₀, N₂) over dt:
  dN₂/dt = -(W_se × N₂ - W_abs × N₀) - N₂/τ₂₁
  dN₀/dt = +(W_se × N₂ - W_abs × N₀) + N₂/τ₂₁  [+ N₃/τ₃₂ ≈ 0 in signal pass]
```

Note: dt here is the actual time step t[it+1]-t[it], NOT Δt_travel = Δz/v_g.

### 8.4 B-integral

```
B(x, y) = k₀ × n₂ × Σ_iz |U(x, y, iz)|² × Δz

B_max = max over (x, y)  → must be < π for acceptable beam quality
Warning threshold: B > π/2
Critical threshold: B > π  → strong self-focusing, simulation invalid
```

### 8.5 Beam quality M²

From second-moment widths:
```
<x>  = Σ x  × |U|² / Σ |U|²
<x²> = Σ x² × |U|² / Σ |U|²
σ_x² = <x²> - <x>²

σ_x_f = beam width in far field (propagate U by z_R, then measure)

M²_x = 4π × σ_x × σ_x_f / λ
```

---

## 9. Rate equations (four_level.py)

### 9.1 Full 4-level system

```
N₀ + N₁ + N₂ + N₃ = N_tot

dN₃/dt = W_p,abs × N₀ - W_p,esa × N₃ - N₃/τ₃₂
dN₂/dt = N₃/τ₃₂ - W_se × N₂ + W_abs × N₁ - N₂/τ₂₁
dN₁/dt = W_se × N₂ + W_abs × N₁ + β × N₂/τ₂₁ - N₁/τ₁₀
dN₀/dt = -(pump source terms) + N₁/τ₁₀ + (1-β) × N₂/τ₂₁
```

### 9.2 Quasi-2-level approximation (τ₁₀ → 0, N₁ = 0)

```
N₀ + N₂ + N₃ = N_tot

Signal-pass gain:
  g(λ) = Γ_s × [σ_e(λ) × N₂ - σ_a(λ) × N₀]   [m⁻¹]
```

For Nd:YAG (true 4-level): N₁ ≠ 0, set `skip_n1_level = False`.

### 9.3 Stimulated emission rates

```
W_se = Γ_s × Σ_λ σ_e(λ) × I_s(λ) / (h × ν(λ))
W_abs = Γ_s × Σ_λ σ_a(λ) × I_s(λ) / (h × ν(λ))
I_s(λ) = P_s(λ) / A_core  [W/m²]
```

For 2D solid-state:
```
I_s(x, y, λ) = |U(x, y, λ)|² / (h × ν(λ))   [photon flux s⁻¹ m⁻²]
W_se(x, y) = Γ_s × Σ_λ σ_e(λ) × |U(x, y, λ)|²  / (h × ν(λ))
```

---

## 10. Material parameters

### 10.1 Cross-sections

**IMPORTANT:** Use measured tabulated cross-sections. Gaussian analytical models
cause subtle errors (see §12 bug history). Minimum validation:

```
σ_abs(λ_pump) > 1e-25 m²      (catch Gaussian underflow)
σ_e(λ_signal) > 1e-25 m²
σ_e/σ_abs ratio at λ_ZL ≈ 1   (McCumber consistency at ZPL)
```

| Material | λ_pump (nm) | σ_abs,pump (m²) | λ_signal (nm) | σ_em,sig (m²) | τ₂₁ (µs) |
|----------|------------|----------------|--------------|--------------|----------|
| Yb:glass | 976 | 2.4×10⁻²⁵ | 1030 | 3.6×10⁻²⁵ | 896 |
| Yb:YAG | 940 | 7.5×10⁻²⁵ | 1030 | 2.5×10⁻²⁴ | 950 |
| Yb:YLF | 960 | 1.8×10⁻²⁴ | 1020 | 2.2×10⁻²⁴ | 2000 |
| Nd:YAG | 808 | 6.5×10⁻²⁴ | 1064 | 2.8×10⁻²⁴ | 230 |

McCumber relation (cross-check):
```
σ_abs(λ) = σ_em(λ) × exp((h×ν(λ) - h×ν_ZL) / (k_B × T))
```

### 10.2 Crystal dopant concentration from at.%

```
Yb:YAG: ρ = 4550 kg/m³, M_YAG = 593.5 g/mol, M_Yb = 173.04 g/mol
N_Yb = (x_at_pct/100) × ρ × N_A / M_YAG × (M_YAG/M_unit)

For 1 at.%:
N_tot = 0.01 × 4550 × 6.022×10²³ / 0.5935 = 4.62×10²⁵ m⁻³
```

Note: SOLID_STATE_SIM_REFERENCE.md quotes 1.38×10²⁶ m⁻³ — verify against
your crystal supplier specification. Concentration varies with host formula.

### 10.3 Lifetimes

| Parameter | Yb:YAG | Yb:glass | Notes |
|-----------|--------|----------|-------|
| τ₂₁ (µs) | 950 | 896 | Radiative lifetime of ²F₅/₂ manifold |
| τ₃₂ (ps) | ~1 | ~1 | Pump manifold thermalization; QSS when Δt ≫ τ₃₂ |
| τ₁₀ (ns) | ~1 | ~1 | Lower Stark level; 0 for quasi-2L |
| β₂₁ | 1.0 | 1.0 | Branching ratio spont. emission to N₁ |

### 10.4 Saturation parameters

Signal saturation power:
```
P_sat = h × ν × A_signal / ((σ_e + σ_a) × τ₂₁)
```

Example — Yb:YAG, w_s = 0.5 mm beam (A = 7.85×10⁻⁷ m²):
```
P_sat = 1.93×10⁻¹⁹ × 7.85×10⁻⁷ / (2.5e-24 + 1.5e-26) / 950e-6
      ≈ 6.3 kW
```

A 1 mJ pulse in 1 ns = 1 MW = 159 × P_sat → deep saturation. Use Frantz-Nodvik.

Frantz-Nodvik amplification:
```
E_sat,FN = h × ν × A_signal / (σ_e + σ_a)   [saturation fluence × area]
E_out = E_sat,FN × ln(1 + G₀ × (exp(E_in/E_sat,FN) - 1))
G₀ = exp(g₀ × L_crystal)                    [small-signal gain]
```

---

## 11. Diagnostics — variables to check at each stage

### Fiber stage diagnostics

```
N2_mean(z, t)         — should approach W_p×τ/(1+W_p×τ) at late t
N0_mean(z, t)         — complement; should not reach 0 (unphysical)
pump_absorbed_frac    — must match 1-exp(-κ_p×L) within 5% for small-signal
g(z, λ, t)            — check: g > 0 where N2 > N0×(σ_a/σ_e)
energy_conservation   — E_signal_out + E_ase ≤ E_signal_in + E_pump_absorbed
```

### Cavity diagnostics

```
n2(k)                 — monotonically rising until lasing threshold
E_intra(k)            — exponential rise after threshold
G_rt(k)               — must exceed 1/(1-loss) at threshold
E_pulse               — compare to Frantz-Nodvik estimate
t_FWHM                — check: > T_rt (single round-trip not captured)
```

### Crystal stage diagnostics

```
N2(x,y,z) post-pump   — should be maximum on beam axis, decay at edges
ΔT(x,y)               — hottest at centre; check T_max < crystal softening point
dn_thermal(x,y)        — parabolic profile for Gaussian pump → thermal lens
f_thermal              — check >> L_crystal
B(x,y)                 — peak must be < π
|U_out(x,y)|²          — Gaussian profile; check for hot spots from thermal/Kerr
M²                     — should be < 1.5 for nominal params
σ_x, σ_y              — check beam is circular (not astigmatic)

Energy audit:
  E_pump_abs = E_pump_in × pump_absorbed_frac
  E_heat = E_pump_abs × η_heat
  E_fluorescence = N2_mean × V_crystal × h×ν / τ₂₁ × T_pump
  E_signal_gain = E_signal_out - E_signal_in
  Check: E_pump_abs ≈ E_heat + E_fluorescence + E_signal_gain  (within 5%)
```

### Sanity checks (auto-run, show pass/fail in GUI)

```python
CHECKS = [
    ("N conservation",        lambda r: abs(r.N0+r.N2-1).max() < 1e-4),
    ("Pump absorption",       lambda r: abs(r.pump_abs_frac - r.expected_abs) < 0.05),
    ("B-integral",            lambda r: r.B_max < np.pi,   "warning" if > pi/2),
    ("Thermal lens paraxial", lambda r: r.f_thermal > 10 * r.L_crystal),
    ("Peak intensity",        lambda r: r.I_peak < r.material.damage_fluence_j_m2 / r.pulse_fwhm_s),
    ("Saturation check",      lambda r: r.E_in / r.E_sat_FN < 1.0,  "warn if > 0.3"),
    ("Gain sign",             lambda r: (r.g0_center > 0)),
    ("Energy conservation",   lambda r: abs(r.E_balance) / r.E_pump_in < 0.05),
]
```

---

## 12. Bug history and fixes

This section records every significant bug found in the fiber code and its
solid-state analog. Read before touching physics code.

### Bug 1 — Unphysical pump depletion (fiber, fixed 2026-05-15)

**Symptom:** Only 32% pump absorbed in 2 m at 17 dB/m. N₀ → 5% floor.
**Root cause:** Pass A used full RK4 four-level march on µs time steps without
QSS on fast manifolds. N₃ → N_tot because τ₃₂ = 1 ps ≪ Δt = µs → rate
W_p×N₀×τ₃₂ overshoots.
**Fix:** `march_populations_pump_qss()` — fast N₃ solved analytically in QSS.
Remove artificial `clip(N₀/N_tot, 0.05, 1.0)` bleach floor.
**Carry to solid-state:** Use same QSS function. Never RK4 on τ₃₂ timescales.

### Bug 2 — Non-self-consistent N₂_ss formula (fiber, fixed 2026-05-15)

**Symptom:** N₂ → N_tot when W_p×τ₂₁ ≫ 1 (strongly pumped).
**Root cause:** Formula `N₂_ss = W_p,abs × N₀ × τ₂₁` does not conserve N_tot
at high pump rates (N₀ keeps decreasing in the formula's own answer).
**Fix:** `N₂_ss = W_p,abs × N₀ × τ₂₁ / (1 + W_p,abs × τ₂₁)` (derived from
dN₂/dt = 0 self-consistently).
**Carry to solid-state:** Same formula in cavity march and crystal Pass A.

### Bug 3 — N₃/τ₃₂ in float32 signal RK4 (fiber, fixed 2026-05-18)

**Symptom:** NaN in populations after 3 signal RK4 steps.
**Root cause:** N₃ retained float32 noise ~1e-7. τ₃₂ = 1e-12 s →
N₃/τ₃₂ ≈ 1e-7/1e-12 = 1e5 s⁻¹ spurious rate, but in float32 arithmetic
intermediate values can reach 1e30 → overflow → NaN.
**Fix:** In signal pass, set N₃ = 0 identically (QSS already drained it).
**Carry to solid-state:** `kernel_signal_rk4_batch` must set N3 = 0.

### Bug 4 — σ_abs(976nm) 10× too small in Gaussian model (fiber)

**Symptom:** N_tot computed 10× too high from dopant calculator, pump power
requirement 13× too large vs real fiber.
**Root cause:** Gaussian analytical model for σ_abs had wrong exponent.
σ_abs(976nm) = 2.4×10⁻²⁵ m² (correct, Xu 2013), not 2.4×10⁻²⁴ m².
**Fix:** Updated Gaussian parameters in `yb_glass.py`. Validated against [1].
**Carry to solid-state:** For Yb:YAG, σ_abs(940nm) = 7.5×10⁻²⁵ m². Always
verify: `material.sigma_abs_at(pump_wl) > 1e-25 m²` in dopant calculator.

### Bug 5 — Γ_p applied twice (fiber)

**Symptom:** N_tot 71× too large when using cladding dopant spec for core-pumped model.
**Root cause:** Dopant calculator divided by Γ_p (correct for cladding), then
pump rate equation also divided by A_clad (equivalent to another Γ_p factor).
**Fix:** Document clearly: measured κ (dB/m) already includes Γ_p.
`N_tot = κ_np / σ_abs` (not `κ_np / (Γ_p × σ_abs)`) for cladding-pumped fibers.
**Carry to solid-state:** Crystal doping uses at.% → N_tot directly. No Γ_p
in the N_tot calculation. Γ_p used only in rate equation I_p = P_p / A_pump.

### Bug 6 — kernel_gain_slab in ASE loop (fiber taichi kernels)

**Symptom:** ASE output 662 mJ from 30 mJ pump — catastrophic energy violation.
**Root cause:** `kernel_gain_slab` advances P_signal AND computes gain. Called
inside the ASE sweep → signal amplified twice per z-step.
**Fix:** Added `kernel_compute_gain_only` that reads N₀, N₂ and writes to
`_cache.gain` without touching P_signal. ASE loop uses this variant.
**Carry to solid-state:** Never call gain+advance kernel in fluorescence loop.

### Bug 7 — σ_a(1030nm) Gaussian underflow (fiber)

**Symptom:** Reabsorption effectively zero; Yb:glass behaves like 4-level system.
**Root cause:** Gaussian tail for σ_a far from peak → underflow to 5.5×10⁻⁸⁰ m².
**Fix:** Replaced analytical tails with Lorentzian wings or tabulated CSV.
**Carry to solid-state:** Always check `sigma_a(signal_wl) > 1e-27 m²`.

---

## 13. Validation targets

### Fiber pre-amp

| Metric | Target | Test |
|--------|--------|------|
| Pump absorbed | > 85% for 17 dB/m × 2 m | `test_pump_absorption.py` |
| N₂(L, t_end) | ≈ W_p×τ/(1+W_p×τ) | manual QSS |
| Small-signal gain | ≈ Γ_s × (σ_e×N₂ - σ_a×N₀) × L | check with low-energy seed |
| Energy conservation | E_out ≤ E_in + E_pump | within 1% |

### Pump cavity

| Metric | Target | Test |
|--------|--------|------|
| Threshold roundtrip | G_rt > 1/(1-loss) | `test_cavity.py` |
| Q-switch pulse energy | Within 20% of Frantz-Nodvik | `test_cavity.py` |
| Inversion at threshold | ≥ n₂_threshold formula | manual |

### Solid-state crystal (Yb:YAG, 1 at.%, 10 mm, 5 mm beam)

| Metric | Target | Test |
|--------|--------|------|
| Small-signal g₀ | ≈ 0.87 dB/mm at full inversion | `test_solid_amplifier.py` |
| P_sat | ≈ 6.3 kW for w = 0.5 mm | manual |
| E_out (FN) | Within 20% of Frantz-Nodvik | `test_solid_amplifier.py` |
| Energy conservation | E_pump_abs ≈ E_heat + E_fluor + E_sig | within 5% |
| N conservation | max|N₀+N₂-1| < 1e-4 | auto-check |
| BPM free-space | Gaussian propagates correctly through z_R | `test_bpm.py` |

### End-to-end

| Metric | Target |
|--------|--------|
| η_overall | E_signal_out / E_pump_total > 10% |
| M² | < 1.5 for nominal parameters |
| B_max | < π/2 for default settings |
| f_thermal | > 10 × L_crystal |

---

## 14. References

[1] S. Xu et al., "Characteristics and Laser Performance of Yb³⁺-Doped Silica
    LMA Fibers Prepared by the Sol–Gel Method," Fibers 2013, 1(3):93.
    σ_abs(976nm) = 2.4×10⁻²⁵ m², τ_f ≈ 896 µs.

[2] R. Paschotta, "Ytterbium-Doped Fiber Amplifiers," RP Photonics Encyclopedia.
    https://www.rp-photonics.com/ytterbium_doped_fiber_amplifiers.html

[3] J.A. Caird et al., "Quantum Electronic Rate-Equation Analysis of Lasers,"
    Appl. Opt. 1989 — Frantz-Nodvik derivation.

[4] W.W. Rigrod, "Gain Saturation and Output Power of Optical Masers," J. Appl.
    Phys. 34, 2602 (1963) — Rigrod analysis for cavity.

[5] J.W. Goodman, "Introduction to Fourier Optics," 3rd ed. — angular spectrum BPM.

[6] T. Südmeyer et al., "Ultrafast thin disk laser oscillators," JOSAB 2009 —
    Yb:YAG thermal lens measurements.

[7] D. Kouznetsov & J.V. Moloney, "Highly efficient, high-gain, short-length,
    and power-scalable incoherent diode slab-pumped fiber amplifier," JOSAB 2003.

[8] Liekki Yb1200-10/125 fiber datasheet — σ(λ) tables for Yb:silica.
    (Use for validation of yb_glass.py cross-sections.)

[9] Northrop Grumman BATC Yb:YAG data — σ(λ) measured at 300K, π polarization.

[10] P. Peterson et al., "Nonlinear propagation effects in high-power pulsed
     fiber amplifiers," Opt. Express 2006 — B-integral limits in fiber CPA.
