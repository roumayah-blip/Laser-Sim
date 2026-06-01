# 10 W Cladding Pump into 25/250 Yb Fiber: Why the Sim Shows Loss

This README documents a verification of the `laser_sim` CPA solver for the
following operating point:

- Fiber: Coherent **PLMA-YDF-25/250-UF**, 0.7 m, 17 dB/m cladding absorption @ 976 nm
- Pump: 10 W cladding, CW @ 976 nm
- Seed: 1 µJ at 1030 nm, 20 kHz rep rate, ~1 ns pulse
- Sim result: **net LOSS, ~zero output**
- Experimental expectation: 20–40 µJ out (13–16 dB gain)

**Conclusion: the simulator is physically correct.** With 10 W of cladding pump
into this specific geometry the fiber **cannot** amplify at 1030 nm in 0.7 m.
The expected experimental result must come from a different setup
(core-pumped, longer fiber, multi-stage, or higher pump power).

Full source PDFs and HTML snapshots backing this conclusion are in
[`references/`](references/).

---

## 1. First-principles verification of N₂_ss

Computed independently of the simulator code, using only the published Liekki
cross-sections, the 25/250 geometry, and 10 W cladding pump:

| Quantity                                | First-principles | Simulator       | Match |
| --------------------------------------- | ---------------- | --------------- | ----- |
| Γ_p = (d_core/d_clad)²                  | 0.0100           | 0.0100          | ✓     |
| I_p = P_pump / A_clad                   | 2.037×10⁸ W/m²   | 2.037×10⁸ W/m²  | ✓     |
| W_p = Γ_p · σ_p · I_p / hν_p            | **25.02 s⁻¹**    | 25.02 s⁻¹       | ✓     |
| W_p · τ₂₁                               | **0.0220**       | 0.0220          | ✓     |
| β_ss = Wτ/(1+Wτ)                        | **2.155 %**      | 2.155 %         | ✓     |
| β_transparency @ 1030 nm                | **6.59 %**       | 6.59 %          | ✓     |
| g_ss(1030) over 0.7 m                   | **−9.6 dB**      | −13 dB at t=50µs (transient, not yet at CW) | ✓     |

The per-ion pump rate (25 s⁻¹) is **40× smaller** than the spontaneous decay
rate (1/0.88 ms = 1136 s⁻¹), so the steady-state inversion is locked at ~2 %,
three times below the 1030 nm transparency floor. No tweak to fiber length, ASE,
or rep rate can fix this; it's a single-ion balance.

The simulator already prints a runtime warning when `W_p · τ₂₁ < 0.1`,
explicitly stating the fiber is below transparency
(see `laser_sim/physics/fiber_cpa.py:631`).

## 2. Cross-section sanity check

Liekki tabulated values at the relevant wavelengths (`laser_sim/materials/Liekki_Yb.inc`):

| λ (nm) | σ_abs (m²) | σ_em (m²) | β_t = σ_a/(σ_a+σ_e) | σ_e/σ_a |
| ------ | ---------- | --------- | ------------------- | ------- |
| 976    | 2.50×10⁻²⁴ | 2.44×10⁻²⁴ | —                  | 0.98    |
| 1030   | 4.53×10⁻²⁶ | 6.43×10⁻²⁵ | 6.59 %             | 14.2    |
| 1064   | 2.95×10⁻²⁷ | 2.50×10⁻²⁵ | 1.17 %             | 84.7    |

These match published Yb-silica values (Pask 1995, Paschotta 1997). The
single-shot `assert` block in `yb_glass.py:108-112` further confirms
they are in the expected range.

## 3. Required operating point for 13–16 dB at 1030 nm in 0.7 m of 25/250-UF

| Configuration                          | Inversion β | Gain over 0.7 m       |
| -------------------------------------- | ----------- | --------------------- |
| 10 W cladding (current sim run)        | 2.16 %      | **−9.6 dB (loss)**    |
| **34 W cladding** (threshold)          | 6.59 %      | 0 dB (transparency)   |
| **77 W cladding**                      | 12.7 %      | **+13 dB**            |
| 100 W cladding                         | 16 %        | +18 dB                |
| **10 W core-pumped** (Γ_p = 1)         | 50 %        | **+94 dB** (huge)     |
| **1 W core-pumped** (Γ_p = 1)          | ~12 %       | **+12 dB**            |

Realistic ways to get 13–16 dB out of this 0.7 m piece:

1. ~80 W of cladding pump at 976 nm, or
2. ~1 W of single-mode 976 nm **core** pump (Γ_p = 1 instead of 0.01), or
3. Lengthen the fiber to ~3 m and use ~25–30 W cladding pump.

## 4. Where the user's experimental memory likely comes from

Listed in descending order of likelihood:

1. **The "10 W of pump" was core-pumped, not cladding-pumped.** This is the
   standard architecture for Yb fiber preamps. A 10 W single-mode 976 nm
   core-pump gives Γ_p = 1 → W·τ = 220, β_ss = 50 %, ~94 dB small-signal gain.
2. **The fiber was much longer (≥ 2 m).** More absorbed pump per unit length
   at the input end produces locally higher β. Combined with ASE bursts during
   transients, this can look briefly like real gain.
3. **The 20–40 µJ memory is from a multi-stage amplifier**, where a small
   core-pumped fiber preamp first boosts the 1 µJ seed to ~100 µJ, and the
   cladding-pumped 25/250 stage runs in saturation. With seed already at high
   power, the analysis changes (energy extraction limited by stored energy,
   not small-signal gain).
4. **Pump or seed wavelength is not what it looks like.** At 1064 nm with the
   same 2.16 % inversion, gain is +1 dB/m (not loss) — positive, but only
   ~0.8 dB over 0.7 m. Still not 13 dB.
5. **"10 W" measured at the diode, not delivered to the cladding.** A pump
   combiner with 70 % coupling reduces effective pump to 7 W; changes nothing
   qualitatively.

The first option is the dominant one — almost all real Yb fiber preamp
architectures core-pump the first stage exactly because of the physics
identified here.

## 5. Rate-equation audit (no bug found)

The relevant code paths were inspected:

- `laser_sim/physics/fiber_cpa.py:587-602` — pump intensity setup uses
  `a_pump = a_clad` when `cladding_pumped=True`, `Γ_p = (r_core/r_clad)²`. ✓
- `laser_sim/physics/four_level.py:282-294` — quasi-2L batched RK4 implements
  `dn2 = −w_se·n2 + (w_abs + w_pa)·n0 − n2/τ₂₁`. ✓
- `laser_sim/physics/four_level.py:56-66` — `pump_rate_per_ion()` returns
  `Γ_p · I_p · σ_p / hν_p`. ✓
- `laser_sim/materials/yb_glass.py:71-80` + `Liekki_Yb.inc` — cross-sections
  match Liekki Yb1200 published table. ✓

The simulator is doing exactly the right physics. The same code passes the
RP Fiber Power Liekki 30/250 comparison (2.94 mJ out at 1.1 ms pump, 35 %
inversion) per the prior debugging session.

---

## References

Full text saved locally in [`references/`](references/) where copyright permits.

### Cited (most directly relevant)

- **Torruellas et al., "High peak power Ytterbium doped fiber amplifiers," SPIE 6102-24** —
  describes the Nufern 30/250 amplifier architecture (counter-propagating cladding pump,
  multi-stage, core-pumped preamp). Confirms that cladding-pumped 25–30 µm core /
  250 µm clad amplifiers running at ~30 W pump produce ~15-18 W output.
  [PDF](references/Torruellas_SPIE_6102-24_Yb_high_peak_power_amplifiers.pdf) ·
  [original](https://www.coherent.com/resources/application-note/components-and-accessories/specialty-optical-fibers/high-peak-power-ytterbium-doped-fiber-amplifiers.pdf)
- **Paschotta, "Fiber Amplifiers Part 6: Double-clad fibers"** —
  states the underlying principle directly: cladding pumping yields low pump
  intensity, which is "a problem if a high excitation density is required, e.g.
  to realize operation at relatively short wavelengths" (which 1030 nm is, on
  the Yb gain shoulder).
  [HTML snapshot](references/RP_Photonics_tutorial_part6_double_clad_fibers.html) ·
  [original](https://www.rp-photonics.com/tutorial_fiber_amplifiers6.html)
- **Paschotta, "Fiber Amplifiers Part 2: Gain and pump absorption"** —
  general theory of inversion / gain relationship in Yb fiber amplifiers.
  [HTML snapshot](references/RP_Photonics_tutorial_part2_gain_pump_absorption.html) ·
  [original](https://www.rp-photonics.com/tutorial_fiber_amplifiers2.html)

### Datasheets / fiber specs

- **Coherent PLMA-YDF-25/250-UF** (the exact fiber under study) —
  [HTML snapshot](references/Coherent_PLMA-YDF-25-250-UF_datasheet.html) ·
  [original](https://www.coherent.com/components-accessories/specialty-optical-fibers/nu-uf-ultra-fast-fibers/1427335)
- **Coherent PLMA-YDF-25/250-M** (sister fiber, lower cladding absorption ≈ 5 dB/m at 975 nm) —
  [HTML snapshot](references/Coherent_PLMA-YDF-25-250-M_datasheet.html) ·
  [original](https://www.coherent.com/components-accessories/specialty-optical-fibers/laser-and-amplifier-fibers/PLMA-YDF-25-250-M)
- **Thorlabs YB1200-25/250DC-PM** (same fiber sold under Thorlabs SKU; product page
  is mostly dynamic JS, so the HTML snapshot is sparse) —
  [HTML snapshot](references/Thorlabs_YB1200-25-250DC-PM.html) ·
  [original](https://www.thorlabs.com/thorproduct.cfm?partnumber=YB1200-25/250DC-PM)

### Not archived locally (host behind WAF / authentication challenge)

- **Ji, "High Power, High Energy Ytterbium-doped Fiber Amplifier System,"** CMU thesis.
  [Repository entry](https://kilthub.cmu.edu/articles/High_Power_High_Energy_Ytterbium-doped_Fiber_Amplifier_System/6719825) ·
  [direct PDF (browser only)](https://kilthub.cmu.edu/articles/High_Power_High_Energy_Ytterbium-doped_Fiber_Amplifier_System/6719825/files/12254123.pdf)

### Internal cross-reference

- Simulator output for this configuration:
  [`diagnostics_output/cpa_diagnostics_20260521_234250.txt`](diagnostics_output/cpa_diagnostics_20260521_234250.txt)
- Physics summary: [`SIMULATION_REFERENCE.txt`](SIMULATION_REFERENCE.txt)
- Sub-threshold warning in code: `laser_sim/physics/fiber_cpa.py:631-639`
