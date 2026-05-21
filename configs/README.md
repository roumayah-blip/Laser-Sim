# Saved simulation configurations

## `functional_5_20_2026.json`

Baseline verified **2026-05-20** (see `reference_diagnostics` in the JSON).

- **Fiber:** 30 µm core, 250 µm cladding, 1 m, 14.6 dB/m @ 976 nm, Liekki `yb_glass`
- **Pump:** 300 W flat-top, 1 ms window
- **Signal:** 10 × 10 µJ pulses, 2.5 ns spacing, 1 ns chirp @ 1030 nm, burst at 1 ms
- **Grid:** `n_z = 500`, CUDA backend, ASE on, diagnostics export on

### GUI

1. Sidebar → **Quick load** → `functional_5_20_2026` (bundled under `laser_sim/gui/presets/`)
2. Or run: `python scripts/install_functional_preset.py` then reload the app

### Programmatic

Load `configs/functional_5_20_2026.json` and map `fiber_preset` + `simulation` into `SimInputs` / `FiberCPAConfig`.
