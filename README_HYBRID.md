# Hybrid Fiber + Solid-State CPA Simulator

`hybrid_sim/` extends the existing `laser_sim/` fiber CPA engine with a 2D solid-state amplifier (Yb:YAG / Yb:YLF / Nd:YAG), optional pump cavity, thermal lensing, Kerr B-integral, and split-step BPM.

## Install

Uses the same virtual environment as `laser_sim`:

```bash
cd "/home/pat/Laser Sim"
python -m venv .venv
.venv/bin/pip install -r requirements.txt  # if present
```

## GUI

```bash
chmod +x run_hybrid_gui.sh
./run_hybrid_gui.sh
```

## Programmatic example

```python
from hybrid_sim.gui.runner import HybridSimInputs, run_hybrid_safe

inp = HybridSimInputs(
    fiber_length_m=2.0,
    ss_crystal="yb_yag",
    ss_doping_at_pct=1.0,
    signal_energy_nj=10.0,
)
out = run_hybrid_safe(inp)
if out.ok:
    print("η =", out.result.eta_overall)
    print("B_max =", out.result.solid.B_integral_max)
```

## Tests

```bash
.venv/bin/pytest tests/test_cavity.py tests/test_thermal.py tests/test_bpm.py \
  tests/test_b_integral_hybrid.py tests/test_solid_amplifier.py -q
# End-to-end (slower):
.venv/bin/pytest tests/test_hybrid.py -q -m slow
```

## Architecture

1. **Fiber** — `laser_sim.run_simulation()` (unchanged ground truth)
2. **Free-space** — Gaussian ABCD waist tracking in `hybrid_sim.gui.runner`
3. **Solid-state** — `hybrid_sim.physics.solid_amplifier`:
   - Pass A: QSS pump inversion + 2D heat source
   - Pass B: 2D heat equation → thermal lens
   - Pass C: Angular-spectrum BPM + gain + Kerr phase

See `HYBRID_SIM_PHYSICS_REFERENCE.md` (copy from your Downloads spec) for equations and validation targets.

## Not yet implemented

- Taichi/CUDA kernels (`taichi_kernels_solid.py`)
- Full 5-tab Streamlit diagnostics panel
- Interactive cavity beam-path SVG
- Spectral multi-λ BPM march (single-λ representative channel today)
