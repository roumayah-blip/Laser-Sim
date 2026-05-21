# Laser Sim

Time-resolved, spectrally resolved **ytterbium fiber CPA amplifier** simulation for chirped ns+ signal bursts, µs–ms pump pulses, ASE, and spontaneous emission. Built for **Yb:glass**, **Yb:YAG**, and **Yb:YLF** with a Streamlit GUI, calculators, and optional **Taichi CUDA** acceleration (RTX 5090 class GPUs).

---

## Detailed reference

**[SIMULATION_REFERENCE.txt](SIMULATION_REFERENCE.txt)** — full code map, two-pass four-level solver, step-by-step z/t marching, pump absorption model, literature lifetimes, and known limitations.

## Table of contents

- [What has been built](#what-has-been-built)
- [Project layout](#project-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [NVIDIA driver setup](#nvidia-driver-setup)
- [Running the GUI](#running-the-gui)
- [Command reference](#command-reference)
- [Using the GUI](#using-the-gui)
- [Programmatic use](#programmatic-use)
- [Physics scope (v0.1)](#physics-scope-v01)
- [Known limitations](#known-limitations)
- [Roadmap and next steps](#roadmap-and-next-steps)

---

## What has been built

### 1. Core simulation (`laser_sim/physics/fiber_cpa.py`, `four_level.py`)

- **Four-level manifolds** N₀, N₁, N₂, N₃ with literature-based τ₂₁, τ₃₂, τ₁₀ (see `SIMULATION_REFERENCE.txt`).
- **Two-pass solver:** Pass A — pump builds populations; Pass B — signal with gain saturation and N₁ bottlenecking.
- **Radially symmetric fiber** with fixed core/cladding mode areas (no transverse beam propagation yet).
- **Cladding or core pumping** with overlap factor Γ_p for geometry.
- **Spectrally resolved** signal on a wavelength grid: `P_s(z, t, λ)` in W/nm.
- **Pump propagation** using measured fiber κ (dB/m): `P(z+Δz)=P(z)·exp(−κ·(N₀/N_tot)·Δz)`; energy absorption fraction reported in results.
- **ASE + spontaneous emission**: forward/backward ASE bins with spontaneous source terms.
- **CPA-oriented pulses**: chirped ns+ envelopes (not fs/ps primary), **pulse packets up to 50 pulses** with **0.5–1000 ns spacing** (default **1 ns**). Overlapping pulses **sum in intensity** in time. Burst start delay aligns the packet with the pump.
- **Pump shapes**: flat-top, Gaussian, trapezoid over **1 µs–5 ms** (configurable).

Implementation is currently **NumPy** (CPU). Taichi is installed and verified on CUDA; the simulation kernels are **not yet ported** to Taichi.

### 2. Pulse library (`laser_sim/pulses/chirp.py`)

| Type | Class / function | Notes |
|------|------------------|--------|
| Chirped CPA packet | `ChirpedBurstSpec`, `build_chirped_burst()` | Spectral width + linear λ→time chirp; 0.5 ns min spacing (default 1 ns); overlapping pulses add in intensity |
| Pump envelope | `PumpPulseSpec`, `build_pump_power()` | Pulsed (µs–ms) or **CW** steady over simulation window |
| fs/ps hook | `ultrashort_mode` on `ChirpedBurstSpec` | Reserved; not used in v0.1 |

### 3. Materials (`laser_sim/materials/`)

| Key | File | Description |
|-----|------|-------------|
| `yb_glass` | `yb_glass.py` | Broad smooth σ_abs, σ_em (typical fiber CPA) |
| `yb_yag` | `yb_yag.py` | Narrow Lorentzian lines (~300 K, π-pol approximate) |
| `yb_ylf` | `yb_ylf.py` | Structured lines (E∥c approximate) |

Cross-sections are **analytical placeholders**. Replace with measured CSV data for production runs (see [Roadmap](#roadmap-and-next-steps)).

### 4. Dopant calculator (`laser_sim/calculators/dopant.py`)

Back-calculates **Yb³⁺ concentration** from common fiber specs:

- Pump absorption (**dB/m** or **total dB** over fiber length)
- Core and cladding diameter (µm)
- Pump wavelength (nm)
- Material (for σ_abs at pump λ)
- Cladding vs core pump geometry

**Default:** datasheet dB/m is treated as **measured fiber attenuation** (Γ_p is **not** applied twice when solving for N). An advanced sidebar option applies Γ_p again for bulk overlap modeling.

Outputs: `concentration_m3`, `concentration_for_rates_m3` (used in simulation), Γ_p, σ_abs, pump area, notes.

### 5. Runtime estimator (`laser_sim/calculators/runtime.py`)

Estimates wall-clock time from grid size:

- `n_z` × `n_t` × `n_λ`, burst count, ASE on/off, backend (`cpu`, `cuda`, `taichi_cuda`)
- Recommends `n_t` and `n_λ` from pump duration, chirp length, burst span, and spectral sampling
- **Calibration factor** slider to match one timed run on your hardware

### 6. Streamlit GUI (`laser_sim/gui/app.py`)

**Sidebar:** material, fiber geometry, pump absorption → dopant N, optional manual override.

**Tab — Calculators & grid planner:** pump/signal/burst parameters, grid recommendation, runtime estimate.

**Tab — Run simulation:** full CPA run with Plotly plots:

- Pump absorption α_pump(z) and mean pump power vs z
- Temporal signal in/out (spectrally integrated)
- Spectrum in/out (time-integrated)
- Output `P(t, λ)` heatmap
- Energy metrics: pump in/out, signal in/out, ASE out

Launch via `./run_gui.sh` or `streamlit run laser_sim/gui/app.py`.

### 7. Utilities

| Script | Purpose |
|--------|---------|
| `scripts/check_cuda.py` | Verify NVIDIA driver + Taichi CUDA kernel smoke test |
| `tests/test_calculators.py` | Unit tests for dopant and runtime calculators |
| `run_dpg.sh` | Launch Dear PyGui desktop app |
| `run_gui.sh` | Launch Streamlit browser app |

### 8. Environment verified on this machine

- **Ubuntu 24.04**, kernel 6.17
- **NVIDIA GeForce RTX 5090 Laptop GPU**, driver **595.58.03** (open kernel module)
- **Taichi 1.7.4** on **CUDA** (Python 3.13 venv)
- Secure Boot was blocking DKMS modules; **disabling Secure Boot** allowed `nvidia` module load (alternative: install signed `linux-modules-nvidia-*` packages)

---

## Project layout

```
Laser Sim/
├── README.md                 # This file
├── requirements.txt
├── run_dpg.sh                # Launch Dear PyGui desktop GUI
├── run_gui.sh                # Launch Streamlit GUI
├── scripts/
│   └── check_cuda.py         # Taichi + NVIDIA smoke test
├── tests/
│   └── test_calculators.py
└── laser_sim/
    ├── __init__.py
    ├── constants.py
    ├── calculators/
    │   ├── dopant.py         # N from pump absorption
    │   └── runtime.py        # Grid + time estimates
    ├── materials/
    │   ├── base.py
    │   ├── yb_glass.py
    │   ├── yb_yag.py
    │   └── yb_ylf.py
    ├── pulses/
    │   └── chirp.py          # CPA + pump envelopes
    ├── physics/
    │   └── fiber_cpa.py      # Main amplifier engine
    └── gui/
        ├── dpg_app.py        # Dear PyGui desktop application
        ├── runner.py         # Shared sim runner + error handling
        └── app.py            # Streamlit application
```

---

## Requirements

- **Python 3.11+** (3.13 works with Taichi 1.7.4 in this project)
- **NVIDIA GPU + driver** for CUDA/Taichi GPU (optional for CPU-only NumPy runs)
- See `requirements.txt`: `numpy`, `scipy`, `streamlit`, `plotly`, `h5py`, `taichi`

---

## Installation

From the project root:

```bash
cd "/home/pat/Laser Sim"

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

Run unit tests (no GPU required):

```bash
source .venv/bin/activate
python tests/test_calculators.py
```

---

## NVIDIA driver setup

### Check driver and GPU

```bash
nvidia-smi
```

Expected: GPU name, driver version (e.g. `595.58.03`), memory.

### If `nvidia-smi` fails

**Symptom:** `couldn't communicate with the NVIDIA driver` or `Key was rejected by service` in `dmesg`.

**Common cause on Ubuntu:** **Secure Boot** blocks unsigned DKMS modules.

**Option A — Disable Secure Boot (done on this workstation)**

1. Reboot → firmware setup (often F2 / F12 / Del on Dell).
2. Disable **Secure Boot**, save, reboot.
3. Verify:

```bash
sudo modprobe nvidia
nvidia-smi
lsmod | grep nvidia
```

**Option B — Keep Secure Boot enabled**

Install Ubuntu **pre-signed** kernel modules for your kernel (example for 6.17.0-23):

```bash
sudo apt update
sudo apt install linux-modules-nvidia-595-open-$(uname -r)
sudo reboot
```

**Option C — Reinstall recommended driver**

```bash
ubuntu-drivers devices
sudo apt install nvidia-driver-595-open
sudo reboot
```

### Verify Taichi CUDA

```bash
cd "/home/pat/Laser Sim"
source .venv/bin/activate
python scripts/check_cuda.py
```

Expected output includes:

```
[Taichi] Starting on arch=cuda
Active arch: Arch.cuda
CUDA kernel OK
```

**Taichi 1.7+ API note:** `ti.lang.impl.current_arch()` was removed. Use:

```python
import taichi as ti
ti.init(arch=ti.cuda)
print(ti.cfg.arch)   # Arch.cuda
```

Optional: allocate more GPU memory for large grids:

```python
ti.init(arch=ti.cuda, device_memory_fraction=0.85)
```

---

## Running the GUI

### Dear PyGui (recommended — native desktop + error log)

```bash
cd "/home/pat/Laser Sim"
source .venv/bin/activate
./run_dpg.sh
```

Or:

```bash
python -m laser_sim.gui.dpg_app
```

Native window with plots and a **Log / errors** panel showing full tracebacks on failure. Supports **pulsed** pump (flat-top, Gaussian, trapezoid) and **CW (steady)** pump over a simulation time window.

### Streamlit (browser UI, optional)

```bash
./run_gui.sh
# or: streamlit run laser_sim/gui/app.py
```

Browser opens at `http://localhost:8501` (default).

---

## Command reference

| Task | Command |
|------|---------|
| Activate venv | `source .venv/bin/activate` |
| Install deps | `pip install -r requirements.txt` |
| Run GUI (desktop) | `./run_dpg.sh` |
| Run GUI (browser) | `./run_gui.sh` |
| Run tests | `python tests/test_calculators.py` |
| Check CUDA | `python scripts/check_cuda.py` |
| GPU status | `nvidia-smi` |
| Load module (if needed) | `sudo modprobe nvidia` |

---

## Using the GUI

### Recommended workflow

1. **Sidebar** — Select material, enter core/cladding diameters, fiber length, pump absorption (dB/m or total dB), pump wavelength. Read off **Yb concentration** (or override manually).
2. **Calculators tab** — Set pump duration (µs), chirped pulse duration (ns), burst count/spacing, `n_z`, points/nm. Note **recommended n_t**, **n_λ**, and **estimated runtime**.
3. **Run tab** — Set pump peak power, pulse energy, burst parameters, **first burst delay** (µs) so chirped pulses sit on pump build-up (typical: 200–500 µs into a 1 ms pump). Click **Run CPA simulation**.
4. Compare energy metrics and plots to your expectations; iterate burst timing and N.

### CPA timing tip

Bursts at `t = 0` see little gain because inversion has not built up. Use **First burst delay after t=0** to place the chirped train after the pump has excited the fiber.

---

## Programmatic use

```python
from laser_sim.materials import load_material, YB_GLASS
from laser_sim.calculators.dopant import estimate_dopant_concentration
from laser_sim.calculators.runtime import estimate_runtime, recommend_time_grid, recommend_wavelength_grid
from laser_sim.physics.fiber_cpa import FiberCPAConfig, run_fiber_cpa
from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec

mat = load_material("yb_glass")
dopant = estimate_dopant_concentration(
    pump_absorption_db_per_m=6.0,
    core_diameter_um=10.0,
    cladding_diameter_um=400.0,
    pump_wavelength_nm=976.0,
    material=mat,
    cladding_pumped=True,
)

cfg = FiberCPAConfig(
    material=mat,
    fiber_length_m=2.0,
    core_diameter_um=10.0,
    cladding_diameter_um=400.0,
    yb_concentration_m3=dopant.concentration_for_rates_m3,
    pump_absorption_db_per_m=6.0,
    signal=ChirpedBurstSpec(
        chirp_duration_s=2e-9,
        burst_count=5,
        burst_spacing_s=1e-9,
        burst_start_time_s=200e-6,
        energy_per_pulse_j=1e-6,
    ),
    pump=PumpPulseSpec(duration_s=1e-3, peak_power_w=200.0, shape="flat_top"),
)

result = run_fiber_cpa(cfg)
print("Signal gain (energy)", result.energy_signal_out_j / result.energy_signal_in_j)
print("Pump energy in/out (mJ)", result.energy_pump_in_j * 1e3, result.energy_pump_out_j * 1e3)
```

---

## Physics scope (v0.1)

| Included | Not yet |
|----------|---------|
| Quasi-two-level Yb (N₁, N₂) | fs/ps GVD, Kerr, NLSE |
| Spectral signal + ASE bins | 3D crystal BPM / diffraction |
| Spontaneous + ASE amplification | Temperature-dependent crystal lines |
| Cladding/core pump geometry | Arbitrary pump waveforms from file |
| Chirped ns+ bursts (≤50) | Taichi GPU simulation kernels |
| Measured κ for pump decay | Pulse packets / shaped pump trains |

---

## Known limitations

- Cross-sections are **placeholder curves**; validate against your glass/crystal data.
- Signal gain is sensitive to **burst timing** vs pump; align delay with your CPA architecture.
- `concentration_for_rates_m3` can be large when reconciling measured κ with rate equations; calibrate against one known fiber.
- Simulation is **single-pass fiber amplifier** (no resonator/cavity).
- NumPy backend may be slow for very large `n_z × n_t × n_λ`; Taichi port is planned.

---

## Roadmap and next steps

### Immediate (this week)

#### 1. Validate against one real fiber

Compare simulated pump absorption vs z, signal energy out, and spectrum to a fiber you trust.

```bash
source .venv/bin/activate
./run_gui.sh
# Or run programmatic example above and adjust burst_start_time_s, N, pump power
```

#### 2. Import measured spectra

Add CSV files (wavelength_nm, sigma_abs_m2, sigma_em_m2) and wire into `materials/` (planned: loader in `base.py`). Until then, edit `yb_glass.py` / `yb_yag.py` / `yb_ylf.py`.

#### 3. End-to-end GUI run

```bash
cd "/home/pat/Laser Sim"
source .venv/bin/activate
./run_gui.sh
```

Use **Calculators** tab for grid/runtime, then **Run** with burst delay ~200–500 µs on a 1 ms pump.

---

### Short term (core product)

#### 4. Port simulation to Taichi CUDA

Move `run_fiber_cpa` inner loops to `@ti.kernel` so large CPA grids use the GPU.

```bash
source .venv/bin/activate
python scripts/check_cuda.py   # confirm CUDA before profiling
```

#### 5. Pulse lab upgrades

- Pump waveforms from file  
- Pulse packets (macro × micro burst)  
- Save/load YAML run configs  

#### 6. Calibration workflow

- GUI panel: measured vs simulated  
- Tune runtime `calibration_factor` from one timed run  

---

### Medium term (physics fidelity)

#### 7. CPA timing improvements

Explicit pump/signal delay UI; optional pump-first staging.

#### 8. Stronger inversion model

Pump bleaching, forward + backward pump with power split.

#### 9. YAG / YLF on fiber path

Fine λ mesh near crystal lines; temperature broadening parameter.

---

### Long term (original plan)

#### 10. Crystalline 3D phase

Same rate + spectral core; **BPM or diffraction** propagation in volume; optional waveguide mode.

#### 11. Performance pass

VRAM-aware grid limits; adaptive wavelength clustering near narrow lines.

---

## Priority guide

| Goal | Start with |
|------|------------|
| Trustworthy numbers | Steps **1–2** (validation + real σ(λ)) |
| Faster large runs | Step **4** (Taichi kernels) |
| Daily usability | Steps **5–6** (configs, calibration) |
| Crystals later | Steps **9 → 10** |

---

## License

Add license text here if you distribute the project.
