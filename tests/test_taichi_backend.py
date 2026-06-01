"""Taichi GPU backend vs CPU and LP mode calculator tests."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import taichi  # noqa: F401
except ImportError:
    raise SystemExit(0)

from laser_sim.calculators.dopant import estimate_dopant_concentration
from laser_sim.materials import load_material
from laser_sim.physics.fiber_cpa import FiberCPAConfig, run_fiber_cpa
from laser_sim.physics.lp_modes import find_lp_modes
from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec, build_cpa_time_grid


def _pump_absorption_cfg(*, n_z: int = 150) -> FiberCPAConfig:
    material = load_material("yb_glass")
    db_per_m = 17.0
    dopant = estimate_dopant_concentration(
        pump_absorption_db_per_m=db_per_m,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        pump_wavelength_nm=976.0,
        material=material,
        cladding_pumped=True,
    )
    n_yb = dopant.concentration_m3
    pump = PumpPulseSpec(
        wavelength_nm=976.0,
        peak_power_w=300.0,
        duration_s=500e-6,
        shape="cw",
        cw=True,
    )
    sig = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        bandwidth_nm=4.0,
        chirp_duration_s=0.8e-9,
        packet_energy_j=1e-6,
        burst_start_time_s=200e-6,
    )
    t = build_cpa_time_grid(pump_duration_s=500e-6, spec=sig, pump_cw=True)
    wl = np.linspace(1026.0, 1034.0, 48)
    return FiberCPAConfig(
        material=material,
        fiber_length_m=2.0,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        yb_concentration_m3=n_yb,
        n_z=n_z,
        time_s=t,
        wavelength_nm=wl,
        signal=sig,
        pump=pump,
        include_ase=False,
    )


def test_taichi_pump_absorption():
    cfg = _pump_absorption_cfg()
    r_cpu = run_fiber_cpa(cfg, backend="cpu")
    r_gpu = run_fiber_cpa(cfg, backend="taichi")
    assert r_gpu.pump_power_absorbed_fraction > 0.85
    rel = abs(r_gpu.pump_power_absorbed_fraction - r_cpu.pump_power_absorbed_fraction)
    rel /= max(r_cpu.pump_power_absorbed_fraction, 1e-9)
    assert rel < 0.01, (
        f"Taichi pump absorption {r_gpu.pump_power_absorbed_fraction:.4f} "
        f"vs CPU {r_cpu.pump_power_absorbed_fraction:.4f} (rel err {rel:.3f})"
    )


def _cw_gain_cfg(*, n_z: int = 60) -> FiberCPAConfig:
    material = load_material("yb_glass")
    dopant = estimate_dopant_concentration(
        pump_absorption_db_per_m=6.0,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        pump_wavelength_nm=976.0,
        material=material,
        cladding_pumped=True,
    )
    pump = PumpPulseSpec(
        wavelength_nm=976.0,
        peak_power_w=2000.0,
        duration_s=500e-6,
        shape="cw",
        cw=True,
    )
    sig = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        bandwidth_nm=8.0,
        chirp_duration_s=0.8e-9,
        packet_energy_j=1e-6,
        burst_count=5,
        burst_spacing_s=2.5e-9,
        burst_start_time_s=200e-6,
    )
    t = build_cpa_time_grid(pump_duration_s=500e-6, spec=sig, pump_cw=True)
    wl = np.linspace(1022.0, 1038.0, 48)
    return FiberCPAConfig(
        material=material,
        fiber_length_m=2.0,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        yb_concentration_m3=dopant.concentration_m3,
        n_z=n_z,
        time_s=t,
        wavelength_nm=wl,
        signal=sig,
        pump=pump,
        include_ase=False,
    )


def test_taichi_vs_cpu_gain():
    cfg = _cw_gain_cfg()
    r_cpu = run_fiber_cpa(cfg, backend="cpu")
    r_gpu = run_fiber_cpa(cfg, backend="taichi")
    rel = abs(r_gpu.energy_packet_out_j - r_cpu.energy_packet_out_j)
    rel /= max(r_cpu.energy_packet_out_j, 1e-30)
    assert rel < 0.02, (
        f"packet energy Taichi {r_gpu.energy_packet_out_j:.3e} "
        f"vs CPU {r_cpu.energy_packet_out_j:.3e} (rel {rel:.3f})"
    )


def test_lp_modes_30_250():
    modes = find_lp_modes(15e-6, 0.06, 1030e-9)
    assert modes, "expected at least one guided mode at V≈5.5"
    assert modes[0].name == "LP01"
    assert len(modes) >= 3
    gammas = [m.gamma_overlap for m in modes]
    assert all(0.0 < g <= 1.0 for g in gammas)
    l0 = [m for m in modes if m.l == 0]
    assert l0[0].name == "LP01"
    # Fundamental LP01 is the most core-confined l=0 mode (highest Γ).
    assert l0[0].gamma_overlap == max(m.gamma_overlap for m in l0)
