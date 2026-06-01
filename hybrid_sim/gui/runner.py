"""Hybrid fiber + solid-state simulation orchestration."""

from __future__ import annotations

import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np

from hybrid_sim.calculators.dopant import concentration_from_at_percent, validate_cross_sections
from hybrid_sim.calculators.runtime import estimate_solid_runtime_s
from hybrid_sim.materials import load_material
from hybrid_sim.physics.cavity import CavityConfig
from hybrid_sim.physics.solid_amplifier import SolidAmplifierConfig, SolidAmplifierResult, run_solid_amplifier
from hybrid_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec

from laser_sim.gui.runner import SimInputs, SimRunOutcome, run_simulation


@dataclass
class HybridSimInputs:
    fiber_material: str = "yb_glass"
    fiber_length_m: float = 2.0
    fiber_core_um: float = 10.0
    fiber_clad_um: float = 125.0
    fiber_pump_abs_db_per_m: float = 6.0
    fiber_pump_power_w: float = 2.0
    fiber_pump_duration_us: float = 1000.0

    collimator_f_mm: float = 11.0
    relay_f1_mm: float = 75.0
    relay_f2_mm: float = 150.0

    use_cavity_pump: bool = True
    cavity_crystal: str = "nd_yag"
    cavity_length_mm: float = 150.0
    cavity_r_oc: float = 0.70
    cavity_pump_power_w: float = 300.0
    cavity_pump_duration_ms: float = 1.0
    q_switch: bool = True
    q_switch_open_time_us: float = 10.0

    ss_crystal: str = "yb_yag"
    ss_crystal_length_mm: float = 10.0
    ss_crystal_diameter_mm: float = 10.0
    ss_doping_at_pct: float = 1.0
    ss_n_z: int = 100
    ss_n_x: int = 64
    ss_n_y: int = 64
    ss_beam_waist_mm: float = 0.5
    ss_pump_waist_mm: float = 0.6
    include_thermal: bool = True
    include_kerr: bool = True

    chirp_duration_ns: float = 2.0
    burst_count: int = 5
    burst_spacing_ns: float = 2.5
    burst_delay_us: float = 200.0
    signal_energy_nj: float = 10.0

    n_lam: int = 32
    wavelength_min_nm: float = 1020.0
    wavelength_max_nm: float = 1040.0
    backend: str = "numpy"


@dataclass
class HybridSimResult:
    fiber: SimRunOutcome
    solid: SolidAmplifierResult
    eta_overall: float
    beam_waist_collimator_m: float
    beam_waist_crystal_m: float


def _fiber_inputs(h: HybridSimInputs) -> SimInputs:
    return SimInputs(
        material_key=h.fiber_material,
        core_diameter_um=h.fiber_core_um,
        cladding_diameter_um=h.fiber_clad_um,
        fiber_length_m=h.fiber_length_m,
        pump_absorption_db_per_m=h.fiber_pump_abs_db_per_m,
        pump_peak_power_w=h.fiber_pump_power_w,
        pump_duration_s=h.fiber_pump_duration_us * 1e-6,
        chirp_duration_s=h.chirp_duration_ns * 1e-9,
        burst_count=h.burst_count,
        burst_spacing_s=h.burst_spacing_ns * 1e-9,
        burst_start_s=h.burst_delay_us * 1e-6,
        packet_energy_j=h.signal_energy_nj * 1e-9,
        backend=h.backend if h.backend != "numpy" else "cpu",
    )


def gaussian_beam_propagate(
    w0_m: float,
    wavelength_m: float,
    f_m: float,
    n: float = 1.0,
) -> float:
    """Beam waist after thin lens (Gaussian ABCD, w_in = w0 at lens)."""
    z_r = np.pi * w0_m**2 * n / wavelength_m
    if abs(f_m) < 1e-12:
        return w0_m
    # q at lens: iz = 0, q = i*z_R
    q = 1j * z_r
    q_out = q / (1.0 - q / f_m)
    inv_q = 1.0 / q_out
    w2 = -wavelength_m / (np.pi * n * np.imag(inv_q))
    return float(np.sqrt(max(w2, 0.0)))


def run_hybrid_simulation(inp: HybridSimInputs) -> HybridSimResult:
    fiber_out = run_simulation(_fiber_inputs(inp))
    if not fiber_out.ok or fiber_out.result is None:
        raise RuntimeError(fiber_out.error_message or "Fiber stage failed")

    ss_mat = load_material(inp.ss_crystal)
    dop = concentration_from_at_percent(inp.ss_doping_at_pct, ss_mat)
    validate_cross_sections(ss_mat, ss_mat.default_pump_wavelength_nm, ss_mat.default_signal_wavelength_nm)

    sig = ChirpedBurstSpec(
        chirp_duration_s=inp.chirp_duration_ns * 1e-9,
        burst_count=inp.burst_count,
        burst_spacing_s=inp.burst_spacing_ns * 1e-9,
        burst_start_time_s=inp.burst_delay_us * 1e-6,
        packet_energy_j=inp.signal_energy_nj * 1e-9,
    )

    cavity_cfg = None
    pump_spec = None
    if inp.use_cavity_pump:
        cav_mat = load_material(inp.cavity_crystal)
        cav_dop = concentration_from_at_percent(1.0, cav_mat)
        cavity_cfg = CavityConfig(
            crystal=cav_mat,
            crystal_length_m=0.01,
            cavity_length_m=inp.cavity_length_mm * 1e-3,
            yb_concentration_m3=cav_dop.concentration_m3,
            r_oc=inp.cavity_r_oc,
            pump_power_w=inp.cavity_pump_power_w,
            pump_duration_s=inp.cavity_pump_duration_ms * 1e-3,
            q_switch_on_time_s=inp.q_switch_open_time_us * 1e-6 if inp.q_switch else None,
        )
    else:
        pump_spec = PumpPulseSpec(
            peak_power_w=inp.cavity_pump_power_w,
            duration_s=inp.cavity_pump_duration_ms * 1e-3,
        )

    ss_cfg = SolidAmplifierConfig(
        material=ss_mat,
        crystal_length_m=inp.ss_crystal_length_mm * 1e-3,
        crystal_diameter_m=inp.ss_crystal_diameter_mm * 1e-3,
        yb_concentration_m3=dop.concentration_m3,
        cavity=cavity_cfg,
        pump_pulse=None if inp.use_cavity_pump else pump_spec,
        signal=sig,
        beam_waist_m=inp.ss_beam_waist_mm * 1e-3,
        pump_waist_m=inp.ss_pump_waist_mm * 1e-3,
        n_z=inp.ss_n_z,
        n_x=inp.ss_n_x,
        n_y=inp.ss_n_y,
        include_thermal=inp.include_thermal,
        include_kerr=inp.include_kerr,
        n_lam=inp.n_lam,
        wavelength_min_nm=inp.wavelength_min_nm,
        wavelength_max_nm=inp.wavelength_max_nm,
    )

    solid = run_solid_amplifier(ss_cfg)
    e_pump_total = solid.energy_pump_in_j + inp.fiber_pump_power_w * inp.fiber_pump_duration_us * 1e-6
    eta = solid.energy_signal_out_j / max(e_pump_total, 1e-30)

    wl = inp.wavelength_min_nm * 1e-9
    w_fiber = 0.5 * inp.fiber_core_um * 1e-6
    w_col = gaussian_beam_propagate(w_fiber, wl, inp.collimator_f_mm * 1e-3)
    w_relay = gaussian_beam_propagate(w_col, wl, inp.relay_f2_mm * 1e-3) * (inp.relay_f2_mm / inp.relay_f1_mm)

    return HybridSimResult(
        fiber=fiber_out,
        solid=solid,
        eta_overall=float(eta),
        beam_waist_collimator_m=w_col,
        beam_waist_crystal_m=w_relay,
    )


@dataclass
class HybridRunOutcome:
    ok: bool
    result: HybridSimResult | None = None
    error_message: str = ""
    traceback_text: str = ""
    solid_runtime_estimate_s: float | None = None


def run_hybrid_safe(inp: HybridSimInputs) -> HybridRunOutcome:
    try:
        est = estimate_solid_runtime_s(inp.ss_n_z, 500, inp.ss_n_x, inp.ss_n_y, backend=inp.backend)
        res = run_hybrid_simulation(inp)
        return HybridRunOutcome(ok=True, result=res, solid_runtime_estimate_s=est)
    except Exception as e:
        return HybridRunOutcome(
            ok=False,
            error_message=str(e),
            traceback_text=traceback.format_exc(),
        )
