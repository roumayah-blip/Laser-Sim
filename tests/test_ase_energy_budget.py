"""Coupled ASE respects pump energy budget."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.materials import load_material
from laser_sim.physics.fiber_cpa import FiberCPAConfig, run_fiber_cpa
from laser_sim.physics.energy_budget import compute_amplifier_energy_budget
from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec


def test_ase_emission_within_pump_budget():
    mat = load_material("yb_glass")
    sig = ChirpedBurstSpec(
        burst_count=3,
        packet_energy_j=5e-6,
        burst_start_time_s=50e-6,
        chirp_duration_s=1e-9,
        burst_spacing_s=5e-9,
        bandwidth_nm=2.0,
    )
    t = np.linspace(0.0, 120e-6, 400)
    wl = np.linspace(1028, 1032, 32)
    cfg = FiberCPAConfig(
        material=mat,
        fiber_length_m=0.5,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        n_z=40,
        time_s=t,
        wavelength_nm=wl,
        signal=sig,
        pump=PumpPulseSpec(peak_power_w=30.0, cw=True, duration_s=0.5e-3),
        include_ase=True,
        yb_concentration_m3=5e24,
    )
    r = run_fiber_cpa(cfg, backend="cpu")
    assert r.energy_balance_ok, (
        f"emit exceeds pump: residual={r.energy_balance_residual_j*1e3:.4f} mJ"
    )
    assert r.energy_ase_out_j >= 0.0
    assert r.ase_fraction_of_emission >= 0.0


def test_no_ase_when_inverted_below_threshold():
    """Very low pump → little ASE; budget still closes."""
    mat = load_material("yb_glass")
    sig = ChirpedBurstSpec(burst_count=1, packet_energy_j=1e-7, burst_start_time_s=10e-6)
    t = np.linspace(0.0, 80e-6, 300)
    wl = np.linspace(1029, 1031, 24)
    cfg = FiberCPAConfig(
        material=mat,
        fiber_length_m=0.2,
        n_z=20,
        time_s=t,
        wavelength_nm=wl,
        signal=sig,
        pump=PumpPulseSpec(peak_power_w=1.0, cw=True, duration_s=0.2e-3),
        include_ase=True,
        yb_concentration_m3=5e24,
    )
    r = run_fiber_cpa(cfg, backend="cpu")
    budget = compute_amplifier_energy_budget(
        t_s=r.t_s,
        z_m=r.z_m,
        populations=r.populations,
        wavelength_nm=r.wavelength_nm,
        n_tot=cfg.yb_concentration_m3,
        tau_21_s=mat.lifetime_s,
        a_signal_m2=r.a_signal_m2,
        eta_guided=r.eta_guided_spontaneous,
        gamma_s=r.gamma_signal,
        energy_pump_in_j=r.energy_pump_in_j,
        energy_pump_out_j=r.energy_pump_out_j,
        energy_packet_in_j=r.energy_packet_in_j,
        energy_packet_out_j=r.energy_packet_out_j,
        energy_ase_out_j=r.energy_ase_out_j,
    )
    assert budget.balance_ok
