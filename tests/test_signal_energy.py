"""Signal energy integration over packet window."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.pulses.chirp import (
    ChirpedBurstSpec,
    build_chirped_burst,
    build_cpa_time_grid,
    integrate_packet_energy,
    integrate_single_pulse_energy,
    packet_energy_expected_j,
)


def test_packet_energy_independent_of_spacing():
    """5×1 µJ pulses: packet integral ≈ 5 µJ for 1 ns or 10 ns spacing."""
    expected_pkt = 5e-6
    wl = np.linspace(1025, 1035, 48)
    for spacing in (1e-9, 10e-9, 50e-9):
        spec = ChirpedBurstSpec(
            burst_count=5,
            burst_spacing_s=spacing,
            packet_energy_j=5e-6,
            burst_start_time_s=200e-6,
            chirp_duration_s=2e-9,
        )
        t = build_cpa_time_grid(
            pump_duration_s=1e-3,
            spec=spec,
            pump_cw=True,
        )
        sig = build_chirped_burst(t, wl, spec)
        e_pkt = integrate_packet_energy(sig, t, wl, spec)
        assert abs(e_pkt - expected_pkt) / expected_pkt < 0.08, (
            f"spacing={spacing*1e9} ns: packet {e_pkt*1e6:.3f} µJ, expected {expected_pkt*1e6:.3f} µJ"
        )
        assert abs(packet_energy_expected_j(spec) - expected_pkt) < 1e-12


def test_single_pulse_energy_near_spec():
    """One pulse 1/e² integral ≈ energy_per_pulse_j."""
    spec = ChirpedBurstSpec(
        burst_count=5,
        burst_spacing_s=10e-9,
        packet_energy_j=5e-6,
        burst_start_time_s=200e-6,
        chirp_duration_s=2e-9,
    )
    wl = np.linspace(1025, 1035, 48)
    t = build_cpa_time_grid(pump_duration_s=1e-3, spec=spec, pump_cw=True)
    sig = build_chirped_burst(t, wl, spec)
    e1 = integrate_single_pulse_energy(sig, t, wl, spec, pulse_index=0)
    assert abs(e1 - 1e-6) / 1e-6 < 0.35  # flat packet: packet_energy_j / burst_count
