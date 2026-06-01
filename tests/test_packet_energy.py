"""Packet-level energy normalisation."""

from __future__ import annotations

import numpy as np

from laser_sim.pulses.chirp import ChirpedBurstSpec, build_chirped_burst


def test_packet_energy_preserved():
    t = np.linspace(0, 50e-9, 5000)
    wl = np.linspace(1029, 1031, 32)
    spec = ChirpedBurstSpec(
        burst_count=5,
        packet_energy_j=50e-6,
        pulse_relative_powers=(0.5, 0.8, 1.0, 1.2, 1.5),
        burst_start_time_s=5e-9,
        chirp_duration_s=1e-9,
        burst_spacing_s=5e-9,
    )
    p = build_chirped_burst(t, wl, spec)
    dlam = np.gradient(wl)
    dt = np.gradient(t)
    e = float(np.sum(p * dlam[None, :] * dt[:, None]))
    assert abs(e - 50e-6) / 50e-6 < 0.01, f"Packet energy {e * 1e6:.3f} µJ != 50 µJ"


def test_flat_weights_give_flat_energy():
    spec = ChirpedBurstSpec(burst_count=4, packet_energy_j=40e-6)
    assert abs(spec.energy_per_pulse_j - 10e-6) < 1e-10
