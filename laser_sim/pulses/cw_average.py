"""CW signal at the time-averaged power of a chirped pulse packet."""

from __future__ import annotations

import numpy as np

from laser_sim.pulses.chirp import (
    ChirpedBurstSpec,
    _spectral_weights,
    packet_duration_s,
    packet_energy_expected_j,
    rep_period_s,
)


def packet_average_power_w(spec: ChirpedBurstSpec) -> float:
    """
    Mean signal power (W) equivalent to the pulse packet.

    Non-rep-rate: E_packet / T_packet_span.
    Rep-rate: E_pulse × rep_rate (duty cycle).
    """
    if spec.rep_rate_mode:
        return float(spec.energy_per_pulse_j) * float(spec.rep_rate_hz)
    e_pkt = packet_energy_expected_j(spec)
    effective_period = max(
        packet_duration_s(spec),
        float(spec.burst_spacing_s),
        10.0 * float(spec.chirp_duration_s),
    )
    return e_pkt / effective_period


def build_cw_average_signal(
    time_s: np.ndarray,
    wavelength_nm: np.ndarray,
    spec: ChirpedBurstSpec,
) -> np.ndarray:
    """
    Constant-in-time P(t, λ) in W/nm with ∫ P(t,λ) dλ ≈ P_avg at every t.
    """
    t = np.asarray(time_s, dtype=np.float64)
    wl = np.asarray(wavelength_nm, dtype=np.float64)
    p_avg = packet_average_power_w(spec)
    sw = _spectral_weights(wl, spec)
    dlam = np.gradient(wl)
    dlam = np.where(np.abs(dlam) < 1e-12, 1e-12, dlam)
    row = p_avg * sw / dlam
    return np.outer(np.ones(t.size, dtype=np.float64), row)
