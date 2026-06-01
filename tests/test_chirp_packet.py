"""Tests for ns pulse-packet spacing and intensity superposition."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.pulses.chirp import (
    MIN_BURST_SPACING_S,
    ChirpedBurstSpec,
    build_chirped_burst,
)


def test_min_spacing_clamped():
    spec = ChirpedBurstSpec(burst_spacing_s=0.1e-9)
    assert spec.burst_spacing_s >= MIN_BURST_SPACING_S


def test_overlapping_pulses_sum_intensity():
    """Two pulses at 0.5 ns spacing with ~2 ns chirp overlap → higher peak than one."""
    chirp = 2e-9
    spacing = 0.5e-9
    t = np.linspace(0, 5e-9, 2000)
    wl = np.linspace(1028, 1032, 32)

    one = ChirpedBurstSpec(
        burst_count=1,
        chirp_duration_s=chirp,
        burst_spacing_s=spacing,
        packet_energy_j=1e-6,
    )
    two = ChirpedBurstSpec(
        burst_count=2,
        chirp_duration_s=chirp,
        burst_spacing_s=spacing,
        packet_energy_j=2e-6,
    )
    p1 = build_chirped_burst(t, wl, one)
    p2 = build_chirped_burst(t, wl, two)
    peak1 = p1.sum(axis=1).max()
    peak2 = p2.sum(axis=1).max()
    assert peak2 > peak1 * 1.2


def test_default_packet_spacing_1ns():
    spec = ChirpedBurstSpec()
    assert spec.burst_spacing_s == 1e-9
    assert spec.burst_count == 5
