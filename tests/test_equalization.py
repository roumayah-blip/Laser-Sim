"""Packet equalization weights."""

from __future__ import annotations

import numpy as np

from laser_sim.physics.equalization import estimate_flat_packet_weights


def test_inverse_gain_weighting():
    e_in = np.array([1.0, 1.0, 1.0, 1.0, 1.0]) * 1e-6
    e_out = np.array([1.0, 2.0, 4.0, 8.0, 16.0]) * 1e-6
    w = estimate_flat_packet_weights(e_in, e_out, clip_ratio=1e6)
    assert abs(np.mean(w) - 1.0) < 1e-10
    ratios = w[:-1] / w[1:]
    assert all(abs(r - 2.0) < 0.01 for r in ratios)


def test_flat_input_gives_uniform_weights():
    e_in = e_out = np.ones(5) * 1e-6
    w = estimate_flat_packet_weights(e_in, e_out)
    assert np.allclose(w, 1.0)
