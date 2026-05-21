"""B-integral and packet power-shaping tests."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def test_passive_fiber_b():
    from laser_sim.physics.b_integral import compute_b_integral

    lam = 1030e-9
    n2 = 2.6e-20
    a_eff = 100e-12
    p_peak = 1000.0
    l_passive = 1.0
    expected = 2 * np.pi / lam * n2 * p_peak / a_eff * l_passive
    r = compute_b_integral(
        wavelength_m=lam,
        a_eff_m2=a_eff,
        p_peak_in_w=p_peak,
        p_peak_out_w=p_peak,
        l_active_m=0.0,
        l_passive_before_m=l_passive,
    )
    assert abs(r.b_total_rad - expected) / expected < 1e-6


def test_pulse_weights_normalization():
    from laser_sim.pulses.chirp import ChirpedBurstSpec, _get_pulse_weights

    spec = ChirpedBurstSpec(
        burst_count=5,
        pulse_relative_powers=(1.0, 1.0, 1.2, 1.4, 1.6),
    )
    w = _get_pulse_weights(spec)
    assert abs(np.mean(w) - 1.0) < 1e-10
    assert abs(np.sum(w) - 5.0) < 1e-10
    assert w[4] > w[0]


def test_pulse_weights_flat():
    from laser_sim.pulses.chirp import ChirpedBurstSpec, _get_pulse_weights

    spec = ChirpedBurstSpec(burst_count=3, pulse_relative_powers=None)
    w = _get_pulse_weights(spec)
    assert np.allclose(w, 1.0)


def test_wrong_weight_count_raises():
    from laser_sim.pulses.chirp import ChirpedBurstSpec

    try:
        ChirpedBurstSpec(burst_count=5, pulse_relative_powers=(1.0, 1.0, 1.0))
        raise AssertionError("should have raised")
    except ValueError:
        pass


def test_b_severity_thresholds():
    from laser_sim.physics.b_integral import compute_b_integral

    for b_target, expected_sev in [
        (0.5, "excellent"),
        (2.0, "moderate"),
        (4.0, "significant"),
        (6.0, "severe"),
    ]:
        lam = 1030e-9
        n2 = 2.6e-20
        a_eff = 100e-12
        p_peak = b_target * lam * a_eff / (2 * np.pi * n2 * 1.0)
        r = compute_b_integral(
            wavelength_m=lam,
            a_eff_m2=a_eff,
            p_peak_in_w=p_peak,
            p_peak_out_w=p_peak,
            l_active_m=0.0,
            l_passive_before_m=1.0,
        )
        assert r.severity == expected_sev, (
            f"B={r.b_total_rad:.2f} got {r.severity}, want {expected_sev}"
        )
