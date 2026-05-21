"""Unit tests for dopant and runtime calculators."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.calculators.dopant import estimate_dopant_concentration
from laser_sim.calculators.runtime import estimate_runtime, recommend_time_grid
from laser_sim.materials import YB_GLASS


def test_dopant_roundtrip_with_gamma():
    est = estimate_dopant_concentration(
        pump_absorption_db_per_m=6.0,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        pump_wavelength_nm=976.0,
        material=YB_GLASS,
        cladding_pumped=True,
    )
    alpha_check = est.concentration_m3 * est.sigma_abs_pump_m2 * est.gamma_pump
    assert abs(alpha_check - est.alpha_np_per_m) / est.alpha_np_per_m < 1e-6
    assert est.concentration_m3 > 1e20


def test_core_size_scales_concentration():
    """Fixed κ and cladding: larger core ⇒ larger Γ_p ⇒ lower N."""
    kwargs = dict(
        pump_absorption_db_per_m=17.0,
        cladding_diameter_um=400.0,
        pump_wavelength_nm=976.0,
        material=YB_GLASS,
        cladding_pumped=True,
    )
    n10 = estimate_dopant_concentration(core_diameter_um=10.0, **kwargs).concentration_m3
    n20 = estimate_dopant_concentration(core_diameter_um=20.0, **kwargs).concentration_m3
    assert n20 < n10
    assert abs(n20 / n10 - (10.0 / 20.0) ** 2) < 1e-6


def test_runtime_scales_with_grid():
    a = estimate_runtime(
        fiber_length_m=2.0,
        n_z=100,
        n_t=400,
        n_lambda=80,
        backend="cpu",
    )
    b = estimate_runtime(
        fiber_length_m=2.0,
        n_z=200,
        n_t=400,
        n_lambda=80,
        backend="cpu",
    )
    assert b.estimated_seconds > a.estimated_seconds


def test_recommend_time_covers_pump():
    t0, t1, nt = recommend_time_grid(
        pump_duration_s=2e-3,
        burst_span_s=100e-6,
        chirped_pulse_duration_s=3e-9,
        burst_count=10,
    )
    assert t1 > 2e-3
    assert nt > 100
