"""Quasi-two-level rate equation tests."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.materials import YB_GLASS
from laser_sim.physics.four_level import (
    FourLevelLifetimes,
    level_derivatives,
    lifetimes_from_material,
    march_level_interval,
    march_populations_pump_qss,
)


def test_conservation_after_march():
    lt = lifetimes_from_material(YB_GLASS)
    n_tot = 6e24
    n0, n2, n3 = march_level_interval(
        n_tot,
        0.0,
        0.0,
        n_tot,
        50e-6,
        ip_w_m2=5e9,
        p_sig_row=np.zeros(8),
        dlam=np.ones(8) * 1e-9,
        sigma_e=np.ones(8) * 2e-25,
        sigma_a=np.ones(8) * 2e-25,
        hnu=np.ones(8) * 2e-19,
        gamma_p=0.01,
        gamma_s=1.0,
        sigma_p=2e-25,
        sigma_ep=1e-26,
        hnu_p=2e-19,
        a_signal=1e-11,
        lifetimes=lt,
        include_signal=False,
    )
    assert abs(n0 + n2 + n3 - n_tot) / n_tot < 1e-6
    assert n2 > 0


def test_stimulated_emission_depletes_n2_quasi2l():
    """With inversion (N₂ > N₀), strong signal depletes N₂ via stimulated emission."""
    lt = lifetimes_from_material(YB_GLASS)
    n_tot = 6e24
    n2_0 = 0.6 * n_tot
    n0_0 = 0.4 * n_tot
    dlam = np.array([1e-9])
    sig = np.array([1e9])  # strong field for measurable depletion in one step
    n0, n2, n3 = march_level_interval(
        n0_0,
        n2_0,
        0.0,
        n_tot,
        10e-6,
        ip_w_m2=0.0,
        p_sig_row=sig,
        dlam=dlam,
        sigma_e=np.array([2e-25]),
        sigma_a=np.array([2e-25]),
        hnu=np.array([2e-19]),
        gamma_p=0.01,
        gamma_s=1.0,
        sigma_p=2e-25,
        sigma_ep=0.0,
        hnu_p=2e-19,
        a_signal=1e-11,
        lifetimes=lt,
        include_signal=True,
    )
    assert n2 < n2_0
    assert n0 > n0_0


def test_qss_high_pump_saturation():
    """N2 must not exceed N_tot even when W·τ >> 1."""
    lt = lifetimes_from_material(YB_GLASS)
    n_tot = 1.635e25
    sigma_p = 2.4e-25
    gamma_p = 1.0
    hnu_p = 6.626e-34 * 3e8 / 976e-9
    a_core = np.pi * (15e-6) ** 2
    p_pump = 100.0
    t = np.linspace(0, 10e-3, 1000)  # >> τ₂₁ so QSS reaches W·τ/(1+W·τ) limit
    p_pf = np.ones(1000) * p_pump
    p_pb = np.zeros(1000)
    n0, n1, n2, n3 = march_populations_pump_qss(
        t,
        p_pf,
        p_pb,
        n_tot=n_tot,
        a_pump=a_core,
        gamma_p=gamma_p,
        sigma_p=sigma_p,
        sigma_ep=0.0,
        hnu_p=hnu_p,
        lifetimes=lt,
    )
    assert np.all(n2 <= n_tot * 1.001), f"N2 exceeded N_tot: max N2/N_tot = {(n2/n_tot).max():.4f}"
    assert np.all(n0 >= 0), "N0 went negative"
    assert np.all(n2 >= 0), "N2 went negative"
    expected_frac = 150.0 / 151.0
    final_frac = n2[-1] / n_tot
    assert abs(final_frac - expected_frac) < 0.05, (
        f"N2/N_tot={final_frac:.4f}, expected≈{expected_frac:.4f}"
    )


def test_derivatives_sum_to_zero():
    lt = FourLevelLifetimes(1e-12, 840e-6, 1e-9, skip_n1_level=True)
    n_tot = 1e25
    dn0, dn2, dn3 = level_derivatives(
        1e22,
        1e20,
        1e18,
        n_tot,
        w_p_abs=100.0,
        w_p_esa=1.0,
        w_se=1e3,
        w_abs=1e2,
        lifetimes=lt,
    )
    assert abs(dn0 + dn2 + dn3) < 1e-6 * n_tot
