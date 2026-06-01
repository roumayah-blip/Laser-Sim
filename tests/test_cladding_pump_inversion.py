"""Cladding-pumped steady-state inversion vs RP / first-principles brief."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.materials import load_material
from laser_sim.materials.base import cladding_area_m2, overlap_cladding_pump
from laser_sim.physics.four_level import (
    lifetimes_from_material,
    march_populations_pump_qss,
    pump_rate_per_ion,
    steady_state_n2_fraction_pump,
)


def test_steady_state_n2_10w_yb1200_25_250_matches_rp_brief():
    """Liekki 25/250, 10 W CW @ 976 nm → N₂/N_tot ≈ 40% at z=0 (RP benchmark)."""
    mat = load_material("yb_glass")
    r_core, r_clad = 12.5e-6, 125e-6
    gamma_p = overlap_cladding_pump(r_core, r_clad)
    a_clad = cladding_area_m2(r_clad)
    P = 10.0
    lam_p = 976.0

    sigma_p = float(mat.sigma_abs_at(lam_p)[0])
    sigma_ep = float(mat.sigma_em_at(lam_p)[0])
    hnu_p = float(mat.photon_energy_j(lam_p)[0])
    lt = lifetimes_from_material(mat)

    ip = P / a_clad
    w_abs, w_esa = pump_rate_per_ion(
        ip, gamma_p=gamma_p, sigma_p=sigma_p, sigma_ep=sigma_ep, hnu_p=hnu_p
    )
    n2_frac = steady_state_n2_fraction_pump(w_abs, w_esa, lt.tau_21_s)

    p_sat = hnu_p * a_clad / (sigma_p * lt.tau_21_s)
    n2_analytic = (P / p_sat) / (1.0 + (P / p_sat) * (1.0 + sigma_ep / sigma_p))

    assert n2_frac == pytest.approx(n2_analytic, rel=0.02)
    assert n2_frac == pytest.approx(0.403, rel=0.05)

    t = np.linspace(0.0, 5.0 * lt.tau_21_s, 80)
    p = np.full(t.size, P)
    n0, _, n2, _ = march_populations_pump_qss(
        t,
        p,
        np.zeros_like(p),
        n_tot=1.0e26,
        a_pump=a_clad,
        gamma_p=gamma_p,
        sigma_p=sigma_p,
        sigma_ep=sigma_ep,
        hnu_p=hnu_p,
        lifetimes=lt,
    )
    assert n2[-1] / 1.0e26 == pytest.approx(n2_frac, rel=0.05)
