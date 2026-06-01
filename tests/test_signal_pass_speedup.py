"""Batch RK4 matches scalar march_level_interval."""

from __future__ import annotations

import numpy as np

from laser_sim.materials.yb_glass import YB_GLASS
from laser_sim.physics.four_level import (
    lifetimes_from_material,
    march_level_interval,
    march_level_interval_batch,
)


def test_march_batch_matches_scalar():
    lt = lifetimes_from_material(YB_GLASS)
    n_tot = 9.34e25
    n0_arr = np.array([0.66, 0.70]) * n_tot
    n2_arr = np.array([0.33, 0.29]) * n_tot
    nlam = 16
    wl = np.linspace(1029, 1031, nlam)
    sigma_e = np.full(nlam, 6.43e-25)
    sigma_a = np.full(nlam, 4.53e-26)
    hnu = 6.626e-34 * 3e8 / (wl * 1e-9)
    dlam = np.gradient(wl)
    p_sig = np.full((2, nlam), 100.0 / nlam)
    ip = np.array([1e9, 1e9])
    a = 7.07e-10
    dt = 9.69e-12
    hnu_p = 6.626e-34 * 3e8 / 976e-9

    n0b, n2b = march_level_interval_batch(
        n0_arr,
        n2_arr,
        n_tot,
        dt,
        ip_batch=ip,
        p_sig_batch=p_sig,
        dlam=dlam,
        sigma_e=sigma_e,
        sigma_a=sigma_a,
        hnu=hnu,
        gamma_p=0.0144,
        gamma_s=0.769,
        sigma_p=2.5e-24,
        sigma_ep=0.0,
        hnu_p=hnu_p,
        a_signal=a,
        lifetimes=lt,
    )

    for i in range(2):
        n0s, n2s, _ = march_level_interval(
            float(n0_arr[i]),
            float(n2_arr[i]),
            0.0,
            n_tot,
            dt,
            ip_w_m2=float(ip[i]),
            p_sig_row=p_sig[i],
            dlam=dlam,
            sigma_e=sigma_e,
            sigma_a=sigma_a,
            hnu=hnu,
            gamma_p=0.0144,
            gamma_s=0.769,
            sigma_p=2.5e-24,
            sigma_ep=0.0,
            hnu_p=hnu_p,
            a_signal=a,
            lifetimes=lt,
            include_signal=True,
        )
        assert abs(n0b[i] / n0s - 1) < 1e-5, f"N0 mismatch bin {i}"
        assert abs(n2b[i] / n2s - 1) < 1e-5, f"N2 mismatch bin {i}"
