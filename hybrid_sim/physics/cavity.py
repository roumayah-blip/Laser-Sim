"""Round-trip cavity dynamics for pump laser source."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_sim.constants import C0, H
from hybrid_sim.materials.base import Material
from hybrid_sim.physics.four_level import (
    FourLevelLifetimes,
    lifetimes_from_material,
    pump_rate_per_ion,
)


@dataclass
class CavityConfig:
    crystal: Material
    crystal_length_m: float
    cavity_length_m: float
    yb_concentration_m3: float = 1.0e26
    r_oc: float = 0.70
    r_hr: float = 0.999
    internal_loss_per_pass: float = 0.01
    q_switch_on_time_s: float | None = None
    q_switch_holdoff_loss: float = 0.95
    pump_power_w: float = 300.0
    pump_duration_s: float = 1e-3
    pump_wavelength_nm: float | None = None
    n_roundtrips: int = 10_000
    seed_energy_j: float = 1e-15
    gamma_s: float = 1.0


@dataclass
class CavityResult:
    t_s: np.ndarray
    e_intra_j: np.ndarray
    e_output_j: np.ndarray
    n2_fraction: np.ndarray
    g0_per_pass: np.ndarray
    pulse_energy_j: float
    pulse_peak_power_w: float
    pulse_duration_fwhm_s: float
    threshold_roundtrip: int
    round_trip_time_s: float


def _effective_r_oc(cfg: CavityConfig, t_rt: float) -> float:
    if cfg.q_switch_on_time_s is None:
        return cfg.r_oc
    return cfg.r_oc if t_rt >= cfg.q_switch_on_time_s else cfg.r_oc * (1.0 - cfg.q_switch_holdoff_loss)


def frantz_nodvik_energy(
    *,
    g0: float,
    length_m: float,
    sigma_e: float,
    sigma_a: float,
    area_m2: float,
    wavelength_m: float,
    e_in_j: float,
) -> float:
    """Frantz-Nodvik output energy estimate."""
    hnu = H * C0 / wavelength_m
    e_sat = hnu * area_m2 / max(sigma_e + sigma_a, 1e-30)
    G0 = np.exp(g0 * length_m)
    return e_sat * np.log(1.0 + G0 * (np.exp(e_in_j / max(e_sat, 1e-30)) - 1.0))


def run_cavity_simulation(cfg: CavityConfig) -> CavityResult:
    """Causal round-trip march for cavity pump source."""
    mat = cfg.crystal
    pump_nm = cfg.pump_wavelength_nm or mat.default_pump_wavelength_nm
    sig_nm = mat.default_signal_wavelength_nm
    sigma_p = float(mat.sigma_abs_at(pump_nm)[0])
    sigma_e = float(mat.sigma_em_at(sig_nm)[0])
    sigma_a = float(mat.sigma_abs_at(sig_nm)[0])
    hnu_p = float(mat.photon_energy_j(pump_nm)[0])
    hnu_s = float(mat.photon_energy_j(sig_nm)[0])
    n_tot = cfg.yb_concentration_m3
    L = cfg.crystal_length_m
    area = np.pi * (0.5 * 0.01) ** 2  # 10 mm diameter default mode
    lt = lifetimes_from_material(mat)
    if not mat.skip_n1_level:
        lt = FourLevelLifetimes(
            tau_32_s=lt.tau_32_s,
            tau_21_s=lt.tau_21_s,
            tau_10_s=lt.tau_10_s,
            beta_21=lt.beta_21,
            skip_n1_level=False,
        )

    T_rt = 2.0 * cfg.cavity_length_m / C0
    n_rt = int(cfg.n_roundtrips)
    t = np.arange(n_rt, dtype=np.float64) * T_rt
    e_intra = np.zeros(n_rt, dtype=np.float64)
    e_out = np.zeros(n_rt, dtype=np.float64)
    n2_frac = np.zeros(n_rt, dtype=np.float64)
    g0_pass = np.zeros(n_rt, dtype=np.float64)

    n2 = 0.0
    e_intra[0] = cfg.seed_energy_j
    threshold_rt = n_rt

    ip = cfg.pump_power_w / area
    w_p_abs, w_p_esa = pump_rate_per_ion(
        ip, gamma_p=1.0, sigma_p=sigma_p, sigma_ep=sigma_p * 0.1, hnu_p=hnu_p
    )
    _wt = w_p_abs * lt.tau_21_s
    n2_ss_pump = _wt / (1.0 + _wt)

    for k in range(n_rt - 1):
        t_k = t[k]
        r_oc = _effective_r_oc(cfg, t_k)
        loss = (1.0 - r_oc) + cfg.internal_loss_per_pass
        loss = min(max(loss, 1e-6), 0.99)

        n2 = n2_ss_pump - (n2_ss_pump - n2) * np.exp(-T_rt / lt.tau_21_s)
        n0_frac = max(1.0 - n2, 0.0)
        if mat.skip_n1_level:
            g0 = cfg.gamma_s * (sigma_e * n2 - sigma_a * n0_frac) * n_tot
        else:
            n1 = max(1.0 - n0_frac - n2, 0.0)
            g0 = cfg.gamma_s * (sigma_e * n2 - sigma_a * n1) * n_tot
        g0_pass[k] = g0
        G_rt = np.exp(2.0 * g0 * L)
        n2_frac[k] = n2

        if G_rt * (1.0 - loss) > 1.0 and threshold_rt == n_rt:
            threshold_rt = k

        e_intra[k + 1] = e_intra[k] * G_rt * (1.0 - loss) + cfg.seed_energy_j
        e_out[k] = e_intra[k] * (1.0 - r_oc)

        deplete = (G_rt - 1.0) * e_intra[k] / (hnu_s * n_tot * area * L)
        n2 = max(0.0, n2 - deplete / max(n_tot, 1.0))

    pulse_energy = float(np.sum(e_out))
    if pulse_energy <= 0:
        pulse_energy = float(np.max(e_out))
    peak_power = float(np.max(e_out) / max(T_rt, 1e-30))
    above = e_out > 0.5 * np.max(e_out)
    if np.any(above):
        idx = np.where(above)[0]
        fwhm_s = (idx[-1] - idx[0] + 1) * T_rt
    else:
        fwhm_s = T_rt

    return CavityResult(
        t_s=t,
        e_intra_j=e_intra,
        e_output_j=e_out,
        n2_fraction=n2_frac,
        g0_per_pass=g0_pass,
        pulse_energy_j=pulse_energy,
        pulse_peak_power_w=peak_power,
        pulse_duration_fwhm_s=fwhm_s,
        threshold_roundtrip=threshold_rt,
        round_trip_time_s=T_rt,
    )
