"""Solid-state 2D amplifier: pump inversion, thermal, BPM signal pass."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_sim.constants import C0, H, NM_TO_M
from hybrid_sim.materials.base import Material
from hybrid_sim.physics.bpm import (
    angular_spectrum_propagator,
    beam_quality_m2,
    bpm_step,
)
from hybrid_sim.physics.cavity import CavityConfig, CavityResult, run_cavity_simulation
from hybrid_sim.physics.four_level import (
    advance_pump_power_z,
    lifetimes_from_material,
    march_populations_pump_qss,
    pump_rate_per_ion,
)
from hybrid_sim.physics.gain_2d import gain_coefficient_xy
from hybrid_sim.physics.kerr import b_integral_2d, check_b_integral, kerr_phase_2d
from hybrid_sim.physics.thermal import dn_thermal_field, solve_heat_2d, thermal_lens_focal_length
from hybrid_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec, build_cpa_time_grid


@dataclass
class SolidAmplifierConfig:
    material: Material
    crystal_length_m: float
    crystal_diameter_m: float
    yb_concentration_m3: float = 1.38e26
    pump_pulse: PumpPulseSpec | None = None
    cavity: CavityConfig | None = None
    signal: ChirpedBurstSpec | None = None
    pump_wavelength_nm: float | None = None
    signal_wavelength_nm: float | None = None
    beam_waist_m: float = 5e-4
    pump_waist_m: float = 6e-4
    n_z: int = 100
    n_x: int = 64
    n_y: int = 64
    include_thermal: bool = True
    include_kerr: bool = True
    include_fluorescence: bool = True
    n_lam: int = 32
    wavelength_min_nm: float = 1020.0
    wavelength_max_nm: float = 1040.0
    gamma_s: float = 1.0
    gamma_p: float = 1.0


@dataclass
class SolidAmplifierResult:
    U_out: np.ndarray
    spectrum_out: np.ndarray
    wl_nm: np.ndarray
    N0_xyz: np.ndarray
    N2_xyz: np.ndarray
    z_m: np.ndarray
    t_s: np.ndarray
    x_m: np.ndarray
    y_m: np.ndarray
    energy_signal_in_j: float
    energy_signal_out_j: float
    energy_pump_in_j: float
    energy_pump_absorbed_j: float
    energy_fluorescence_j: float
    energy_heat_j: float
    M2_x: float
    M2_y: float
    beam_waist_out_m: float
    T_rise_max_k: float
    thermal_lens_f_m: float
    B_integral_max: float
    b_integral_warning: str
    cavity_result: CavityResult | None = None
    pump_absorbed_frac: float = 0.0
    g0_db_per_mm: float = 0.0


def _gaussian_xy(xx: np.ndarray, yy: np.ndarray, w: float) -> np.ndarray:
    return np.exp(-2.0 * (xx**2 + yy**2) / max(w**2, 1e-30))


def _build_pump_envelope(cfg: SolidAmplifierConfig, t: np.ndarray) -> np.ndarray:
    if cfg.cavity is not None:
        cav = run_cavity_simulation(cfg.cavity)
        p_out = np.interp(
            t,
            cav.t_s,
            cav.e_output_j / max(cav.round_trip_time_s, 1e-30),
            left=0.0,
            right=0.0,
        )
        return p_out, cav
    if cfg.pump_pulse is not None:
        from hybrid_sim.pulses.chirp import build_pump_power

        return build_pump_power(t, cfg.pump_pulse), None
    return np.full(t.size, 100.0), None


def run_solid_amplifier(cfg: SolidAmplifierConfig) -> SolidAmplifierResult:
    mat = cfg.material
    pump_nm = cfg.pump_wavelength_nm or mat.default_pump_wavelength_nm
    sig_nm = cfg.signal_wavelength_nm or mat.default_signal_wavelength_nm
    n_tot = cfg.yb_concentration_m3
    L = cfg.crystal_length_m
    nz = max(cfg.n_z, 2)
    nx, ny = cfg.n_x, cfg.n_y
    dz = L / (nz - 1)
    aperture = max(4.0 * cfg.beam_waist_m, cfg.crystal_diameter_m * 0.5)
    dx = aperture / nx
    dy = aperture / ny
    z = np.linspace(0.0, L, nz)
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    pump_profile = _gaussian_xy(xx, yy, cfg.pump_waist_m)
    sig_profile = _gaussian_xy(xx, yy, cfg.beam_waist_m)
    a_pump = np.sum(pump_profile) * dx * dy
    a_signal = np.pi * cfg.beam_waist_m**2

    sigma_p = float(mat.sigma_abs_at(pump_nm)[0])
    sigma_ep = float(mat.sigma_abs_at(pump_nm)[0]) * 0.05
    sigma_e = float(mat.sigma_em_at(sig_nm)[0])
    sigma_a = float(mat.sigma_abs_at(sig_nm)[0])
    hnu_p = float(mat.photon_energy_j(pump_nm)[0])
    hnu_s = float(mat.photon_energy_j(sig_nm)[0])
    lt = lifetimes_from_material(mat)
    eta_heat = 1.0 - (pump_nm * NM_TO_M) / (sig_nm * NM_TO_M)

    pump_dur = 1e-3
    if cfg.pump_pulse is not None:
        pump_dur = cfg.pump_pulse.duration_s
    elif cfg.cavity is not None:
        pump_dur = cfg.cavity.pump_duration_s
    sig_spec = cfg.signal or ChirpedBurstSpec(
        burst_start_time_s=200e-6,
        chirp_duration_s=2e-9,
        packet_energy_j=10e-9,
    )
    t = build_cpa_time_grid(
        pump_duration_s=pump_dur,
        spec=sig_spec,
    )
    pump_env, cavity_res = _build_pump_envelope(cfg, t)
    nt = t.size

    wl = np.linspace(cfg.wavelength_min_nm, cfg.wavelength_max_nm, cfg.n_lam)
    dlam = float(wl[1] - wl[0]) if wl.size > 1 else 1.0
    sigma_e_lam = mat.sigma_em_at(wl)
    sigma_a_lam = mat.sigma_abs_at(wl)
    hnu_lam = mat.photon_energy_j(wl)

    # Pass A: pump inversion (z, t) mean over transverse Gaussian weight
    n0 = np.zeros((nz, nt), dtype=np.float64)
    n2 = np.zeros((nz, nt), dtype=np.float64)
    n3 = np.zeros((nz, nt), dtype=np.float64)
    P_pump = np.zeros((nz, nt), dtype=np.float64)
    Q_heat = np.zeros((nz, nx, ny), dtype=np.float64)

    for it in range(nt):
        P_pump[0, it] = pump_env[it]

    for iz in range(nz):
        p_pf = P_pump[iz]
        p_pb = P_pump[iz]
        n0_i, _, n2_i, n3_i = march_populations_pump_qss(
            t,
            p_pf,
            p_pb,
            n_tot=n_tot,
            a_pump=a_pump,
            gamma_p=cfg.gamma_p,
            sigma_p=sigma_p,
            sigma_ep=sigma_ep,
            hnu_p=hnu_p,
            lifetimes=lt,
            initial_n2_fraction=float(n2[iz, 0] / n_tot) if iz == 0 else float(n2[iz - 1, -1] / n_tot),
        )
        n0[iz] = n0_i
        n2[iz] = n2_i
        n3[iz] = n3_i
        if iz < nz - 1:
            P_pump[iz + 1] = advance_pump_power_z(
                P_pump[iz], n0[iz], sigma_p, dz, gamma_p=cfg.gamma_p
            )
        alpha_p = cfg.gamma_p * sigma_p * n0[iz] / n_tot
        I_p = (P_pump[iz, :, None, None] / a_pump) * pump_profile[None, :, :]
        Q_heat[iz] += np.trapezoid(alpha_p[:, None, None] * I_p * eta_heat, t, axis=0)

    pump_in = float(np.trapezoid(pump_env, t))
    pump_out = float(np.trapezoid(P_pump[-1], t))
    pump_abs_frac = 1.0 - pump_out / max(pump_in, 1e-30)
    energy_pump_in = pump_in
    energy_pump_abs = pump_in * pump_abs_frac

    n0_mid = n0[nz // 2, -1] / n_tot
    n2_mid = n2[nz // 2, -1] / n_tot

    # Pass B: thermal
    T_rise = np.zeros((nz, nx, ny), dtype=np.float64)
    dn_th = np.zeros((nz, nx, ny), dtype=np.float64)
    if cfg.include_thermal and mat.thermal_cond_w_mk > 0:
        T_rise = solve_heat_2d(
            Q_heat,
            dt_s=pump_dur,
            kappa_t=mat.thermal_cond_w_mk,
            rho=mat.density_kg_m3,
            cp=mat.cp_j_kg_k,
        )
        dn_th = dn_thermal_field(mat.dndt_per_k, T_rise)
    f_th = thermal_lens_focal_length(
        mat.dndt_per_k,
        T_rise[nz // 2],
        dx,
        dy,
        sig_nm * NM_TO_M,
        L,
    )
    T_max = float(np.max(T_rise))

    # Pass C: signal BPM (single λ channel, representative)
    wavelength_m = sig_nm * NM_TO_M
    k0 = 2.0 * np.pi * mat.n_group / wavelength_m
    H = angular_spectrum_propagator(nx, ny, dx, dy, dz, wavelength_m, mat.n_group)
    U = np.zeros((nx, ny), dtype=np.complex128)
    U += np.sqrt(cfg.signal.packet_energy_j / max(a_signal * np.sum(sig_profile**2) * dx * dy, 1e-30)) * sig_profile

    intensity_z = np.zeros((nz, nx, ny), dtype=np.float64)
    intensity_z[0] = np.abs(U) ** 2

    n0_frac = n0 / n_tot
    n2_frac = n2 / n_tot
    g0_center = float(
        gain_coefficient_xy(
            np.array([n0_frac[nz // 2, -1]]),
            np.array([n2_frac[nz // 2, -1]]),
            sigma_a,
            sigma_e,
            gamma_s=cfg.gamma_s,
            n_tot=1.0,
        )[0]
    )
    g0_db_mm = g0_center * 10.0 / np.log(10.0) / 1000.0

    for iz in range(nz - 1):
        n0_xy = n0_frac[iz, -1] * pump_profile
        n2_xy = n2_frac[iz, -1] * pump_profile
        g_xy = gain_coefficient_xy(n0_xy, n2_xy, sigma_a, sigma_e, gamma_s=cfg.gamma_s, n_tot=1.0)
        g_half = 0.5 * g_xy * dz
        I_loc = np.abs(U) ** 2
        phase = np.zeros((nx, ny), dtype=np.float64)
        if cfg.include_kerr and mat.n2_m2_per_w > 0:
            phase += kerr_phase_2d(I_loc, mat.n2_m2_per_w, k0, dz)
        if cfg.include_thermal:
            phase += k0 * dn_th[iz] * dz
        U = bpm_step(U, H, g_half, phase)
        intensity_z[iz + 1] = np.abs(U) ** 2

    B_map = b_integral_2d(intensity_z, mat.n2_m2_per_w, k0, dz)
    B_max = float(np.max(B_map))
    b_warn = check_b_integral(B_max)
    M2_x, M2_y = beam_quality_m2(U, dx, dy, wavelength_m)
    w_out = cfg.beam_waist_m * np.sqrt(M2_x)

    e_in = float(cfg.signal.packet_energy_j)
    e_out = float(np.sum(np.abs(U) ** 2) * dx * dy)
    spectrum = np.abs(U) ** 2
    spectrum = spectrum / max(np.sum(spectrum), 1e-30) * e_out / dlam

    fluor = float(np.mean(n2_frac) * n_tot * hnu_s / lt.tau_21_s * L * a_signal * pump_dur)
    e_heat = energy_pump_abs * eta_heat

    return SolidAmplifierResult(
        U_out=U,
        spectrum_out=spectrum,
        wl_nm=wl,
        N0_xyz=np.stack([n0_frac[iz, -1] * pump_profile for iz in range(nz)]),
        N2_xyz=np.stack([n2_frac[iz, -1] * pump_profile for iz in range(nz)]),
        z_m=z,
        t_s=t,
        x_m=x,
        y_m=y,
        energy_signal_in_j=e_in,
        energy_signal_out_j=e_out,
        energy_pump_in_j=energy_pump_in,
        energy_pump_absorbed_j=energy_pump_abs,
        energy_fluorescence_j=fluor,
        energy_heat_j=e_heat,
        M2_x=M2_x,
        M2_y=M2_y,
        beam_waist_out_m=w_out,
        T_rise_max_k=T_max,
        thermal_lens_f_m=f_th,
        B_integral_max=B_max,
        b_integral_warning=b_warn,
        cavity_result=cavity_res,
        pump_absorbed_frac=pump_abs_frac,
        g0_db_per_mm=g0_db_mm,
    )
