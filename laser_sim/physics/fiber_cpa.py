"""
Spectrally resolved, time-resolved Yb fiber CPA amplifier.

Quasi-two-level (N₁ skipped, τ₁₀→0): N₀, N₃, N₂.
  • Datasheet κ (dB/m) sets N_tot only (via dopant calculator).
  • Pump depletion in z: α_p = σ_p · N₀(z,t) from rate equations.
  • Signal gain: g = Γ (σ_e N₂ − σ_a N₀).
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
import numpy as np

from laser_sim.physics.progress import ProgressCallback, emit_progress

from laser_sim.constants import C0
from laser_sim.materials.base import (
    Material,
    cladding_area_m2,
    core_area_m2,
    overlap_cladding_pump,
)
from laser_sim.physics.four_level import (
    FourLevelLifetimes,
    FourLevelPopulations,
    advance_pump_power_z,
    gain_coefficient_m,
    lifetimes_from_material,
    march_level_interval,
    march_populations_pump_qss,
    populations_to_fractions,
    spontaneous_power_w_per_nm,
)
from laser_sim.physics.modes import (
    guided_spontaneous_fraction,
    signal_overlap_gamma,
    v_number,
)
from laser_sim.pulses.chirp import (
    ChirpedBurstSpec,
    PumpPulseSpec,
    build_chirped_signal,
    build_cpa_time_grid,
    build_pump_power,
    integrate_packet_energy,
    integrate_single_pulse_energy,
    packet_energy_expected_j,
)
from laser_sim.physics.rep_rate import analyze_rep_rate_steady_state


@dataclass
class FiberCPAConfig:
    material: Material
    fiber_length_m: float = 2.0
    core_diameter_um: float = 10.0
    cladding_diameter_um: float = 400.0
    cladding_pumped: bool = True
    core_na: float = 0.06
    yb_concentration_m3: float = 5e25
    pump_absorption_db_per_m: float | None = None
    # INFORMATIONAL ONLY after N_tot is set. Never used in physics kernels.
    # Physics uses α_p(z,t) = γ_p · σ_p · N₀(z,t) from the rate equations.
    n_z: int = 150
    time_s: np.ndarray | None = None
    wavelength_nm: np.ndarray | None = None
    signal: ChirpedBurstSpec = field(default_factory=ChirpedBurstSpec)
    pump: PumpPulseSpec = field(default_factory=PumpPulseSpec)
    forward_pump: bool = True
    backward_pump: bool = False
    backward_pump_fraction: float = 0.0
    include_ase: bool = True
    initial_n2_fraction: float = 0.0
    initial_n2_fraction_z: np.ndarray | None = None
    four_level: FourLevelLifetimes | None = None
    use_group_velocity_walkoff: bool = False
    use_nonlinear: bool = False
    lp_modes: list | None = None
    signal_channels: list = field(default_factory=list)
    pump_channels: list = field(default_factory=list)


@dataclass
class FiberCPAResult:
    z_m: np.ndarray
    t_s: np.ndarray
    wavelength_nm: np.ndarray
    pump_fwd_w: np.ndarray
    pump_bwd_w: np.ndarray
    signal_fwd_w_nm: np.ndarray
    ase_fwd_w_nm: np.ndarray
    ase_bwd_w_nm: np.ndarray
    n2_fraction: np.ndarray
    populations: FourLevelPopulations
    pump_absorption_db_per_m: np.ndarray
    sigma_abs_signal_m2: np.ndarray
    sigma_em_signal_m2: np.ndarray
    sigma_abs_pump_m2: float
    sigma_em_pump_m2: float
    pump_wavelength_nm: float
    kappa_datasheet_np_m: float
    energy_pump_in_j: float
    energy_pump_out_j: float
    energy_pulse_in_j: float
    energy_pulse_out_j: float
    energy_packet_in_j: float
    energy_packet_out_j: float
    energy_packet_expected_j: float
    energy_ase_out_j: float
    pump_power_absorbed_fraction: float
    notes: str
    energy_unguided_spont_j: float = 0.0
    energy_balance_ok: bool = True
    energy_balance_residual_j: float = 0.0
    ase_fraction_of_emission: float = 0.0
    steady_state_reached: bool = False
    steady_state_metric: float = float("nan")
    rep_rate_hz: float | None = None
    n_periods_simulated: int = 0
    dt_travel_s: float = 0.0
    four_level_lifetimes: FourLevelLifetimes | None = None
    gamma_signal: float = 1.0
    eta_guided_spontaneous: float = 0.0
    v_number: float = 0.0
    signal_mode_area_m2: float = 0.0
    a_signal_m2: float = 0.0
    pulse_energies_in_j: np.ndarray | None = None
    pulse_energies_out_j: np.ndarray | None = None
    g0_small_signal_np_m: np.ndarray | None = None
    g0_small_signal_db_m: np.ndarray | None = None
    lp_modes: list | None = None
    gamma_signal_per_mode: np.ndarray | None = None
    populations_after_pump: FourLevelPopulations | None = None
    signal_channels_info: list = field(default_factory=list)
    pump_channels_info: list = field(default_factory=list)


def _multichannel_mode(cfg: FiberCPAConfig) -> bool:
    return bool(cfg.signal_channels) or bool(cfg.pump_channels)


def _build_grids(cfg: FiberCPAConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if cfg.time_s is None:
        t = build_cpa_time_grid(
            pump_duration_s=cfg.pump.duration_s,
            spec=cfg.signal,
            pump_cw=cfg.pump.cw,
        )
    else:
        t = np.asarray(cfg.time_s, dtype=np.float64)

    if cfg.wavelength_nm is None:
        half = max(cfg.signal.bandwidth_nm * 0.6, 2.0)
        wl = np.linspace(
            cfg.signal.center_wavelength_nm - half,
            cfg.signal.center_wavelength_nm + half,
            max(int(3 * cfg.signal.bandwidth_nm), 48),
        )
    else:
        wl = np.asarray(cfg.wavelength_nm, dtype=np.float64)

    z = np.linspace(0.0, cfg.fiber_length_m, cfg.n_z)
    return z, t, wl


def _integrate_pump_energy(pump_w: np.ndarray, t_s: np.ndarray) -> float:
    return float(np.sum(pump_w * np.gradient(t_s)))


def _pump_pass(
    *,
    z: np.ndarray,
    t: np.ndarray,
    p_p_in: np.ndarray,
    n_tot: float,
    dz: float,
    sigma_p: float,
    gamma_p: float,
    sigma_ep: float,
    hnu_p: float,
    a_pump: float,
    lifetimes: FourLevelLifetimes,
    forward_pump: bool,
    backward_pump: bool,
    backward_fraction: float,
    progress_callback: ProgressCallback | None = None,
    progress_base: float = 0.05,
    progress_span: float = 0.30,
    initial_n2_fraction: float = 0.0,
    initial_n2_fraction_z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build N₀,N₂,N₃(z,t) from pump rates; advance P_p with α_p = σ_p N₀."""
    nz, nt = z.size, t.size
    init_z = (
        np.asarray(initial_n2_fraction_z, dtype=np.float64)
        if initial_n2_fraction_z is not None
        else None
    )
    if init_z is not None and init_z.size != nz:
        raise ValueError(
            f"initial_n2_fraction_z has size {init_z.size}, expected {nz}"
        )
    p_p_fwd = np.zeros((nz, nt))
    p_p_bwd = np.zeros((nz, nt))
    n0 = np.zeros((nz, nt))
    n1 = np.zeros((nz, nt))
    n2 = np.zeros((nz, nt))
    n3 = np.zeros((nz, nt))

    if forward_pump:
        p_p_fwd[0] = p_p_in
    if backward_pump:
        p_p_bwd[-1] = p_p_in * backward_fraction

    # A backward pump propagates along −z, so it needs a reverse z-sweep.
    # Populations depend on both fields, so iterate forward-march + backward
    # sweep to self-consistency (3 sweeps converge well for moderate α_p·L).
    has_bwd = backward_pump and backward_fraction > 0.0
    n_sweeps = 3 if has_bwd else 1
    for sweep in range(n_sweeps):
        last_sweep = sweep == n_sweeps - 1
        for iz in range(nz):
            if last_sweep and (iz == 0 or iz == nz - 1 or iz % max(1, nz // 20) == 0):
                frac = progress_base + progress_span * (iz + 1) / max(nz, 1)
                emit_progress(progress_callback, frac, f"Pump pass: z slice {iz + 1}/{nz}")
            f2_iz = float(init_z[iz]) if init_z is not None else float(initial_n2_fraction)
            n0[iz], n1[iz], n2[iz], n3[iz] = march_populations_pump_qss(
                t,
                p_p_fwd[iz],
                p_p_bwd[iz],
                n_tot=n_tot,
                a_pump=a_pump,
                gamma_p=gamma_p,
                sigma_p=sigma_p,
                sigma_ep=sigma_ep,
                hnu_p=hnu_p,
                lifetimes=lifetimes,
                initial_n2_fraction=f2_iz,
            )
            if iz >= nz - 1:
                continue
            p_p_fwd[iz + 1] = advance_pump_power_z(
                p_p_fwd[iz],
                n0[iz],
                sigma_p,
                dz,
                gamma_p=gamma_p,
                n2=n2[iz],
                sigma_ep=sigma_ep,
            )
        if has_bwd:
            for iz in range(nz - 2, -1, -1):
                p_p_bwd[iz] = advance_pump_power_z(
                    p_p_bwd[iz + 1],
                    n0[iz],
                    sigma_p,
                    dz,
                    gamma_p=gamma_p,
                    n2=n2[iz],
                    sigma_ep=sigma_ep,
                )

    return p_p_fwd, p_p_bwd, n0, n1, n2, n3


def _time_step_deltas(t: np.ndarray) -> np.ndarray:
    """Duration of interval ending at index it: dt[it] = t[it] - t[it-1] (dt[0] = dt[1])."""
    dt = np.zeros_like(t, dtype=np.float64)
    if t.size > 1:
        diff = np.diff(t)
        dt[1:] = diff
        dt[0] = diff[0]
    return dt


# Coupled ASE iterations per z-slab (spontaneous + stimulated deplete N₂).
N_ASE_COUPLED_ITERS = 1


def _local_spontaneous_source_w_nm(
    n2: np.ndarray,
    sigma_e: np.ndarray,
    hnu: np.ndarray,
    wl: np.ndarray,
    *,
    tau_21_s: float,
    eta_guided: float,
    gamma_s: float,
    n_tot: float,
) -> np.ndarray:
    """Spontaneous ASE source (nt, nlam) W/nm; zero where N₂ is negligible."""
    nt = n2.size
    nlam = sigma_e.size
    out = np.zeros((nt, nlam), dtype=np.float64)
    floor = 1e-12 * n_tot
    for it in range(nt):
        n2i = float(n2[it])
        if n2i > floor:
            out[it] = spontaneous_power_w_per_nm(
                n2i,
                sigma_e,
                hnu,
                wl,
                tau_21_s=tau_21_s,
                eta_guided=eta_guided,
                gamma_s=gamma_s,
            )
    return out


def _propagate_ase_z_slab(
    iz: int,
    *,
    n0: np.ndarray,
    n2: np.ndarray,
    n_tot: float,
    dz: float,
    a_signal: float,
    gamma_s: float,
    sigma_a: np.ndarray,
    sigma_e: np.ndarray,
    hnu: np.ndarray,
    wl: np.ndarray,
    tau_21_s: float,
    eta_guided: float,
    p_ase_f: np.ndarray,
    p_ase_b: np.ndarray,
) -> None:
    """Forward/backward ASE slab with local N₂(t) spontaneous source and g(N₀,N₂)."""
    gain_ase = gain_coefficient_m(n0[iz], n2[iz], sigma_a, sigma_e, gamma_s=gamma_s)
    gdz = np.clip(gain_ase * dz, -50.0, 50.0)
    spont = _local_spontaneous_source_w_nm(
        n2[iz],
        sigma_e,
        hnu,
        wl,
        tau_21_s=tau_21_s,
        eta_guided=eta_guided,
        gamma_s=gamma_s,
        n_tot=n_tot,
    )
    src = spont * a_signal * dz
    p_ase_f[iz + 1] = np.maximum(p_ase_f[iz] * np.exp(gdz) + src, 0.0)
    p_ase_b[iz] = np.maximum(p_ase_b[iz + 1] * np.exp(-gdz) + src, 0.0)


def _deplete_populations_from_ase_slab(
    iz: int,
    *,
    n0: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    n3: np.ndarray,
    n_tot: float,
    dt_slab: float,
    dlam: np.ndarray,
    sigma_e: np.ndarray,
    sigma_a: np.ndarray,
    hnu: np.ndarray,
    gamma_p: float,
    gamma_s: float,
    sigma_p: float,
    sigma_ep: float,
    hnu_p: float,
    a_signal: float,
    a_pump: float,
    lifetimes: FourLevelLifetimes,
    p_p_fwd: np.ndarray,
    p_p_bwd: np.ndarray,
    p_ase_f: np.ndarray,
    p_ase_b: np.ndarray,
    ase_threshold_w: float = 1e-6,
    pump_channel_rows: list | None = None,
    t_s: np.ndarray | None = None,
) -> None:
    """RK4 over slab transit: ASE stimulated emission depletes N₂ (active bins only)."""
    if dt_slab <= 0.0:
        return
    nt = n0.shape[1]
    nlam = dlam.size
    zero_sig = np.zeros(nlam, dtype=np.float64)
    n2_floor = 1e-12 * n_tot
    p_ase_mid = 0.25 * (
        p_ase_f[iz] + p_ase_f[iz + 1] + p_ase_b[iz] + p_ase_b[iz + 1]
    )
    p_t = np.sum(p_ase_mid * dlam, axis=1)
    active = np.where((n2[iz] > n2_floor) & (p_t >= ase_threshold_w))[0]
    for it in active:
        p_ase_row = p_ase_mid[it]
        ip = float(p_p_fwd[iz, it] + p_p_bwd[iz, it]) / a_pump
        n0i, n2i, n3i = march_level_interval(
            float(n0[iz, it]),
            float(n2[iz, it]),
            float(n3[iz, it]),
            n_tot,
            dt_slab,
            ip_w_m2=ip,
            p_sig_row=zero_sig,
            p_ase_row=p_ase_row,
            dlam=dlam,
            sigma_e=sigma_e,
            sigma_a=sigma_a,
            hnu=hnu,
            gamma_p=gamma_p,
            gamma_s=gamma_s,
            sigma_p=sigma_p,
            sigma_ep=sigma_ep,
            hnu_p=hnu_p,
            a_signal=a_signal,
            lifetimes=lifetimes,
            include_signal=False,
            couple_ase=True,
            pump_channel_rows=pump_channel_rows,
            pump_time_index=int(it),
            pump_time_s=t_s,
        )
        n0[iz, it] = n0i
        n2[iz, it] = n2i
        n3[iz, it] = n3i
        n1[iz, it] = 0.0


def _signal_pass(
    *,
    z: np.ndarray,
    t: np.ndarray,
    wl: np.ndarray,
    p_s_in: np.ndarray,
    n0: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    n3: np.ndarray,
    p_p_fwd: np.ndarray,
    p_p_bwd: np.ndarray,
    n_tot: float,
    dz: float,
    dt_travel: float,
    gamma_s: float,
    gamma_p: float,
    sigma_a: np.ndarray,
    sigma_e: np.ndarray,
    sigma_p: float,
    sigma_ep: float,
    hnu: np.ndarray,
    hnu_p: float,
    a_signal: float,
    a_pump: float,
    eta_guided: float,
    lifetimes: FourLevelLifetimes,
    include_ase: bool,
    burst_start_time_s: float = 0.0,
    progress_callback: ProgressCallback | None = None,
    progress_base: float = 0.38,
    progress_span: float = 0.54,
    pump_channel_rows_provider: Callable[[int], list] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz, nt, nlam = z.size, t.size, wl.size
    p_s = np.zeros((nz, nt, nlam))
    p_ase_f = np.zeros((nz, nt, nlam))
    p_ase_b = np.zeros((nz, nt, nlam))
    p_s[0] = p_s_in
    # Pump-pass populations (per z,t); restored whenever signal/ASE coupling is off.
    n0_pump = n0.copy()
    n2_pump = n2.copy()
    n3_pump = n3.copy()

    dlam = np.gradient(wl)
    sig_threshold_w = 0.05
    ase_threshold_w = 1e-6
    dt_arr = _time_step_deltas(t)
    n_iters = N_ASE_COUPLED_ITERS if include_ase else 1
    it_first_active = int(np.clip(np.searchsorted(t, float(burst_start_time_s), side="left"), 0, nt - 1))

    for iz in range(nz - 1):
        if iz == 0 or iz == nz - 2 or iz % max(1, (nz - 1) // 20) == 0:
            frac = progress_base + progress_span * (iz + 1) / max(nz - 1, 1)
            emit_progress(
                progress_callback,
                frac,
                f"Signal pass: z slice {iz + 1}/{nz - 1}",
            )

        for _ase_pass in range(n_iters):
            n0_run = float(n0_pump[iz, it_first_active])
            n2_run = float(n2_pump[iz, it_first_active])
            n3_run = float(n3_pump[iz, it_first_active])
            in_coupled_interval = False

            for it in range(nt):
                if it < it_first_active:
                    n0i = float(n0_pump[iz, it])
                    n2i = float(n2_pump[iz, it])
                    n3i = float(n3_pump[iz, it])
                elif not in_coupled_interval:
                    n0i = float(n0_pump[iz, it])
                    n2i = float(n2_pump[iz, it])
                    n3i = float(n3_pump[iz, it])
                else:
                    n0i, n2i, n3i = n0_run, n2_run, n3_run

                gain_it = gamma_s * (sigma_e * n2i - sigma_a * n0i)
                gdz_it = np.clip(gain_it * dz, -50.0, 50.0)
                p_in = p_s[iz, it]
                p_out = np.maximum(p_in * np.exp(gdz_it), 0.0)
                p_s[iz + 1, it] = p_out

                dt_it = float(dt_arr[it + 1]) if it + 1 < nt else 0.0
                couple_sig = False
                couple_ase = False
                if it >= it_first_active and dt_it > 0.0:
                    p_ase_row = p_ase_f[iz, it] + p_ase_b[iz, it] if include_ase else np.zeros(nlam)
                    sig_w = float(np.sum(0.5 * (p_in + p_out) * dlam))
                    ase_w = float(np.sum(p_ase_row * dlam)) if include_ase else 0.0
                    couple_sig = sig_w >= sig_threshold_w
                    couple_ase = include_ase and ase_w >= ase_threshold_w

                    if couple_sig or couple_ase:
                        # Multi-channel pump rates (if any) are summed inside
                        # march_level_interval; outside coupled intervals the
                        # populations revert to the pump-pass (QSS) values,
                        # which already include every pump channel.
                        pump_rows = (
                            pump_channel_rows_provider(iz)
                            if pump_channel_rows_provider is not None
                            else None
                        )
                        if not in_coupled_interval:
                            n0_run = float(n0_pump[iz, it])
                            n2_run = float(n2_pump[iz, it])
                            n3_run = float(n3_pump[iz, it])
                            in_coupled_interval = True
                        ip = float(p_p_fwd[iz, it] + p_p_bwd[iz, it]) / a_pump
                        p_row = 0.5 * (p_in + p_out) if couple_sig else np.zeros(nlam)
                        n0_run, n2_run, n3_run = march_level_interval(
                            n0_run,
                            n2_run,
                            n3_run,
                            n_tot,
                            dt_it,
                            ip_w_m2=ip,
                            p_sig_row=p_row,
                            p_ase_row=p_ase_row,
                            dlam=dlam,
                            sigma_e=sigma_e,
                            sigma_a=sigma_a,
                            hnu=hnu,
                            gamma_p=gamma_p,
                            gamma_s=gamma_s,
                            sigma_p=sigma_p,
                            sigma_ep=sigma_ep,
                            hnu_p=hnu_p,
                            a_signal=a_signal,
                            lifetimes=lifetimes,
                            include_signal=couple_sig,
                            couple_ase=couple_ase,
                            pump_channel_rows=pump_rows,
                            pump_time_index=it,
                            pump_time_s=t,
                        )
                        n0i, n2i, n3i = n0_run, n2_run, n3_run
                    else:
                        in_coupled_interval = False
                        n0i = float(n0_pump[iz, it])
                        n2i = float(n2_pump[iz, it])
                        n3i = float(n3_pump[iz, it])

                if it >= it_first_active:
                    n0[iz, it] = n0i
                    n2[iz, it] = n2i
                    n3[iz, it] = n3i
                    n1[iz, it] = 0.0

            if include_ase:
                _propagate_ase_z_slab(
                    iz,
                    n0=n0,
                    n2=n2,
                    n_tot=n_tot,
                    dz=dz,
                    a_signal=a_signal,
                    gamma_s=gamma_s,
                    sigma_a=sigma_a,
                    sigma_e=sigma_e,
                    hnu=hnu,
                    wl=wl,
                    tau_21_s=lifetimes.tau_21_s,
                    eta_guided=eta_guided,
                    p_ase_f=p_ase_f,
                    p_ase_b=p_ase_b,
                )
                _deplete_populations_from_ase_slab(
                    iz,
                    n0=n0,
                    n1=n1,
                    n2=n2,
                    n3=n3,
                    n_tot=n_tot,
                    dt_slab=dt_travel,
                    pump_channel_rows=(
                        pump_channel_rows_provider(iz)
                        if pump_channel_rows_provider is not None
                        else None
                    ),
                    t_s=t,
                    dlam=dlam,
                    sigma_e=sigma_e,
                    sigma_a=sigma_a,
                    hnu=hnu,
                    gamma_p=gamma_p,
                    gamma_s=gamma_s,
                    sigma_p=sigma_p,
                    sigma_ep=sigma_ep,
                    hnu_p=hnu_p,
                    a_signal=a_signal,
                    a_pump=a_pump,
                    lifetimes=lifetimes,
                    p_p_fwd=p_p_fwd,
                    p_p_bwd=p_p_bwd,
                    p_ase_f=p_ase_f,
                    p_ase_b=p_ase_b,
                )

    return p_s, p_ase_f, p_ase_b, n0, n1, n2, n3


def run_fiber_cpa(
    cfg: FiberCPAConfig,
    backend: str = "cpu",
    progress_callback: ProgressCallback | None = None,
) -> FiberCPAResult:
    if _multichannel_mode(cfg):
        if backend == "taichi":
            warnings.warn(
                "Multi-channel mode uses the CPU solver (Taichi path not implemented).",
                stacklevel=2,
            )
        from laser_sim.physics.fiber_cpa_multichannel import run_fiber_cpa_multichannel

        return run_fiber_cpa_multichannel(cfg, progress_callback=progress_callback)
    if backend == "taichi":
        from laser_sim.physics.taichi_isolated import run_fiber_cpa_taichi_isolated
        from laser_sim.physics.taichi_solver import run_fiber_cpa_taichi, use_isolated_taichi

        if use_isolated_taichi():
            return run_fiber_cpa_taichi_isolated(cfg, progress_callback=progress_callback)
        return run_fiber_cpa_taichi(cfg, progress_callback=progress_callback)
    if backend != "cpu":
        raise ValueError(f"Unknown backend {backend!r}; use 'cpu' or 'taichi'.")

    emit_progress(progress_callback, 0.02, "Building grids and cross-sections…")

    z, t, wl = _build_grids(cfg)
    nz, nt, nlam = z.size, t.size, wl.size
    dz = float(z[1] - z[0]) if nz > 1 else cfg.fiber_length_m
    v_g = C0 / cfg.material.n_group
    dt_travel = dz / v_g
    lifetimes = cfg.four_level or lifetimes_from_material(cfg.material)

    r_core = 0.5 * cfg.core_diameter_um * 1e-6
    r_clad = 0.5 * cfg.cladding_diameter_um * 1e-6
    a_core = core_area_m2(r_core)
    a_clad = cladding_area_m2(r_clad)

    if cfg.cladding_pumped:
        gamma_p = overlap_cladding_pump(r_core, r_clad)
        a_pump = a_clad
    else:
        gamma_p = 1.0
        a_pump = a_core
    wl_m = float(cfg.signal.center_wavelength_nm) * 1e-9
    gamma_s, a_signal, a_core_geom = signal_overlap_gamma(
        r_core, cfg.signal.center_wavelength_nm, cfg.core_na
    )
    v_no = v_number(r_core, cfg.core_na, wl_m)
    eta_guided = guided_spontaneous_fraction(v_no)

    sigma_a = cfg.material.sigma_abs_at(wl)
    sigma_e = cfg.material.sigma_em_at(wl)
    sigma_p = float(cfg.material.sigma_abs_at(cfg.pump.wavelength_nm)[0])
    sigma_ep = float(cfg.material.sigma_em_at(cfg.pump.wavelength_nm)[0])
    hnu_p = float(cfg.material.photon_energy_j(cfg.pump.wavelength_nm)[0])
    hnu = cfg.material.photon_energy_j(wl)
    n_tot = cfg.yb_concentration_m3
    # kappa_datasheet: reporting only. Pump z-propagation uses α_p = γ_p·σ_p·N₀(z,t)
    # from the rate equations, NOT this fixed datasheet value.
    kappa_datasheet = gamma_p * sigma_p * n_tot

    if cfg.pump_absorption_db_per_m is not None and cfg.pump_absorption_db_per_m > 0:
        kappa_implied_db = kappa_datasheet * (10.0 / np.log(10.0))
        ratio = kappa_implied_db / cfg.pump_absorption_db_per_m
        if ratio > 20.0 or ratio < 0.05:
            warnings.warn(
                f"N_tot={n_tot:.3e} m⁻³ implies κ={kappa_implied_db:.1f} dB/m "
                f"but pump_absorption_db_per_m={cfg.pump_absorption_db_per_m:.1f}. "
                f"Ratio={ratio:.1f}×. If you switched between cladding-pumped and "
                f"core-pumped, the overridden N_tot may be for the wrong geometry. "
                f"N_tot is a fiber property (ions/m³) independent of pump geometry; "
                f"only γ_p and A_pump change when you change pump coupling.",
                stacklevel=2,
            )

    p_p_in = build_pump_power(t, cfg.pump)

    from laser_sim.physics.four_level import pump_rate_per_ion, steady_state_n2_fraction_pump

    peak_pump_w = float(np.max(p_p_in)) if p_p_in.size else cfg.pump.peak_power_w
    i_p_peak = peak_pump_w / a_pump
    w_p_abs_peak, w_p_esa_peak = pump_rate_per_ion(
        i_p_peak, sigma_p=sigma_p, sigma_ep=sigma_ep, hnu_p=hnu_p
    )
    n2_ss_peak = steady_state_n2_fraction_pump(
        w_p_abs_peak, w_p_esa_peak, lifetimes.tau_21_s
    )
    sig_idx = int(np.argmin(np.abs(wl - cfg.signal.center_wavelength_nm)))
    beta_t = float(
        sigma_a[sig_idx] / (sigma_a[sig_idx] + sigma_e[sig_idx] + 1e-30)
    )
    if n2_ss_peak < beta_t:
        p_sat = hnu_p * a_pump / (sigma_p * lifetimes.tau_21_s)
        warnings.warn(
            f"Pump-only steady-state N₂/N_tot ≈ {n2_ss_peak*100:.1f}% at peak pump "
            f"({peak_pump_w:.2g} W), below signal transparency β_t ≈ {beta_t*100:.2f}% "
            f"at {cfg.signal.center_wavelength_nm:.0f} nm. "
            f"Cladding P_sat ≈ {p_sat:.2f} W; increase pump or use core pumping.",
            stacklevel=2,
        )

    p_s_in = build_chirped_signal(t, wl, cfg.signal)

    p_p_fwd, p_p_bwd, n0, n1, n2, n3 = _pump_pass(
        z=z,
        t=t,
        p_p_in=p_p_in,
        n_tot=n_tot,
        dz=dz,
        sigma_p=sigma_p,
        gamma_p=gamma_p,
        sigma_ep=sigma_ep,
        hnu_p=hnu_p,
        a_pump=a_pump,
        lifetimes=lifetimes,
        forward_pump=cfg.forward_pump,
        backward_pump=cfg.backward_pump,
        backward_fraction=cfg.backward_pump_fraction,
        progress_callback=progress_callback,
        initial_n2_fraction=cfg.initial_n2_fraction,
        initial_n2_fraction_z=cfg.initial_n2_fraction_z,
    )

    pops_after_pump = populations_to_fractions(
        n0.copy(), n1.copy(), n2.copy(), n3.copy(), n_tot
    )

    emit_progress(progress_callback, 0.36, "Pump pass complete — signal pass…")

    p_s, p_ase_f, p_ase_b, n0, n1, n2, n3 = _signal_pass(
        z=z,
        t=t,
        wl=wl,
        p_s_in=p_s_in,
        n0=n0,
        n1=n1,
        n2=n2,
        n3=n3,
        p_p_fwd=p_p_fwd,
        p_p_bwd=p_p_bwd,
        n_tot=n_tot,
        dz=dz,
        dt_travel=dt_travel,
        gamma_s=gamma_s,
        gamma_p=gamma_p,
        sigma_a=sigma_a,
        sigma_e=sigma_e,
        sigma_p=sigma_p,
        sigma_ep=sigma_ep,
        hnu=hnu,
        hnu_p=hnu_p,
        a_signal=a_signal,
        a_pump=a_pump,
        eta_guided=eta_guided,
        lifetimes=lifetimes,
        include_ase=cfg.include_ase,
        burst_start_time_s=cfg.signal.burst_start_time_s,
        progress_callback=progress_callback,
    )

    emit_progress(progress_callback, 0.94, "Integrating energies and diagnostics…")

    pops = populations_to_fractions(n0, n1, n2, n3, n_tot)

    from laser_sim.physics.diagnostics import steady_state_time_index

    it_ss = steady_state_time_index(t, cfg.signal)
    g0_spec = gain_coefficient_m(
        n0[:, it_ss],
        n2[:, it_ss],
        sigma_a,
        sigma_e,
        gamma_s=gamma_s,
    )
    g0_np = np.mean(g0_spec, axis=1)
    g0_db = g0_np * (10.0 / np.log(10.0))

    pump_abs_db_m = np.zeros(nz)
    for iz in range(1, nz):
        p0 = max(float(np.mean(p_p_fwd[iz - 1])), 1e-15)
        p1 = max(float(np.mean(p_p_fwd[iz])), 1e-15)
        pump_abs_db_m[iz] = (-np.log(p1 / p0) / max(dz, 1e-12)) * 10.0 / np.log(10.0)

    e_pump_in = _integrate_pump_energy(p_p_fwd[0], t)
    e_pump_out = _integrate_pump_energy(p_p_fwd[-1], t)
    e_pkt_expected = packet_energy_expected_j(cfg.signal)
    e_pkt_in = integrate_packet_energy(p_s[0], t, wl, cfg.signal)
    e_pkt_out = integrate_packet_energy(p_s[-1], t, wl, cfg.signal)
    e_pls_in = integrate_single_pulse_energy(p_s[0], t, wl, cfg.signal, pulse_index=0)
    e_pls_out = integrate_single_pulse_energy(p_s[-1], t, wl, cfg.signal, pulse_index=0)
    n_burst = int(cfg.signal.burst_count)
    pulse_e_in = np.array(
        [
            integrate_single_pulse_energy(p_s[0], t, wl, cfg.signal, pulse_index=b)
            for b in range(n_burst)
        ]
    )
    pulse_e_out = np.array(
        [
            integrate_single_pulse_energy(p_s[-1], t, wl, cfg.signal, pulse_index=b)
            for b in range(n_burst)
        ]
    )
    e_ase = (
        integrate_packet_energy(p_ase_f[-1] + p_ase_b[0], t, wl, cfg.signal)
        if cfg.include_ase
        else 0.0
    )
    e_pump_abs_frac = 1.0 - e_pump_out / max(e_pump_in, 1e-30)

    from laser_sim.physics.energy_budget import compute_amplifier_energy_budget

    budget = compute_amplifier_energy_budget(
        t_s=t,
        z_m=z,
        populations=pops,
        wavelength_nm=wl,
        n_tot=n_tot,
        tau_21_s=lifetimes.tau_21_s,
        a_signal_m2=a_signal,
        eta_guided=eta_guided,
        gamma_s=gamma_s,
        energy_pump_in_j=e_pump_in,
        energy_pump_out_j=e_pump_out,
        energy_packet_in_j=e_pkt_in,
        energy_packet_out_j=e_pkt_out,
        energy_ase_out_j=e_ase,
    )
    if not budget.balance_ok:
        warnings.warn(
            f"Energy budget: emission ({budget.total_emission_j*1e3:.4f} mJ) exceeds "
            f"pump absorbed ({budget.pump_absorbed_j*1e3:.4f} mJ) by "
            f"{budget.balance_residual_j*1e3:.4f} mJ. Check saturation or grid resolution.",
            stacklevel=2,
        )

    notes = (
        f"N_tot={n_tot:.3e} m⁻³  N₂_max/N_tot={float(n2.max())/n_tot:.4f}. "
        f"Quasi-2L (N₁ skipped). κ = Γ_p·σ_p·N = {kappa_datasheet:.3f} m⁻¹; "
        f"α_p = Γ_p·σ_p·N₀. τ₂₁={lifetimes.tau_21_s:.3e}s (spont N₂→N₀). "
        f"Γ_s={gamma_s:.4f}, η_guided={eta_guided:.4f}, V={v_no:.2f}, "
        f"A_sig={a_signal*1e12:.2f} µm². Δt_travel={dt_travel*1e12:.2f} ps."
    )
    ss_ok, ss_metric, ss_notes = False, float("nan"), ""
    if cfg.signal.rep_rate_mode:
        ss_ok, ss_metric, ss_notes = analyze_rep_rate_steady_state(t, p_s[-1], wl, cfg.signal)
        notes = ss_notes

    emit_progress(progress_callback, 1.0, "Done")

    return FiberCPAResult(
        z_m=z,
        t_s=t,
        wavelength_nm=wl,
        pump_fwd_w=p_p_fwd,
        pump_bwd_w=p_p_bwd,
        signal_fwd_w_nm=p_s,
        ase_fwd_w_nm=p_ase_f,
        ase_bwd_w_nm=p_ase_b,
        n2_fraction=pops.n2,
        populations=pops,
        populations_after_pump=pops_after_pump,
        pump_absorption_db_per_m=pump_abs_db_m,
        sigma_abs_signal_m2=sigma_a,
        sigma_em_signal_m2=sigma_e,
        sigma_abs_pump_m2=sigma_p,
        sigma_em_pump_m2=sigma_ep,
        pump_wavelength_nm=float(cfg.pump.wavelength_nm),
        kappa_datasheet_np_m=kappa_datasheet,
        energy_pump_in_j=e_pump_in,
        energy_pump_out_j=e_pump_out,
        energy_pulse_in_j=e_pls_in,
        energy_pulse_out_j=e_pls_out,
        energy_packet_in_j=e_pkt_in,
        energy_packet_out_j=e_pkt_out,
        energy_packet_expected_j=e_pkt_expected,
        energy_ase_out_j=e_ase,
        energy_unguided_spont_j=budget.spontaneous_unguided_j,
        energy_balance_ok=budget.balance_ok,
        energy_balance_residual_j=budget.balance_residual_j,
        ase_fraction_of_emission=budget.ase_fraction_of_emission,
        pump_power_absorbed_fraction=e_pump_abs_frac,
        notes=notes + (
            f" Energy budget: emit={budget.total_emission_j*1e3:.3f} mJ, "
            f"pump abs={budget.pump_absorbed_j*1e3:.3f} mJ, "
            f"ASE frac={budget.ase_fraction_of_emission*100:.1f}%, "
            f"unguided spont={budget.spontaneous_unguided_j*1e6:.3f} µJ."
        ),
        steady_state_reached=ss_ok,
        steady_state_metric=ss_metric,
        rep_rate_hz=cfg.signal.rep_rate_hz if cfg.signal.rep_rate_mode else None,
        n_periods_simulated=cfg.signal.n_periods if cfg.signal.rep_rate_mode else 0,
        dt_travel_s=dt_travel,
        four_level_lifetimes=lifetimes,
        gamma_signal=gamma_s,
        eta_guided_spontaneous=eta_guided,
        v_number=v_no,
        signal_mode_area_m2=a_signal,
        a_signal_m2=a_signal,
        pulse_energies_in_j=pulse_e_in,
        pulse_energies_out_j=pulse_e_out,
        g0_small_signal_np_m=g0_np,
        g0_small_signal_db_m=g0_db,
    )
