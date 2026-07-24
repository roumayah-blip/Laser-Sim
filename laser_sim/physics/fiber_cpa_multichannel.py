"""Multi-channel CPU path for ``run_fiber_cpa``."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from laser_sim.constants import C0
from laser_sim.materials.base import cladding_area_m2, core_area_m2, overlap_cladding_pump
from laser_sim.physics.fiber_cpa import (
    FiberCPAConfig,
    FiberCPAResult,
    _integrate_pump_energy,
    _signal_pass,
)
from laser_sim.physics.four_level import (
    FourLevelPopulations,
    advance_pump_power_z,
    gain_coefficient_m,
    lifetimes_from_material,
    march_populations_pump_qss,
    populations_to_fractions,
    pump_rate_per_ion,
    steady_state_n2_fraction_pump,
)
from laser_sim.physics.modes import guided_spontaneous_fraction, signal_overlap_gamma, v_number
from laser_sim.physics.progress import ProgressCallback, emit_progress
from laser_sim.physics.rep_rate import analyze_rep_rate_steady_state
from laser_sim.pulses.chirp import (
    build_chirped_signal,
    build_pump_power,
    integrate_packet_energy,
    integrate_single_pulse_energy,
    packet_energy_expected_j,
)
from laser_sim.pulses.multichannel import (
    PumpChannel,
    SignalChannel,
    build_multichannel_time_grid,
    collect_signal_wavelength_grids,
    enabled_pump_channels,
    enabled_signal_channels,
)


@dataclass(frozen=True)
class ResolvedPumpChannel:
    """Per-pump-channel geometry and cross-sections at z=0."""

    channel: PumpChannel
    sigma_p: float
    sigma_ep: float
    hnu_p: float
    gamma_p: float
    a_pump_m2: float
    p_fwd: np.ndarray
    p_bwd: np.ndarray


def _resolve_pump_channels(
    cfg: FiberCPAConfig,
    t: np.ndarray,
    z: np.ndarray,
    r_core: float,
    r_clad: float,
) -> list[ResolvedPumpChannel]:
    nz, nt = z.size, t.size
    out: list[ResolvedPumpChannel] = []
    for pc in enabled_pump_channels(cfg.pump_channels):
        if pc.cladding_pumped:
            gamma_p = overlap_cladding_pump(r_core, r_clad)
            a_pump = cladding_area_m2(r_clad)
        else:
            gamma_p = 1.0
            a_pump = core_area_m2(r_core)
        lam = float(pc.spec.wavelength_nm)
        sigma_p = float(cfg.material.sigma_abs_at(lam)[0])
        sigma_ep = float(cfg.material.sigma_em_at(lam)[0])
        hnu_p = float(cfg.material.photon_energy_j(lam)[0])
        p_t = build_pump_power(t, pc.spec)
        p_fwd = np.zeros((nz, nt), dtype=np.float64)
        p_bwd = np.zeros((nz, nt), dtype=np.float64)
        fwd_frac = 1.0 - float(np.clip(pc.backward_fraction, 0.0, 1.0))
        p_fwd[0] = p_t * fwd_frac
        if pc.backward_fraction > 0.0:
            p_bwd[-1] = p_t * float(pc.backward_fraction)
        out.append(
            ResolvedPumpChannel(
                channel=pc,
                sigma_p=sigma_p,
                sigma_ep=sigma_ep,
                hnu_p=hnu_p,
                gamma_p=gamma_p,
                a_pump_m2=a_pump,
                p_fwd=p_fwd,
                p_bwd=p_bwd,
            )
        )
    return out


def _pump_rows_at_z(rpch: ResolvedPumpChannel, iz: int) -> tuple[np.ndarray, np.ndarray, float, float, float, float]:
    return (
        rpch.p_fwd[iz],
        rpch.p_bwd[iz],
        rpch.sigma_p,
        rpch.sigma_ep,
        rpch.hnu_p,
        rpch.a_pump_m2,
    )


def _pump_pass_multichannel(
    *,
    z: np.ndarray,
    t: np.ndarray,
    pump_chs: list[ResolvedPumpChannel],
    n_tot: float,
    dz: float,
    lifetimes,
    progress_callback: ProgressCallback | None,
    initial_n2_fraction: float = 0.0,
    initial_n2_fraction_z: np.ndarray | None = None,
) -> tuple[list[ResolvedPumpChannel], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nz = z.size
    init_z = (
        np.asarray(initial_n2_fraction_z, dtype=np.float64)
        if initial_n2_fraction_z is not None
        else None
    )
    n0 = np.zeros((nz, t.size))
    n1 = np.zeros((nz, t.size))
    n2 = np.zeros((nz, t.size))
    n3 = np.zeros((nz, t.size))

    # p_pf/p_pb args are unused when pump_channel_rows is given — the rates
    # come from the per-channel rows (live views of rp.p_fwd/p_bwd).
    zeros_t = np.zeros_like(t)
    # Backward pumps need a reverse z-sweep; iterate to self-consistency
    # (same scheme as the single-channel _pump_pass).
    any_bwd = any(rp.channel.backward_fraction > 0.0 for rp in pump_chs)
    n_sweeps = 3 if any_bwd else 1
    for sweep in range(n_sweeps):
        last_sweep = sweep == n_sweeps - 1
        for iz in range(nz):
            if last_sweep and (iz == 0 or iz == nz - 1 or iz % max(1, nz // 20) == 0):
                emit_progress(
                    progress_callback,
                    0.05 + 0.30 * (iz + 1) / max(nz, 1),
                    f"Pump pass (multi): z {iz + 1}/{nz}",
                )
            rows = [_pump_rows_at_z(rp, iz) for rp in pump_chs]
            f2_iz = float(init_z[iz]) if init_z is not None else float(initial_n2_fraction)
            n0[iz], n1[iz], n2[iz], n3[iz] = march_populations_pump_qss(
                t,
                zeros_t,
                zeros_t,
                n_tot=n_tot,
                a_pump=1.0,
                gamma_p=1.0,
                sigma_p=1.0,
                sigma_ep=1.0,
                hnu_p=1.0,
                lifetimes=lifetimes,
                initial_n2_fraction=f2_iz,
                pump_channel_rows=rows,
            )
            if iz >= nz - 1:
                continue
            for rp in pump_chs:
                rp.p_fwd[iz + 1] = advance_pump_power_z(
                    rp.p_fwd[iz],
                    n0[iz],
                    rp.sigma_p,
                    dz,
                    gamma_p=rp.gamma_p,
                    n2=n2[iz],
                    sigma_ep=rp.sigma_ep,
                )
        if any_bwd:
            for iz in range(nz - 2, -1, -1):
                for rp in pump_chs:
                    if rp.channel.backward_fraction > 0.0:
                        rp.p_bwd[iz] = advance_pump_power_z(
                            rp.p_bwd[iz + 1],
                            n0[iz],
                            rp.sigma_p,
                            dz,
                            gamma_p=rp.gamma_p,
                            n2=n2[iz],
                            sigma_ep=rp.sigma_ep,
                        )

    return pump_chs, n0, n1, n2, n3


def run_fiber_cpa_multichannel(
    cfg: FiberCPAConfig,
    progress_callback: ProgressCallback | None = None,
) -> FiberCPAResult:
    """CPU solver with multiple signal and/or pump channels."""
    emit_progress(progress_callback, 0.02, "Building multi-channel grids…")
    sig_chs = enabled_signal_channels(cfg.signal_channels)
    pump_chs_cfg = enabled_pump_channels(cfg.pump_channels)
    if not sig_chs:
        raise ValueError("multichannel mode requires at least one enabled signal channel")

    if cfg.time_s is not None:
        t = np.asarray(cfg.time_s, dtype=np.float64)
    else:
        t = build_multichannel_time_grid(
            signal_channels=cfg.signal_channels,
            pump_channels=cfg.pump_channels,
        )
    if cfg.wavelength_nm is not None:
        wl = np.asarray(cfg.wavelength_nm, dtype=np.float64)
        sig_slices = [(ch, slice(0, wl.size)) for ch in sig_chs]
    else:
        wl, sig_slices = collect_signal_wavelength_grids(cfg.signal_channels)

    z = np.linspace(0.0, cfg.fiber_length_m, cfg.n_z)
    nz, nt, nlam = z.size, t.size, wl.size
    dz = float(z[1] - z[0]) if nz > 1 else cfg.fiber_length_m

    lifetimes = cfg.four_level or lifetimes_from_material(cfg.material)
    r_core = 0.5 * cfg.core_diameter_um * 1e-6
    r_clad = 0.5 * cfg.cladding_diameter_um * 1e-6
    ref_ch = sig_chs[0]
    wl_m = float(ref_ch.spec.center_wavelength_nm) * 1e-9
    gamma_s, a_signal, _ = signal_overlap_gamma(r_core, ref_ch.spec.center_wavelength_nm, cfg.core_na)
    v_no = v_number(r_core, cfg.core_na, wl_m)
    eta_guided = guided_spontaneous_fraction(v_no)

    sigma_a = cfg.material.sigma_abs_at(wl)
    sigma_e = cfg.material.sigma_em_at(wl)
    hnu = cfg.material.photon_energy_j(wl)
    n_tot = cfg.yb_concentration_m3

    pump_resolved = _resolve_pump_channels(cfg, t, z, r_core, r_clad)
    if not pump_resolved:
        raise ValueError("multichannel mode requires at least one enabled pump channel")

    p_s_in = np.zeros((nt, nlam), dtype=np.float64)
    for ch, sl in sig_slices:
        p_s_in[:, sl] = build_chirped_signal(t, wl[sl], ch.spec)

    ref_pump = pump_resolved[0]
    from laser_sim.physics.four_level import multichannel_pump_rate_per_ion

    intensities = [
        (float(np.max(rp.p_fwd[0]) / max(rp.a_pump_m2, 1e-30)), rp.sigma_p, rp.sigma_ep, rp.hnu_p)
        for rp in pump_resolved
    ]
    w_abs_t, w_esa_t = multichannel_pump_rate_per_ion(intensities)
    sig_idx = int(np.argmin(np.abs(wl - ref_ch.spec.center_wavelength_nm)))
    beta_t = float(sigma_a[sig_idx] / (sigma_a[sig_idx] + sigma_e[sig_idx] + 1e-30))
    if steady_state_n2_fraction_pump(w_abs_t, w_esa_t, lifetimes.tau_21_s) < beta_t:
        warnings.warn(
            f"Multi-channel pump-only N₂/N_tot below transparency at "
            f"{ref_ch.spec.center_wavelength_nm:.0f} nm.",
            stacklevel=2,
        )

    pump_resolved, n0, n1, n2, n3 = _pump_pass_multichannel(
        z=z,
        t=t,
        pump_chs=pump_resolved,
        n_tot=n_tot,
        dz=dz,
        lifetimes=lifetimes,
        progress_callback=progress_callback,
        initial_n2_fraction=cfg.initial_n2_fraction,
        initial_n2_fraction_z=cfg.initial_n2_fraction_z,
    )
    pops_after_pump = populations_to_fractions(n0.copy(), n1.copy(), n2.copy(), n3.copy(), n_tot)

    p_p_fwd = np.sum(np.stack([rp.p_fwd for rp in pump_resolved], axis=0), axis=0)
    p_p_bwd = np.sum(np.stack([rp.p_bwd for rp in pump_resolved], axis=0), axis=0)

    burst_start = min(ch.spec.burst_start_time_s for ch in sig_chs)
    ref_sigma_p = ref_pump.sigma_p
    ref_sigma_ep = ref_pump.sigma_ep
    ref_hnu_p = ref_pump.hnu_p
    ref_gamma_p = ref_pump.gamma_p
    ref_a_pump = ref_pump.a_pump_m2

    emit_progress(progress_callback, 0.36, "Signal pass (multi-channel)…")

    p_s, p_ase_f, p_ase_b, n0, n1, n2, n3 = _signal_pass_multichannel(
        z=z,
        t=t,
        wl=wl,
        p_s_in=p_s_in,
        n0=n0,
        n1=n1,
        n2=n2,
        n3=n3,
        pump_resolved=pump_resolved,
        n_tot=n_tot,
        dz=dz,
        dt_travel=dz / (C0 / cfg.material.n_group),
        gamma_s=gamma_s,
        sigma_a=sigma_a,
        sigma_e=sigma_e,
        hnu=hnu,
        a_signal=a_signal,
        eta_guided=eta_guided,
        lifetimes=lifetimes,
        include_ase=cfg.include_ase,
        burst_start_time_s=burst_start,
        progress_callback=progress_callback,
        ref_gamma_p=ref_gamma_p,
        ref_sigma_p=ref_sigma_p,
        ref_sigma_ep=ref_sigma_ep,
        ref_hnu_p=ref_hnu_p,
        ref_a_pump=ref_a_pump,
    )

    return _assemble_multichannel_result(
        cfg=cfg,
        z=z,
        t=t,
        wl=wl,
        sig_slices=sig_slices,
        pump_resolved=pump_resolved,
        p_p_fwd=p_p_fwd,
        p_p_bwd=p_p_bwd,
        p_s=p_s,
        p_ase_f=p_ase_f,
        p_ase_b=p_ase_b,
        n0=n0,
        n1=n1,
        n2=n2,
        n3=n3,
        pops_after_pump=pops_after_pump,
        n_tot=n_tot,
        gamma_s=gamma_s,
        a_signal=a_signal,
        eta_guided=eta_guided,
        v_no=v_no,
        lifetimes=lifetimes,
        ref_pump=ref_pump,
        progress_callback=progress_callback,
    )


def _signal_pass_multichannel(
    *,
    z,
    t,
    wl,
    p_s_in,
    n0,
    n1,
    n2,
    n3,
    pump_resolved: list[ResolvedPumpChannel],
    n_tot: float,
    dz: float,
    dt_travel: float,
    gamma_s: float,
    sigma_a,
    sigma_e,
    hnu,
    a_signal: float,
    eta_guided: float,
    lifetimes,
    include_ase: bool,
    burst_start_time_s: float,
    progress_callback: ProgressCallback | None,
    ref_gamma_p: float,
    ref_sigma_p: float,
    ref_sigma_ep: float,
    ref_hnu_p: float,
    ref_a_pump: float,
):
    """Like ``_signal_pass`` but sums pump rates from all channels at each z,t."""
    from laser_sim.physics.fiber_cpa import _signal_pass

    p_fwd = np.sum(np.stack([rp.p_fwd for rp in pump_resolved], axis=0), axis=0)
    p_bwd = np.sum(np.stack([rp.p_bwd for rp in pump_resolved], axis=0), axis=0)

    return _signal_pass(
        z=z,
        t=t,
        wl=wl,
        p_s_in=p_s_in,
        n0=n0,
        n1=n1,
        n2=n2,
        n3=n3,
        p_p_fwd=p_fwd,
        p_p_bwd=p_bwd,
        n_tot=n_tot,
        dz=dz,
        dt_travel=dt_travel,
        gamma_s=gamma_s,
        gamma_p=ref_gamma_p,
        sigma_a=sigma_a,
        sigma_e=sigma_e,
        sigma_p=ref_sigma_p,
        sigma_ep=ref_sigma_ep,
        hnu=hnu,
        hnu_p=ref_hnu_p,
        a_signal=a_signal,
        a_pump=ref_a_pump,
        eta_guided=eta_guided,
        lifetimes=lifetimes,
        include_ase=include_ase,
        burst_start_time_s=burst_start_time_s,
        progress_callback=progress_callback,
        pump_channel_rows_provider=lambda iz: [_pump_rows_at_z(rp, iz) for rp in pump_resolved],
    )


def _assemble_multichannel_result(
    *,
    cfg,
    z,
    t,
    wl,
    sig_slices,
    pump_resolved,
    p_p_fwd,
    p_p_bwd,
    p_s,
    p_ase_f,
    p_ase_b,
    n0,
    n1,
    n2,
    n3,
    pops_after_pump,
    n_tot,
    gamma_s,
    a_signal,
    eta_guided,
    v_no,
    lifetimes,
    ref_pump,
    progress_callback,
) -> FiberCPAResult:
    from laser_sim.physics.diagnostics import steady_state_time_index
    from laser_sim.physics.energy_budget import compute_amplifier_energy_budget

    nz = z.size
    dz = float(z[1] - z[0]) if nz > 1 else cfg.fiber_length_m
    pops = populations_to_fractions(n0, n1, n2, n3, n_tot)
    sigma_a = cfg.material.sigma_abs_at(wl)
    sigma_e = cfg.material.sigma_em_at(wl)

    ref_spec = enabled_signal_channels(cfg.signal_channels)[0].spec
    it_ss = steady_state_time_index(t, ref_spec)
    g0_np = np.mean(
        gain_coefficient_m(n0[:, it_ss], n2[:, it_ss], sigma_a, sigma_e, gamma_s=gamma_s),
        axis=1,
    )
    g0_db = g0_np * (10.0 / np.log(10.0))

    pump_abs_db_m = np.zeros(nz)
    for iz in range(1, nz):
        p0 = max(float(np.mean(p_p_fwd[iz - 1])), 1e-15)
        p1 = max(float(np.mean(p_p_fwd[iz])), 1e-15)
        pump_abs_db_m[iz] = (-np.log(p1 / p0) / max(dz, 1e-12)) * 10.0 / np.log(10.0)

    e_pump_in = _integrate_pump_energy(p_p_fwd[0], t)
    e_pump_out = _integrate_pump_energy(p_p_fwd[-1], t)

    signal_info: list[dict] = []
    pulse_e_in_list: list[float] = []
    pulse_e_out_list: list[float] = []
    e_pkt_in = 0.0
    e_pkt_out = 0.0
    e_pkt_expected = 0.0
    for ch, sl in sig_slices:
        wl_ch = wl[sl]
        e_in = integrate_packet_energy(p_s[0, :, sl], t, wl_ch, ch.spec)
        e_out = integrate_packet_energy(p_s[-1, :, sl], t, wl_ch, ch.spec)
        e_pkt_in += e_in
        e_pkt_out += e_out
        e_pkt_expected += packet_energy_expected_j(ch.spec)
        n_burst = int(ch.spec.burst_count)
        pe_in = [
            integrate_single_pulse_energy(p_s[0, :, sl], t, wl_ch, ch.spec, pulse_index=b)
            for b in range(n_burst)
        ]
        pe_out = [
            integrate_single_pulse_energy(p_s[-1, :, sl], t, wl_ch, ch.spec, pulse_index=b)
            for b in range(n_burst)
        ]
        pulse_e_in_list.extend(pe_in)
        pulse_e_out_list.extend(pe_out)
        signal_info.append(
            {
                "name": ch.name,
                "slice": sl,
                "center_nm": float(ch.spec.center_wavelength_nm),
                "energy_in_j": float(e_in),
                "energy_out_j": float(e_out),
                "p_in_w_nm": np.asarray(p_s[0, :, sl], dtype=np.float64),
                "p_out_w_nm": np.asarray(p_s[-1, :, sl], dtype=np.float64),
                "spec": ch.spec,
            }
        )

    pump_info: list[dict] = []
    for rp in pump_resolved:
        e_in = _integrate_pump_energy(rp.p_fwd[0], t)
        e_out = _integrate_pump_energy(rp.p_fwd[-1], t)
        pump_info.append(
            {
                "name": rp.channel.name,
                "wavelength_nm": float(rp.channel.spec.wavelength_nm),
                "energy_in_j": float(e_in),
                "energy_out_j": float(e_out),
                "absorbed_fraction": float(1.0 - e_out / max(e_in, 1e-30)),
                "p_z_w": np.mean(rp.p_fwd, axis=1),
                "cladding_pumped": rp.channel.cladding_pumped,
            }
        )

    e_ase = 0.0
    if cfg.include_ase:
        # Same convention as the single-channel path: integrate emitted ASE
        # over the packet window (union across signal channels).
        from laser_sim.pulses.chirp import (
            _integrate_over_time_window,
            packet_integration_window_s,
        )

        windows = [packet_integration_window_s(ch.spec) for ch, _ in sig_slices]
        t_lo = min(w[0] for w in windows)
        t_hi = max(w[1] for w in windows)
        e_ase = float(
            _integrate_over_time_window(p_ase_f[-1] + p_ase_b[0], t, wl, t_lo, t_hi)
        )

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

    ss_ok, ss_metric, ss_notes = False, float("nan"), ""
    if ref_spec.rep_rate_mode:
        ss_ok, ss_metric, ss_notes = analyze_rep_rate_steady_state(t, p_s[-1], wl, ref_spec)

    kappa = ref_pump.gamma_p * ref_pump.sigma_p * n_tot
    emit_progress(progress_callback, 1.0, "Done (multi-channel)")

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
        sigma_abs_pump_m2=ref_pump.sigma_p,
        sigma_em_pump_m2=ref_pump.sigma_ep,
        pump_wavelength_nm=float(ref_pump.channel.spec.wavelength_nm),
        kappa_datasheet_np_m=kappa,
        energy_pump_in_j=e_pump_in,
        energy_pump_out_j=e_pump_out,
        energy_pulse_in_j=float(pulse_e_in_list[0]) if pulse_e_in_list else 0.0,
        energy_pulse_out_j=float(pulse_e_out_list[0]) if pulse_e_out_list else 0.0,
        energy_packet_in_j=e_pkt_in,
        energy_packet_out_j=e_pkt_out,
        energy_packet_expected_j=e_pkt_expected,
        energy_ase_out_j=e_ase,
        pump_power_absorbed_fraction=1.0 - e_pump_out / max(e_pump_in, 1e-30),
        notes=f"Multi-channel CPU: {len(signal_info)} signal, {len(pump_info)} pump. {ss_notes}",
        energy_unguided_spont_j=budget.spontaneous_unguided_j,
        energy_balance_ok=budget.balance_ok,
        energy_balance_residual_j=budget.balance_residual_j,
        ase_fraction_of_emission=budget.ase_fraction_of_emission,
        steady_state_reached=ss_ok,
        steady_state_metric=ss_metric,
        rep_rate_hz=ref_spec.rep_rate_hz if ref_spec.rep_rate_mode else None,
        n_periods_simulated=ref_spec.n_periods if ref_spec.rep_rate_mode else 0,
        dt_travel_s=dz / (C0 / cfg.material.n_group),
        four_level_lifetimes=lifetimes,
        gamma_signal=gamma_s,
        eta_guided_spontaneous=eta_guided,
        v_number=v_no,
        signal_mode_area_m2=a_signal,
        a_signal_m2=a_signal,
        pulse_energies_in_j=np.asarray(pulse_e_in_list, dtype=np.float64),
        pulse_energies_out_j=np.asarray(pulse_e_out_list, dtype=np.float64),
        g0_small_signal_np_m=g0_np,
        g0_small_signal_db_m=g0_db,
        signal_channels_info=signal_info,
        pump_channels_info=pump_info,
    )
