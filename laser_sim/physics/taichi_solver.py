"""
Taichi GPU orchestration for the Yb fiber CPA two-pass solver.
"""

from __future__ import annotations

import warnings

import numpy as np

from laser_sim.constants import C0
from laser_sim.materials.base import (
    cladding_area_m2,
    core_area_m2,
    overlap_cladding_pump,
)
from laser_sim.physics.fiber_cpa import (
    FiberCPAConfig,
    FiberCPAResult,
    _build_grids,
    _integrate_pump_energy,
    _pump_pass,
)
from laser_sim.physics.progress import ProgressCallback, emit_progress
from laser_sim.physics.four_level import (
    FourLevelLifetimes,
    gain_coefficient_m,
    lifetimes_from_material,
    populations_to_fractions,
)
from laser_sim.physics.modes import guided_spontaneous_fraction, signal_overlap_gamma, v_number
from laser_sim.physics import taichi_kernels as tk
from laser_sim.pulses.chirp import (
    build_chirped_signal,
    build_pump_power,
    integrate_packet_energy,
    integrate_single_pulse_energy,
    packet_energy_expected_j,
)
from laser_sim.physics.rep_rate import analyze_rep_rate_steady_state

N_ASE_ITER = 1


def _in_streamlit() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def use_isolated_taichi() -> bool:
    """Streamlit reruns break in-process CUDA contexts; use a spawn child process."""
    return _in_streamlit()


def _ensure_taichi_runtime() -> str:
    """Initialize Taichi in the current process (CLI / tests / GPU worker child)."""
    arch = "cuda"
    try:
        tk.init_taichi(arch=arch, fp="f32", force_reinit=True)
    except Exception:
        tk.abandon_taichi_runtime()
        arch = "cpu"
        tk.init_taichi(arch=arch, fp="f32", force_reinit=True)
    tk.warmup()
    return arch


def _sanitize_population_fractions(
    n0: np.ndarray,
    n2: np.ndarray,
    n3: np.ndarray,
    n_tot: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove NaN/inf and clamp before GPU upload (prevents spurious N2=1 from bad f32 state)."""
    n0f = np.nan_to_num(n0 / n_tot, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
    n2f = np.nan_to_num(n2 / n_tot, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    n0f = np.clip(n0f, 0.0, 1.0)
    n2f = np.clip(n2f, 0.0, 0.99)
    n3f = np.zeros_like(n0f, dtype=np.float32)
    s = n0f + n2f
    over = s > 1.0
    if np.any(over):
        n0f[over] /= s[over]
        n2f[over] /= s[over]
    return n0f, n2f, n3f


def _upload_spectral(wl: np.ndarray, sigma_a: np.ndarray, sigma_e: np.ndarray, hnu: np.ndarray) -> None:
    dlam = np.gradient(wl).astype(np.float32)
    tk._cache.sigma_e.from_numpy(sigma_e.astype(np.float32))
    tk._cache.sigma_a.from_numpy(sigma_a.astype(np.float32))
    tk._cache.hnu.from_numpy(hnu.astype(np.float32))
    tk._cache.dlam.from_numpy(dlam)


def _upload_time_steps(t: np.ndarray) -> None:
    from laser_sim.physics.fiber_cpa import _time_step_deltas

    tk._cache.dt_arr.from_numpy(_time_step_deltas(t).astype(np.float32))


def run_pump_pass_taichi(
    *,
    z: np.ndarray,
    t: np.ndarray,
    n_lam: int,
    p_p_in: np.ndarray,
    n_tot: float,
    dz: float,
    sigma_p: float,
    gamma_p: float,
    sigma_ep: float,
    hnu_p: float,
    a_pump: float,
    tau_32: float,
    tau_21: float,
    lifetimes,
    forward_pump: bool,
    backward_pump: bool,
    backward_fraction: float,
    progress_callback: ProgressCallback | None = None,
    initial_n2_fraction: float = 0.0,
    initial_n2_fraction_z: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Pump pass: causal QSS march on CPU (CUDA cannot run the long serial-in-t kernel),
    then optional GPU pump z-advance for consistency with the Taichi signal pass fields.
    """
    from laser_sim.physics.four_level import FourLevelLifetimes

    if not isinstance(lifetimes, FourLevelLifetimes):
        lifetimes = FourLevelLifetimes(
            tau_32_s=tau_32, tau_21_s=tau_21, tau_10_s=1e-9, skip_n1_level=True
        )

    return _pump_pass(
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
        forward_pump=forward_pump,
        backward_pump=backward_pump,
        backward_fraction=backward_fraction,
        progress_callback=progress_callback,
        progress_base=0.05,
        progress_span=0.28,
        initial_n2_fraction=initial_n2_fraction,
        initial_n2_fraction_z=initial_n2_fraction_z,
    )


def run_signal_pass_taichi(
    *,
    z: np.ndarray,
    t: np.ndarray,
    wl: np.ndarray,
    p_s_in: np.ndarray,
    n0: np.ndarray,
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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Causal signal + coupled ASE on CPU (populations depleted by signal and ASE).
    """
    from laser_sim.physics.fiber_cpa import _signal_pass

    nz, nt, nlam = z.size, t.size, wl.size
    n1 = np.zeros_like(n0)
    return _signal_pass(
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
        include_ase=include_ase,
        burst_start_time_s=burst_start_time_s,
        progress_callback=progress_callback,
        progress_base=0.38,
        progress_span=0.54,
    )


def run_fiber_cpa_taichi(
    cfg: FiberCPAConfig,
    progress_callback: ProgressCallback | None = None,
) -> FiberCPAResult:
    emit_progress(progress_callback, 0.01, "Initializing Taichi GPU runtime…")
    _ensure_taichi_runtime()
    emit_progress(progress_callback, 0.02, "Taichi: building grids…")

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
    gamma_s, a_signal, _ = signal_overlap_gamma(
        r_core, cfg.signal.center_wavelength_nm, cfg.core_na
    )
    lp_modes_list = cfg.lp_modes
    gamma_signal_per_mode = None
    if lp_modes_list is None:
        from laser_sim.physics.lp_modes import find_lp_modes

        lp_modes_list = find_lp_modes(r_core, cfg.core_na, wl_m)
    if lp_modes_list:
        gamma_s = lp_modes_list[0].gamma_overlap
        gamma_signal_per_mode = np.array([m.gamma_overlap for m in lp_modes_list], dtype=np.float64)

    v_no = v_number(r_core, cfg.core_na, wl_m)
    eta_guided = guided_spontaneous_fraction(v_no)

    sigma_a = cfg.material.sigma_abs_at(wl)
    sigma_e = cfg.material.sigma_em_at(wl)
    sigma_p = float(cfg.material.sigma_abs_at(cfg.pump.wavelength_nm)[0])
    sigma_ep = float(cfg.material.sigma_em_at(cfg.pump.wavelength_nm)[0])
    hnu_p = float(cfg.material.photon_energy_j(cfg.pump.wavelength_nm)[0])
    hnu = cfg.material.photon_energy_j(wl)
    n_tot = cfg.yb_concentration_m3
    kappa_datasheet = gamma_p * sigma_p * n_tot

    tk._cache.ensure(nz, nt, nlam)
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
            f"at {cfg.signal.center_wavelength_nm:.0f} nm (P_sat_clad ≈ {p_sat:.2f} W).",
            stacklevel=2,
        )

    p_s_in = build_chirped_signal(t, wl, cfg.signal)

    p_p_fwd, p_p_bwd, n0, n1, n2, n3 = run_pump_pass_taichi(
        z=z,
        t=t,
        n_lam=nlam,
        p_p_in=p_p_in,
        n_tot=n_tot,
        dz=dz,
        sigma_p=sigma_p,
        gamma_p=gamma_p,
        sigma_ep=sigma_ep,
        hnu_p=hnu_p,
        a_pump=a_pump,
        tau_32=lifetimes.tau_32_s,
        tau_21=lifetimes.tau_21_s,
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

    # Snapshot full pump field (n_z × n_t) on GPU before signal pass touches cache.
    tk._cache.ensure(nz, nt, nlam)
    tk._cache.p_pump_fwd.from_numpy(p_p_fwd.astype(np.float32))
    tk._cache.p_pump_bwd.from_numpy(p_p_bwd.astype(np.float32))
    p_p_fwd = tk._cache.p_pump_fwd.to_numpy().astype(np.float64)
    p_p_bwd = tk._cache.p_pump_bwd.to_numpy().astype(np.float64)

    emit_progress(progress_callback, 0.36, "Pump pass complete — GPU signal pass…")

    p_s, p_ase_f, p_ase_b, n0, n1, n2, n3 = run_signal_pass_taichi(
        z=z,
        t=t,
        wl=wl,
        p_s_in=p_s_in,
        n0=n0,
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

    emit_progress(progress_callback, 0.94, "Integrating energies…")

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

    lp_note = ""
    if lp_modes_list:
        names = ", ".join(m.name for m in lp_modes_list[:6])
        lp_note = f" LP modes ({len(lp_modes_list)}): {names}. Γ_s(LP01)={gamma_s:.4f}."

    notes = (
        f"N_tot={n_tot:.3e} m⁻³  N₂_max/N_tot={float(n2.max())/n_tot:.4f}. "
        f"Taichi backend. κ = Γ_p·σ_p·N = {kappa_datasheet:.3f} m⁻¹; "
        f"α_p = Γ_p·σ_p·N₀. τ₂₁={lifetimes.tau_21_s:.3e}s. "
        f"Γ_s={gamma_s:.4f}, η_guided={eta_guided:.4f}, V={v_no:.2f}, "
        f"A_sig={a_signal*1e12:.2f} µm². Δt_travel={dt_travel*1e12:.2f} ps.{lp_note}"
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
        notes=notes
        + (
            f" ASE frac={budget.ase_fraction_of_emission*100:.1f}% of emission; "
            f"balance={'OK' if budget.balance_ok else 'OVER'}."
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
        lp_modes=lp_modes_list,
        gamma_signal_per_mode=gamma_signal_per_mode,
    )
