"""
CPA fiber amplifier diagnostics: equations, coefficients, per-z steady-state tables.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from laser_sim.constants import C0, DB_PER_NP, H
from laser_sim.materials.base import Material
from laser_sim.physics.fiber_cpa import FiberCPAConfig, FiberCPAResult
from laser_sim.physics.four_level import FourLevelLifetimes, gain_coefficient_m, pump_rate_per_ion
from laser_sim.physics.b_integral import BIntegralResult, N2_SILICA_M2_PER_W
from laser_sim.physics.modes import signal_mode_area_m2
from laser_sim.pulses.chirp import (
    ChirpedBurstSpec,
    _get_pulse_weights,
    first_pulse_center_time_s,
    nearest_time_index,
    packet_duration_s,
    packet_time_extent_s,
)
from laser_sim.pulses.cw_average import packet_average_power_w


@dataclass
class DiagnosticsTables:
    steady_state_time_s: float
    pulse_time_s: float
    z_m: np.ndarray
    n0_fraction: np.ndarray
    n2_fraction: np.ndarray
    n3_fraction: np.ndarray
    inversion_n2_over_n0_n2: np.ndarray
    pump_power_w: np.ndarray
    pump_step_transmission: np.ndarray
    alpha_p_np_m: np.ndarray
    alpha_p_db_m: np.ndarray
    g0_center_m: np.ndarray
    g0_spectral_mean_m: np.ndarray
    G0_power_per_step: np.ndarray
    G_pulse_power_per_step: np.ndarray
    pulse_power_w: np.ndarray


def _fmt_si(value: float, unit: str) -> str:
    """Human-readable SI: scientific notation if very small/large, else prefix."""
    if not np.isfinite(value):
        return f"{value} {unit}"
    av = abs(value)
    if av == 0:
        return f"0 {unit}"
    if av < 1e-3 or av >= 1e4:
        return f"{value:.6e} {unit}"
    for scale, prefix in (
        (1e-9, "n"),
        (1e-6, "µ"),
        (1e-3, "m"),
        (1.0, ""),
        (1e3, "k"),
        (1e6, "M"),
    ):
        if av < 1000 * scale:
            return f"{value / scale:.6g} {prefix}{unit}"
    return f"{value:.6e} {unit}"


def steady_state_time_index(t_s: np.ndarray, spec: ChirpedBurstSpec) -> int:
    """Pump-only sample time: just before the packet arrives."""
    margin = max(1e-9, 0.01 * packet_duration_s(spec))
    t_ss = max(float(spec.burst_start_time_s) - margin, float(t_s[0]))
    return nearest_time_index(t_s, t_ss)


def pulse_time_index(t_s: np.ndarray, spec: ChirpedBurstSpec) -> int:
    return nearest_time_index(t_s, first_pulse_center_time_s(spec))


def _integrated_power_w_nm(row: np.ndarray, dlam: np.ndarray) -> float:
    return float(np.sum(row * dlam))


def compute_diagnostics_tables(
    result: FiberCPAResult,
    cfg: FiberCPAConfig,
    *,
    gamma_p: float,
    gamma_s: float,
    dz: float,
) -> DiagnosticsTables:
    z = result.z_m
    t = result.t_s
    wl = result.wavelength_nm
    spec = cfg.signal
    nz = z.size
    dlam = np.gradient(wl)
    it_ss = steady_state_time_index(t, spec)
    it_p = pulse_time_index(t, spec)
    t_ss = float(t[it_ss])
    t_p = float(t[it_p])

    pops = result.populations
    n0 = pops.n0[:, it_ss]
    n2 = pops.n2[:, it_ss]
    n3 = pops.n3[:, it_ss]
    # Final populations (after signal + ASE); pump-only snapshot is in populations_after_pump.
    inv = n2 / np.maximum(n0 + n2, 1e-30)

    p_pump = result.pump_fwd_w[:, it_ss]
    pump_t = np.ones(nz)
    pump_t[1:] = p_pump[1:] / np.maximum(p_pump[:-1], 1e-30)

    sigma_a = result.sigma_abs_signal_m2
    sigma_e = result.sigma_em_signal_m2
    ic = int(np.argmin(np.abs(wl - spec.center_wavelength_nm)))
    g0_c = gain_coefficient_m(
        n0 * cfg.yb_concentration_m3,
        n2 * cfg.yb_concentration_m3,
        np.array([sigma_a[ic]]),
        np.array([sigma_e[ic]]),
        gamma_s=gamma_s,
    )[:, 0]
    g_spec = gain_coefficient_m(
        n0 * cfg.yb_concentration_m3,
        n2 * cfg.yb_concentration_m3,
        sigma_a,
        sigma_e,
        gamma_s=gamma_s,
    )
    g0_mean = np.mean(g_spec, axis=1)
    g0_c = np.asarray(g0_c, dtype=np.float64)
    g0_mean = np.asarray(g0_mean, dtype=np.float64)

    gdz0 = np.clip(g0_mean * dz, -50.0, 50.0)
    G0 = np.exp(gdz0)

    n0_d = n0 * cfg.yb_concentration_m3
    alpha_np = gamma_p * result.sigma_abs_pump_m2 * n0_d
    alpha_db = alpha_np * (10.0 / np.log(10.0))

    G_pulse = np.ones(nz)
    p_pulse = np.zeros(nz)
    for iz in range(nz - 1):
        pin = _integrated_power_w_nm(result.signal_fwd_w_nm[iz, it_p], dlam)
        pout = _integrated_power_w_nm(result.signal_fwd_w_nm[iz + 1, it_p], dlam)
        p_pulse[iz] = pin
        G_pulse[iz] = pout / max(pin, 1e-30)
    p_pulse[-1] = _integrated_power_w_nm(result.signal_fwd_w_nm[-1, it_p], dlam)

    return DiagnosticsTables(
        steady_state_time_s=t_ss,
        pulse_time_s=t_p,
        z_m=z,
        n0_fraction=n0,
        n2_fraction=n2,
        n3_fraction=n3,
        inversion_n2_over_n0_n2=inv,
        pump_power_w=p_pump,
        pump_step_transmission=pump_t,
        alpha_p_np_m=alpha_np,
        alpha_p_db_m=alpha_db,
        g0_center_m=g0_c,
        g0_spectral_mean_m=g0_mean,
        G0_power_per_step=G0,
        G_pulse_power_per_step=G_pulse,
        pulse_power_w=p_pulse,
    )


def _equations_section() -> str:
    return """
================================================================================
EQUATIONS (quasi-two-level Yb, N1 neglected, tau_10 -> 0)
================================================================================

Conservation:  N0 + N2 + N3 = N_tot

Pump rates (per ion, s^-1):
  W_p,abs = Gamma_p * (I_p / h nu_p) * sigma_p
  W_p,esa = Gamma_p * (I_p / h nu_p) * sigma_ep
  I_p = P_pump / A_pump

Pump-only QSS (fast N3, march N2 with tau_21):
  N3_ss = W_p,abs * N0 * tau_32 / (1 + W_p,esa * tau_32)
  N2_ss = (W_p,abs * tau_21) / (1 + W_p,abs * tau_21) * N_tot   [self-consistent; not W*N0*tau]

Signal-pass RK4 (quasi-2L; causal in time at each z):
  At each z: n_running starts at burst_start (pump-pass N2 there); march only for it >= burst_start.
  Pre-burst: pump-pass populations kept; signal gain uses local N0,N2; no RK4 march.
  For it >= burst_start: g(it) from n_running; P_s[iz+1,it] *= exp(g*dz); RK4 over dt[it] (NOT dt_travel)
  N3 held at 0 in GPU quasi-2L RK4 (no N3/tau_32 term)

Pump depletion along +z (per time slice):
  alpha_p = Gamma_p * sigma_p * N0(z,t)     [m^-1]
  P_p(z+dz,t) = P_p(z,t) * exp(-alpha_p * dz)

Signal coupling (spectral row P_s in W/nm):
  W_se = Gamma_s * Sum_lambda sigma_e(lambda) * I_s(lambda) / h nu(lambda)
  W_abs = Gamma_s * Sum_lambda sigma_a(lambda) * I_s(lambda) / h nu(lambda)
  I_s(lambda) = P_s(lambda) / A_core

Small-signal gain coefficient:
  g(lambda) = Gamma_s * (sigma_e(lambda)*N2 - sigma_a(lambda)*N0)   [m^-1]

Power gain over dz (uniform g):
  G = exp(g * dz)

Dopant from datasheet kappa (dB/m):
  alpha_np = kappa_dB/m / (10/ln 10)
  N_tot = alpha_np / (Gamma_p * sigma_p)   [default; optional legacy N=alpha_np/sigma_p]

Rep-rate CW steady state:
  CW inversion N2_ss/N_tot = W_p*tau21 / (1 + W_p*tau21)
  Warm-start: populations initialised at N2_ss (no startup transient)
  T_rep = 1/f_rep;  burst_start = T_rep (auto)
  Convergence: compare last two packets; steady_state_tol = user setting

ASE (coupled to populations):
  Spontaneous: dn2/dt includes -N2/tau21 (all fluorescence depletes inversion)
  Guided birth: eta*Gamma_s of that decay feeds ASE source at local N2(z,t)
  Stimulated: W_se from signal AND ASE intensity (incoherent sum) depletes N2
  After each z-slab: ASE exp(g*dz) + local spont; RK4 over dt_travel with ASE-only W_se
  Energy: E_emit = E_sig_net + E_ASE_out + E_unguided; require E_emit <= E_pump_abs
"""


def _coefficients_section(
    cfg: FiberCPAConfig,
    result: FiberCPAResult,
    dopant: Any,
    *,
    gamma_p: float,
    gamma_s: float,
    a_core: float,
    a_mode_m2: float,
    a_pump: float,
    dz: float,
    dt_travel: float,
    material: Material,
) -> str:
    n_tot = cfg.yb_concentration_m3
    sigma_p = result.sigma_abs_pump_m2
    ic = int(np.argmin(np.abs(result.wavelength_nm - cfg.signal.center_wavelength_nm)))
    sigma_a_c = float(result.sigma_abs_signal_m2[ic])
    sigma_e_c = float(result.sigma_em_signal_m2[ic])
    lt = result.four_level_lifetimes
    p_avg = packet_average_power_w(cfg.signal)

    lines = [
        "",
        "=" * 80,
        "COEFFICIENTS (SI; check prefixes)",
        "=" * 80,
        f"  c0                         {_fmt_si(C0, 'm/s')}",
        f"  h                          {_fmt_si(H, 'J·s')}",
        f"  DB_PER_NP (dB->Np/m)       {DB_PER_NP:.6g}",
        "",
        "Fiber geometry",
        f"  core diameter              {_fmt_si(cfg.core_diameter_um * 1e-6, 'm')}",
        f"  cladding diameter          {_fmt_si(cfg.cladding_diameter_um * 1e-6, 'm')}",
        f"  fiber length L             {_fmt_si(cfg.fiber_length_m, 'm')}",
        f"  n_z steps, dz              {cfg.n_z}, {_fmt_si(dz, 'm')}",
        f"  A_core                     {_fmt_si(a_core, 'm²')}",
        f"  A_pump (clad/core)         {_fmt_si(a_pump, 'm²')}",
        f"  Gamma_p (overlap)          {gamma_p:.6g}  (dimensionless)",
        f"  Gamma_s (signal)           {gamma_s:.6g}",
        f"  A_eff (LP01 mode area)     {_fmt_si(a_mode_m2, 'm²')}",
        f"  A_eff (LP01 mode area)     {a_mode_m2 * 1e12:.2f} µm²",
        f"  MFD (1/e² intensity diam.) {np.sqrt(4 * a_mode_m2 / np.pi) * 1e6:.2f} µm",
        "",
        "Material / Yb",
        f"  material                   {material.name}",
        f"  n_group                    {material.n_group:.6g}",
        f"  N_tot (Yb density)         {_fmt_si(n_tot, 'm⁻³')}",
        f"  sigma_abs pump @ {result.pump_wavelength_nm:.1f} nm  {_fmt_si(sigma_p, 'm²')}",
        f"  sigma_abs signal @ {cfg.signal.center_wavelength_nm:.1f} nm {_fmt_si(sigma_a_c, 'm²')}",
        f"  sigma_em  signal @ {cfg.signal.center_wavelength_nm:.1f} nm {_fmt_si(sigma_e_c, 'm²')}",
        f"  tau_32 (N3->N2)            {_fmt_si(lt.tau_32_s if lt else 0, 's')}",
        f"  tau_21 (N2 fluores.)       {_fmt_si(lt.tau_21_s if lt else 0, 's')}",
        f"  kappa = Gamma_p*sigma_p*N  {_fmt_si(result.kappa_datasheet_np_m, 'm⁻¹')}",
        f"  kappa (dB/m)               {result.kappa_datasheet_np_m * (10/np.log(10)):.4g}",
        "",
        "Pump (from config)",
        f"  pump wavelength            {_fmt_si(cfg.pump.wavelength_nm * 1e-9, 'm')}",
        f"  pump peak / CW power       {_fmt_si(cfg.pump.peak_power_w, 'W')}",
        f"  datasheet pump abs (dB/m)  {cfg.pump_absorption_db_per_m}",
        "",
        "Signal packet",
        f"  center wavelength          {_fmt_si(cfg.signal.center_wavelength_nm * 1e-9, 'm')}",
        f"  bandwidth (FWHM scale)     {_fmt_si(cfg.signal.bandwidth_nm * 1e-9, 'm')}",
        f"  chirp duration             {_fmt_si(cfg.signal.chirp_duration_s, 's')}",
        f"  energy / pulse             {_fmt_si(cfg.signal.energy_per_pulse_j, 'J')}",
        f"  burst count                {cfg.signal.burst_count}",
        f"  burst spacing              {_fmt_si(cfg.signal.burst_spacing_s, 's')}",
    ]
    if cfg.signal.pulse_relative_powers is not None:
        weights = _get_pulse_weights(cfg.signal)
        weight_str = "  ".join(f"{w:.3f}" for w in weights)
        lines.extend(
            [
                f"  pulse shape weights         {weight_str}  (normalized)",
                f"  peak-to-mean ratio          {float(np.max(weights)):.3f}x",
            ]
        )
    else:
        lines.append("  pulse shape weights         flat (all equal)")
    lines.extend(
        [
            f"  burst start                {_fmt_si(cfg.signal.burst_start_time_s, 's')}",
            f"  packet avg power (equiv.)  {_fmt_si(p_avg, 'W')}",
            f"  CW-average mode            {cfg.signal.cw_average_power_mode}",
            "",
            "Grid / march",
            f"  n_t                        {result.t_s.size}",
            f"  n_lambda                   {result.wavelength_nm.size}",
            f"  dt_travel (= dz/v_g)       {_fmt_si(dt_travel, 's')}",
            f"  v_g = c/n_g                {_fmt_si(C0 / material.n_group, 'm/s')}",
        ]
    )
    if dopant is not None:
        lines.extend(
            [
                "",
                "Dopant estimate",
                f"  Gamma_p (calc)             {dopant.gamma_pump:.6g}",
                f"  N from kappa (calc)        {_fmt_si(dopant.concentration_m3, 'm⁻³')}",
                f"  alpha_np from dB/m         {_fmt_si(dopant.alpha_np_per_m, 'm⁻¹')}",
            ]
        )
    return "\n".join(lines)


def _b_integral_section(
    b: BIntegralResult,
    *,
    l_passive_before_m: float,
    l_active_m: float,
    l_passive_after_m: float,
    n2_m2_per_w: float = N2_SILICA_M2_PER_W,
) -> str:
    return "\n".join(
        [
            "",
            "=" * 80,
            "B-INTEGRAL (NONLINEAR PHASE)",
            "=" * 80,
            f"  n₂ (fused silica, 1030 nm):  {n2_m2_per_w:.2e} m²/W",
            f"  A_eff (LP01 mode):           {b.a_eff_m2 * 1e12:.2f} µm²",
            f"  P_peak in:                   {b.p_peak_in_w * 1e-3:.2f} kW",
            f"  P_peak out:                  {b.p_peak_out_w * 1e-3:.2f} kW",
            "",
            f"  Passive before  (L = {l_passive_before_m:.3f} m):  "
            f"B = {b.b_passive_before_rad:.4f} rad   "
            f"L_NL = {b.l_nl_passive_before_m * 100:.2f} cm",
            f"  Active fiber    (L = {l_active_m:.3f} m):  "
            f"B = {b.b_active_rad:.4f} rad   "
            f"L_NL = {b.l_nl_active_m * 100:.2f} cm (at output power)",
            f"  Passive after   (L = {l_passive_after_m:.3f} m):  "
            f"B = {b.b_passive_after_rad:.4f} rad   "
            f"L_NL = {b.l_nl_passive_after_m * 100:.2f} cm",
            "  ─────────────────────────────────────────────────────",
            f"  TOTAL:                           B = {b.b_total_rad:.4f} rad   "
            f"[{b.severity}]",
            "",
            "  Threshold guide: < 1 rad excellent, 1–3 rad moderate, "
            "3–5 rad significant, > 5 rad severe.",
        ]
    )


def _lit_match(sim: float, lit: float, tol: float = 0.15) -> str:
    if not np.isfinite(sim) or not np.isfinite(lit) or lit == 0:
        return "N/A"
    return "YES" if abs(sim - lit) / abs(lit) < tol else "NO"


def _time_grid_section(
    t_s: np.ndarray,
    cfg: FiberCPAConfig,
    dt_travel: float,
    material: Material,
) -> str:
    t = np.asarray(t_s, dtype=np.float64)
    spec = cfg.signal
    pump = cfg.pump
    n_t = t.size
    t0, t1 = float(t[0]), float(t[-1])
    window = max(t1 - t0, 1e-30)
    bs = float(spec.burst_start_time_s)
    pkt_end = packet_time_extent_s(spec)

    i_coarse = t < bs - 1e-15
    n_coarse = int(np.sum(i_coarse))
    i_fine = ~i_coarse
    n_fine = int(np.sum(i_fine))
    if n_coarse > 1:
        dt_coarse = float(np.mean(np.diff(t[i_coarse])))
    elif n_coarse == 1:
        dt_coarse = window
    else:
        dt_coarse = float("nan")
    if n_fine > 1:
        dt_fine = float(np.mean(np.diff(t[i_fine])))
    elif n_fine == 1:
        dt_fine = 0.0
    else:
        dt_fine = float("nan")

    lt = cfg.four_level or FourLevelLifetimes(
        tau_32_s=1e-12, tau_21_s=material.lifetime_s, tau_10_s=1e-9
    )
    tau_21 = lt.tau_21_s
    tau_32 = lt.tau_32_s
    ratio_21 = dt_travel / max(tau_21, 1e-30)
    ratio_32 = dt_travel / max(tau_32, 1e-30)

    pump_dur = pump.duration_s if pump.cw else pump.duration_s
    pump_end = pump.start_time_s + pump_dur

    frac_coarse = (min(bs, t1) - t0) / window
    frac_fine = 1.0 - frac_coarse
    bar_len = 50
    n_coarse_bar = max(0, int(round(frac_coarse * bar_len)))
    n_fine_bar = bar_len - n_coarse_bar
    bar = "[" + "-" * n_coarse_bar + "f" * n_fine_bar + "]"

    lines = [
        "",
        "=" * 80,
        "TIME GRID ANATOMY",
        "=" * 80,
        f"  Total time window:     {t0 * 1e6:.4f} to {t1 * 1e6:.4f} µs",
        f"  n_t:                   {n_t}",
        f"  Coarse region:         t[0] to burst_start ({bs * 1e6:.4f} µs), ~{n_coarse} points",
        f"                         Mean dt in coarse region: {dt_coarse * 1e6:.4f} µs",
        f"  Fine region:           burst_start to end, ~{n_fine} points",
        f"                         Mean dt in fine region: {dt_fine * 1e12:.4f} ps",
        f"  dt_travel:             {dt_travel * 1e12:.4f} ps  (dz/v_g, signal transit per z-slab)",
        f"  dt_travel / tau_21:    {ratio_21:.6e}  (should be << 1 for stable QSS)",
        f"  dt_travel / tau_32:    {ratio_32:.6e}  (MUST be >> 1 for QSS on N3)",
        f"  Pump pulse:            start {pump.start_time_s * 1e6:.4f} µs, "
        f"duration {pump_dur * 1e6:.4f} µs, end {pump_end * 1e6:.4f} µs, shape={pump.shape}",
        f"  Signal packet:         burst_start {bs * 1e6:.4f} µs, "
        f"chirp {spec.chirp_duration_s * 1e9:.4f} ns, pulse_count={spec.burst_count}",
        f"  burst_start / tau_21:  {bs / tau_21:.6e}  (pump fill fraction before packet)",
        "",
        "  Time axis (coarse | fine):",
        f"  {bar}  coarse={frac_coarse * 100:.1f}%  fine={frac_fine * 100:.1f}%",
        "  ^t=0        ^burst_start              ^t_end",
        "",
        "  Time-marching algorithm:",
        "    PASS A (pump): At each z-slice iz, march N0,N2,N3 forward in time from t=0",
        "      For each time step: N3 QSS (tau_32 << dt), N2 exponential approach to N2_ss.",
        "      Then advance P_pump to iz+1: P_p *= exp(-Gamma_p*sigma_p*N0*dz).",
        "    PASS B (signal): At each z-slice iz:",
        "      1. Compute g(t,lam) = Gamma_s*(sigma_e*N2 - sigma_a*N0)  -- shape (n_t, n_lam)",
        "      2. Advance signal: P_s[iz+1] = P_s[iz] * exp(g*dz), clip g*dz to [-50,50]",
        "      3. Causal RK4: n_running from t[0]; each it uses dt[it]=t[it+1]-t[it] (not dt_travel)",
        "         Leading-edge depletion reduces gain for trailing bins and later pulses",
        "         (quasi-2L on GPU: no N3/tau_32 term)",
        "      4. Optional ASE sweep (gain on ASE fields only; signal advanced once per z)",
    ]
    return "\n".join(lines)


def _time_dynamics_section(
    result: FiberCPAResult,
    cfg: FiberCPAConfig,
    material: Material,
    n_tot: float,
    gamma_p: float,
    gamma_s: float,
    a_pump: float,
    sigma_p: float,
    hnu_p: float,
    lifetimes: FourLevelLifetimes | None,
) -> str:
    t = result.t_s
    n_t = int(t.size)
    wl = result.wavelength_nm
    spec = cfg.signal
    dlam = np.gradient(wl)
    ic = int(np.argmin(np.abs(wl - spec.center_wavelength_nm)))
    sa_c = float(result.sigma_abs_signal_m2[ic])
    se_c = float(result.sigma_em_signal_m2[ic])
    tau_21 = lifetimes.tau_21_s if lifetimes else material.lifetime_s

    pops_pump = result.populations_after_pump or result.populations
    iz = 0
    p_pump = result.pump_fwd_w[iz]
    p_sig = result.signal_fwd_w_nm[iz]
    n0f = pops_pump.n0[iz]
    n2f = pops_pump.n2[iz]

    bs = spec.burst_start_time_s
    pkt_lo = max(float(t[0]), bs - 2.0 * spec.chirp_duration_s)
    pkt_hi = packet_time_extent_s(spec) + 2.0 * spec.chirp_duration_s
    in_packet = (t >= pkt_lo) & (t <= pkt_hi)

    max_rows = 150
    stride = max(1, n_t // max_rows)
    indices = set(range(0, n_t, stride))
    indices.update(int(i) for i in np.where(in_packet)[0])
    indices.add(0)
    indices.add(n_t - 1)
    indices.add(steady_state_time_index(t, spec))
    indices.add(pulse_time_index(t, spec))
    idx_list = sorted(indices)

    lines = [
        "",
        "=" * 80,
        "PER-TIME-STEP DYNAMICS AT z=0",
        "=" * 80,
        f"  Populations: {'after pump pass' if result.populations_after_pump is not None else 'after full run'}",
        f"  Signal march starts at burst_start = {spec.burst_start_time_s * 1e6:.4f} µs (index {int(np.clip(np.searchsorted(t, spec.burst_start_time_s, side='left'), 0, n_t - 1))})",
        f"  N_tot = {n_tot:.6e} m^-3,  Gamma_p = {gamma_p:.6g},  Gamma_s = {gamma_s:.6g}",
        "",
        f"{'t(us)':>10} {'dt(us)':>9} {'P_pump':>10} {'I_p':>11} {'W*tau21':>9} "
        f"{'N0':>8} {'N2':>8} {'W_p':>10} {'N2_ss':>8} {'approach':>9} "
        f"{'P_sig':>11} {'g0_Np/m':>11}",
    ]

    g0_series = np.zeros(n_t)
    w_p_series = np.zeros(n_t)
    sat_series = np.zeros(n_t)
    n2ss_series = np.zeros(n_t)

    for it in range(n_t):
        ip = float(p_pump[it]) / a_pump
        w_abs, _ = pump_rate_per_ion(
            ip, gamma_p=gamma_p, sigma_p=sigma_p, sigma_ep=result.sigma_em_pump_m2, hnu_p=hnu_p
        )
        w_p_series[it] = w_abs
        sat_series[it] = w_abs * tau_21
        n2ss_series[it] = sat_series[it] / (1.0 + sat_series[it])
        g0_series[it] = float(
            gain_coefficient_m(
                np.array([n0f[it] * n_tot]),
                np.array([n2f[it] * n_tot]),
                np.array([sa_c]),
                np.array([se_c]),
                gamma_s=gamma_s,
            )[0, 0]
        )

    for it in idx_list:
        dt_i = float(t[it] - t[it - 1]) if it > 0 else 0.0
        ip = float(p_pump[it]) / a_pump
        w_abs = w_p_series[it]
        sat = sat_series[it]
        n0 = n0f[it]
        n2 = n2f[it]
        n2ss = n2ss_series[it]
        n2_ss_abs = n2ss * n_tot
        if n2_ss_abs > 1e-10 * n_tot:
            approach = (n2ss - n2) / n2ss
            approach_s = f"{approach:9.4f}"
        else:
            approach_s = "      nan"
        p_sig_tot = _integrated_power_w_nm(p_sig[it], dlam)
        lines.append(
            f"{t[it] * 1e6:10.4f} {dt_i * 1e6:9.4f} {p_pump[it]:10.3f} {ip:11.3e} {sat:9.4f} "
            f"{n0:8.5f} {n2:8.5f} {w_abs:10.3e} {n2ss:8.5f} {approach_s:>9} "
            f"{p_sig_tot:11.4e} {g0_series[it]:11.4e}"
        )

    beta_t = sa_c / max(sa_c + se_c, 1e-40)
    beta_arr = n2f / np.maximum(n0f + n2f, 1e-30)
    it_ss = steady_state_time_index(t, spec)
    it_pulse = pulse_time_index(t, spec)
    n2_at_ss = float(n2f[it_ss])
    margin_pp = (n2_at_ss - beta_t) * 100.0
    pass_inv = "ABOVE" if n2_at_ss >= beta_t else "BELOW"

    # Transparency crossing (interpolate)
    cross_t = float("nan")
    for it in range(1, n_t):
        if (beta_arr[it - 1] - beta_t) * (beta_arr[it] - beta_t) <= 0:
            f = (beta_t - beta_arr[it - 1]) / max(beta_arr[it] - beta_arr[it - 1], 1e-30)
            cross_t = float(t[it - 1] + f * (t[it] - t[it - 1]))
            break

    imax = int(np.nanargmax(g0_series))
    imin = int(np.nanargmin(g0_series))

    cross_str = (
        f"{cross_t * 1e6:.4f} µs" if np.isfinite(cross_t) else "not crossed in window"
    )
    lines.extend(
        [
            "",
            "  Derived quantities:",
            f"    Transparency beta_t (signal center)     = {beta_t * 100:.4f}%",
            f"    N2(z=0) at t_ss ({t[it_ss] * 1e6:.4f} µs)  = {n2_at_ss * 100:.4f}%  "
            f"[{pass_inv} transparency by {abs(margin_pp):.4f} pp] — "
            f"{'PASS' if n2_at_ss >= beta_t else 'FAIL'}",
            f"    Time of N2 crossing beta_t (interp)     = {cross_str}",
            f"    First pulse peak time                   = {t[it_pulse] * 1e6:.4f} µs "
            f"({t[it_pulse] * 1e9:.4f} ns from t=0)",
            f"    Maximum g0 at center λ                  = {g0_series[imax]:.4e} Np/m at t={t[imax] * 1e6:.4f} µs",
            f"    Minimum g0 at center λ                  = {g0_series[imin]:.4e} Np/m at t={t[imin] * 1e6:.4f} µs",
        ]
    )
    return "\n".join(lines)


def _coefficient_audit_section(
    result: FiberCPAResult,
    cfg: FiberCPAConfig,
    n_tot: float,
    gamma_p: float,
    gamma_s: float,
    a_pump: float,
    a_signal: float,
    sigma_p: float,
    sigma_ep: float,
    hnu_p: float,
    hnu_s: np.ndarray,
    sigma_a: np.ndarray,
    sigma_e: np.ndarray,
    lifetimes: FourLevelLifetimes | None,
    it_ss: int,
    iz: int = 0,
    dz: float = 0.0,
) -> str:
    wl = result.wavelength_nm
    ic = int(np.argmin(np.abs(wl - cfg.signal.center_wavelength_nm)))
    sa_c = float(sigma_a[ic])
    se_c = float(sigma_e[ic])
    hnu_c = float(hnu_s[ic])
    lt = lifetimes or FourLevelLifetimes(
        tau_32_s=1e-12, tau_21_s=cfg.material.lifetime_s, tau_10_s=1e-9
    )
    tau_21 = lt.tau_21_s

    pops = result.populations_after_pump or result.populations
    n0f = float(pops.n0[iz, it_ss])
    n2f = float(pops.n2[iz, it_ss])
    n3f = float(pops.n3[iz, it_ss])
    n0 = n0f * n_tot
    n2 = n2f * n_tot
    n3 = n3f * n_tot

    p_pump = float(result.pump_fwd_w[iz, it_ss])
    ip = p_pump / a_pump
    w_abs, w_esa = pump_rate_per_ion(
        ip, gamma_p=gamma_p, sigma_p=sigma_p, sigma_ep=sigma_ep, hnu_p=hnu_p
    )
    sat_pump = w_abs * tau_21
    n2_ss = sat_pump / (1.0 + sat_pump) * n_tot
    t_pump = float(cfg.signal.burst_start_time_s)
    fill_frac = 1.0 - np.exp(-t_pump / tau_21)

    dlam = np.gradient(wl)
    p_sig_t = np.array(
        [_integrated_power_w_nm(result.signal_fwd_w_nm[iz, it], dlam) for it in range(result.t_s.size)]
    )
    p_sig_peak = float(np.max(p_sig_t))
    i_peak = int(np.argmax(p_sig_t))
    isig = p_sig_peak / a_signal
    p_sat = hnu_c * a_signal / ((se_c + sa_c) * tau_21)
    w_se = gamma_s * se_c * isig / hnu_c
    w_abs_s = gamma_s * sa_c * isig / hnu_c
    sig_sat = w_se * tau_21

    g0 = float(
        gain_coefficient_m(
            np.array([n0]),
            np.array([n2]),
            np.array([sa_c]),
            np.array([se_c]),
            gamma_s=gamma_s,
        )[0, 0]
    )
    g0_db = g0 * (10.0 / np.log(10.0))
    beta_t = sa_c / (sa_c + se_c)
    beta_now = n2f / max(n0f + n2f, 1e-30)
    net = "GAIN" if beta_now >= beta_t else "BELOW TRANSPARENCY"
    margin = (beta_now - beta_t) * 100.0

    eta = result.eta_guided_spontaneous
    sigma_e_norm = float(np.trapezoid(sigma_e, wl))
    if sigma_e_norm <= 0:
        sigma_e_norm = float(np.sum(sigma_e)) or 1.0
    s_vol = eta * gamma_s * n2 * hnu_c * se_c / (tau_21 * sigma_e_norm)
    s_line = s_vol * a_signal * dz

    alpha_p = gamma_p * sigma_p * n0
    alpha_db = alpha_p * (10.0 / np.log(10.0))
    L = float(result.z_m[-1]) if result.z_m.size else cfg.fiber_length_m
    t_pump_trans = np.exp(-alpha_p * L)

    sat_warn = ""
    if p_sig_peak > p_sat:
        sat_warn = f"  WARNING: P_sig/P_sat = {p_sig_peak / p_sat:.2f} > 1 (saturated)"

    lines = [
        "",
        "=" * 80,
        f"RATE AND COEFFICIENT AUDIT (z={iz}, t=t_ss)",
        "=" * 80,
        f"  --- At z={iz}, t={float(result.t_s[it_ss]) * 1e6:.4f} µs (t_ss) ---",
        "",
        "  Population state:",
        f"    N0 = {n0:.4e} m^-3  (fraction: {n0f:.6f})",
        f"    N2 = {n2:.4e} m^-3  (fraction: {n2f:.6f})",
        f"    N3 = {n3:.4e} m^-3  (fraction: {n3f:.6f})",
        "",
        "  Pump rates:",
        f"    I_p = P_pump / A_pump = {p_pump:.4g} W / {a_pump:.4e} m^2 = {ip:.4e} W/m^2",
        f"    W_p,abs = Gamma_p * sigma_p * I_p / hnu_p",
        f"            = {gamma_p:.4g} * {sigma_p:.4e} m^2 * {ip:.4e} W/m^2 / {hnu_p:.4e} J",
        f"            = {w_abs:.4e} s^-1",
        f"    W_p * tau_21 = {sat_pump:.4f}  (pump saturation)",
        f"    N2_ss (self-consistent) = W_p*tau/(1+W_p*tau) * N_tot = {n2_ss:.4e} m^-3 ({n2_ss / n_tot * 100:.2f}%)",
        f"    Inversion build fraction in T_pump = 1 - exp(-T/tau_21) = {fill_frac:.4f}",
        "",
        "  Signal rates (at center wavelength, at peak signal power over time at this z):",
        f"    I_s = P_sig_peak / A_signal = {isig:.4e} W/m^2",
        f"    P_sat = hnu_s / ((sigma_e+sigma_a) * tau_21) * A_signal = {p_sat:.4g} W",
        f"    P_sig / P_sat = {p_sig_peak / max(p_sat, 1e-30):.4g}  (at t={result.t_s[i_peak] * 1e9:.4f} ns peak)",
        sat_warn,
        f"    W_se = Gamma_s * sigma_e * I_s / hnu_s = {w_se:.4e} s^-1",
        f"    W_abs = Gamma_s * sigma_a * I_s / hnu_s = {w_abs_s:.4e} s^-1",
        f"    W_se * tau_21 = {sig_sat:.4f}  (signal saturation parameter)",
        "",
        "  Gain coefficient:",
        f"    g(lam_ctr) = Gamma_s * (sigma_e * N2 - sigma_a * N0)",
        f"               = {gamma_s:.4g} * ({se_c:.4e} * {n2:.4e} - {sa_c:.4e} * {n0:.4e})",
        f"               = {g0:.4e} Np/m = {g0_db:.2f} dB/m",
        f"    Transparency threshold: beta_t = sigma_a/(sigma_a+sigma_e) = {beta_t * 100:.4f}%",
        f"    Current beta = N2/(N0+N2) = {beta_now * 100:.4f}%",
        f"    Net gain status: [{net}] (margin {margin:+.4f} percentage points vs beta_t)",
        "",
        "  ASE source (per unit volume, per nm):",
        f"    S_sp = eta * Gamma_s * N2 * hnu * sigma_e / (tau_21 * sigma_e_norm)",
        f"         = {s_vol:.4e} W/(m^3 nm)  (eta={eta:.4f})",
        f"    S_sp * A_signal * dz = {s_line:.4e} W/nm  (source per z-slab)",
        "",
        "  Pump absorption:",
        f"    alpha_p = Gamma_p * sigma_p * N0 = {alpha_p:.4e} Np/m = {alpha_db:.2f} dB/m",
        f"    Pump e-folding length = {1.0 / max(alpha_p, 1e-30):.4f} m",
        f"    Expected pump transmission over L={L:.4f} m: exp(-alpha_p*L) = {t_pump_trans:.4f} "
        f"({t_pump_trans * 100:.2f}%)",
    ]
    return "\n".join(lines)


def _crosssection_verification_section(
    material: Material,
    pump_wl_nm: float,
    sig_wl_nm: float,
    n_tot: float,
    gamma_p: float,
    kappa_db_m: float | None,
    result: FiberCPAResult,
) -> str:
    sa_p = float(material.sigma_abs_at(pump_wl_nm)[0])
    se_p = float(material.sigma_em_at(pump_wl_nm)[0])
    sa_s = float(material.sigma_abs_at(sig_wl_nm)[0])
    se_s = float(material.sigma_em_at(sig_wl_nm)[0])
    tau_21 = material.lifetime_s

    lit = {
        "sa976": 2.50e-24,
        "se976": 2.44e-24,
        "sa1030": 4.53e-26,
        "se1030": 6.43e-25,
        "tau": 0.88e-3,
        "beta_t": 0.0658,
    }
    ratio_sim = sa_p / max(se_p, 1e-40)
    ratio_lit = lit["sa976"] / lit["se976"]
    em_ratio_sim = se_s / max(sa_s, 1e-40)
    em_ratio_lit = lit["se1030"] / lit["sa1030"]
    beta_t_sim = sa_s / max(sa_s + se_s, 1e-40)

    kappa_spec = float(kappa_db_m) if kappa_db_m is not None else float("nan")
    kappa_impl = gamma_p * sa_p * n_tot * (10.0 / np.log(10.0))
    kappa_ratio = kappa_spec / kappa_impl if kappa_impl > 0 else float("nan")

    a_sig = result.signal_mode_area_m2 if result.signal_mode_area_m2 > 0 else 0.0
    hnu_s = float(material.photon_energy_j(sig_wl_nm)[0])
    p_sat = hnu_s * a_sig / ((se_s + sa_s) * tau_21)
    wl = result.wavelength_nm
    dlam = np.gradient(wl)
    p_sig_t = np.array(
        [_integrated_power_w_nm(result.signal_fwd_w_nm[0, it], dlam) for it in range(result.t_s.size)]
    )
    p_peak = float(np.max(p_sig_t))
    psat_ratio = p_peak / max(p_sat, 1e-30)
    if psat_ratio < 0.3:
        sat_label = "UNSATURATED"
    elif psat_ratio < 10:
        sat_label = "SATURATED"
    else:
        sat_label = "DEEPLY SATURATED"

    def warn2x(sim: float, lit_v: float) -> str:
        if not np.isfinite(sim) or lit_v == 0:
            return ""
        if abs(sim / lit_v) > 2 or abs(sim / lit_v) < 0.5:
            return "  WARNING: >2x from literature"
        return ""

    lines = [
        "",
        "=" * 80,
        "CROSS-SECTION VERIFICATION (vs RP Photonics / Liekki spectroscopy)",
        "=" * 80,
        "  Source: Liekki_Yb.inc (measured by Joona Koponen, 2012)",
        "          tau_Yb = 0.88 ms (radiative lifetime)",
        "",
        f"  {'Quantity':<22} {'Simulation':>18} {'Literature':>18} {'Match?':>8}",
        f"  {'-' * 22} {'-' * 18} {'-' * 18} {'-' * 8}",
        f"  {'sigma_abs(976 nm)':<22} {sa_p:>18.4e} {lit['sa976']:>18.4e} {_lit_match(sa_p, lit['sa976']):>8}",
        warn2x(sa_p, lit["sa976"]),
        f"  {'sigma_em(976 nm)':<22} {se_p:>18.4e} {lit['se976']:>18.4e} {_lit_match(se_p, lit['se976']):>8}",
        warn2x(se_p, lit["se976"]),
        f"  {'sigma_abs/em @976':<22} {ratio_sim:>18.4f} {ratio_lit:>18.4f} {_lit_match(ratio_sim, ratio_lit):>8}",
        f"  {'sigma_abs(1030 nm)':<22} {sa_s:>18.4e} {lit['sa1030']:>18.4e} {_lit_match(sa_s, lit['sa1030']):>8}",
        warn2x(sa_s, lit["sa1030"]),
        f"  {'sigma_em(1030 nm)':<22} {se_s:>18.4e} {lit['se1030']:>18.4e} {_lit_match(se_s, lit['se1030']):>8}",
        warn2x(se_s, lit["se1030"]),
        f"  {'sigma_em/sigma_abs @1030':<22} {em_ratio_sim:>18.4f} {em_ratio_lit:>18.1f} {_lit_match(em_ratio_sim, em_ratio_lit, 0.25):>8}",
        f"  {'tau_21':<22} {tau_21 * 1e3:>18.4f} ms {lit['tau'] * 1e3:>15.4f} ms {_lit_match(tau_21, lit['tau']):>8}",
        f"  {'Transparency beta_t':<22} {beta_t_sim * 100:>17.2f}% {lit['beta_t'] * 100:>17.2f}% {_lit_match(beta_t_sim, lit['beta_t']):>8}",
        "",
        "  Dopant consistency:",
        f"    kappa (dB/m) from spec:            {kappa_spec:.4g}",
        f"    kappa implied by N_tot and sigma:  {kappa_impl:.4g} dB/m  [= Gamma_p*sigma_p*N*DB_PER_NP]",
        f"    Ratio (should be 1.0):             {kappa_ratio:.4f}  "
        f"[{'CONSISTENT' if np.isfinite(kappa_ratio) and 0.85 <= kappa_ratio <= 1.15 else 'INCONSISTENT'}]",
        "",
        "  Signal saturation power:",
        f"    P_sat = hnu_s * A_signal / ((sigma_e+sigma_a)*tau_21) = {p_sat:.4g} W",
        f"    Peak signal power at z=0:          {p_peak:.4g} W",
        f"    Peak / P_sat:                      {psat_ratio:.4g}  [{sat_label}]",
        "",
        "  Match = YES when |sim - lit| / lit < 15%.",
    ]
    for w in (
        warn2x(sa_p, lit["sa976"]),
        warn2x(se_p, lit["se976"]),
        warn2x(sa_s, lit["sa1030"]),
        warn2x(se_s, lit["se1030"]),
    ):
        if w.strip():
            lines.append(w)
    return "\n".join(lines)


def _sanity_checks_section(
    result: FiberCPAResult,
    cfg: FiberCPAConfig,
    n_tot: float,
    gamma_p: float,
    gamma_s: float,
    a_pump: float,
    a_signal: float,
    sigma_p: float,
    hnu_p: float,
    sigma_a_sig: float,
    sigma_e_sig: float,
    hnu_s: float,
    lifetimes: FourLevelLifetimes | None,
    t_ss: float,
    it_ss: int,
    dz: float,
) -> str:
    lt = lifetimes or FourLevelLifetimes(
        tau_32_s=1e-12, tau_21_s=cfg.material.lifetime_s, tau_10_s=1e-9
    )
    tau_21 = lt.tau_21_s
    tau_32 = lt.tau_32_s
    dt_travel = result.dt_travel_s if result.dt_travel_s > 0 else dz * 1e-12

    pops = result.populations
    n0f = pops.n0
    n2f = pops.n2
    n3f = pops.n3
    s = n0f + n2f + n3f
    max_dev = float(np.max(np.abs(s - 1.0))) * 100.0

    ratio_32 = dt_travel / tau_32
    qss_ok = ratio_32 > 10.0

    L = float(result.z_m[-1]) if result.z_m.size else cfg.fiber_length_m
    kappa_np = result.kappa_datasheet_np_m
    expected_abs_frac = 1.0 - np.exp(-kappa_np * L)
    p0 = float(np.mean(result.pump_fwd_w[0]))
    pL = float(np.mean(result.pump_fwd_w[-1]))
    got_abs_frac = 1.0 - pL / max(p0, 1e-30)

    pops_z0 = result.populations_after_pump or pops
    n2_at_ss = float(pops_z0.n2[0, it_ss])
    beta_t = sigma_a_sig / max(sigma_a_sig + sigma_e_sig, 1e-40)
    inv_ok = n2_at_ss >= beta_t
    inv_margin = (n2_at_ss - beta_t) * 100.0

    p_sat = hnu_s * a_signal / ((sigma_e_sig + sigma_a_sig) * tau_21)
    wl = result.wavelength_nm
    dlam = np.gradient(wl)
    p_sig_t = np.array(
        [_integrated_power_w_nm(result.signal_fwd_w_nm[0, it], dlam) for it in range(result.t_s.size)]
    )
    p_peak = float(np.max(p_sig_t))
    psat_ratio = p_peak / max(p_sat, 1e-30)
    if psat_ratio < 0.3:
        sat_status = "PASS"
        sat_label = "OK: unsaturated"
    elif psat_ratio < 100:
        sat_status = "WARN"
        sat_label = "WARN: moderately saturated"
    else:
        sat_status = "FAIL"
        sat_label = "FAIL: P/Psat > 100"

    gamma_kernel = gamma_s
    gamma_sim = result.gamma_signal
    gamma_ok = abs(gamma_kernel - gamma_sim) < 0.02 or abs(gamma_kernel / max(gamma_sim, 1e-30) - 1) < 0.05

    nan_n2 = int(np.isnan(n2f).sum())
    g0 = result.g0_small_signal_np_m
    nan_g0 = int(np.isnan(g0).sum()) if g0 is not None else -1

    pump_in = result.energy_pump_in_j
    pump_out = result.energy_pump_out_j
    pump_abs = pump_in - pump_out
    ase_out = result.energy_ase_out_j
    sig_gain = result.energy_packet_out_j - result.energy_packet_in_j
    ase_ok = (
        result.energy_balance_ok
        if hasattr(result, "energy_balance_ok")
        else (ase_out <= pump_abs * 1.05 if pump_abs > 0 else ase_out < 1e-6)
    )

    stokes = hnu_p / max(hnu_s, 1e-30)
    budget = sig_gain + ase_out
    if pump_abs > 0:
        bal_frac = abs(budget - pump_abs) / pump_abs
        bal_ok = bal_frac < 0.5
    else:
        bal_frac = float("nan")
        bal_ok = True

    checks = [
        ("Conservation: N0+N2+N3 = N_tot", max_dev < 0.01, "PASS" if max_dev < 0.01 else "FAIL",
         f"Max deviation: {max_dev:.4f}%  (should be < 0.01%)"),
        ("QSS validity: dt_travel >> tau_32", qss_ok, "PASS" if qss_ok else "FAIL",
         f"dt_travel = {dt_travel * 1e12:.2f} ps, tau_32 = {tau_32 * 1e12:.2f} ps, ratio = {ratio_32:.2g}"),
        ("Pump absorbed vs kappa*L", abs(got_abs_frac - expected_abs_frac) < 0.05,
         "PASS" if abs(got_abs_frac - expected_abs_frac) < 0.05 else "FAIL",
         f"expected {expected_abs_frac * 100:.2f}% from kappa*L, got {got_abs_frac * 100:.2f}% from P(L)/P(0)"),
        ("Inversion at signal arrival vs beta_t", inv_ok, "PASS" if inv_ok else "WARN",
         f"N2(z=0,t_ss)={n2_at_ss * 100:.2f}%, beta_t={beta_t * 100:.2f}% "
         f"[{'ABOVE' if inv_ok else 'BELOW'} transparency by {abs(inv_margin):.2f} pp]"),
        ("Signal saturation", psat_ratio < 100, sat_status,
         f"P_sig_peak={p_peak:.4g} W, P_sat={p_sat:.4g} W, ratio={psat_ratio:.4g} [{sat_label}]"),
        ("Gamma_s consistency", gamma_ok, "PASS" if gamma_ok else "FAIL",
         f"Gamma_s in gain tables: {gamma_kernel:.6g}, result.gamma_signal: {gamma_sim:.6g} "
         f"[{'CONSISTENT' if gamma_ok else 'MISMATCH'}]"),
        ("No NaN in populations", nan_n2 == 0, "PASS" if nan_n2 == 0 else "FAIL",
         f"NaN count in N2 field: {nan_n2}"),
        ("No NaN in gain table", nan_g0 == 0 if nan_g0 >= 0 else True,
         "PASS" if nan_g0 == 0 else ("FAIL" if nan_g0 > 0 else "N/A"),
         f"NaN count in g0 field: {nan_g0 if nan_g0 >= 0 else 'n/a'}"),
        ("ASE energy < pump absorbed", ase_ok, "PASS" if ase_ok else "FAIL",
         f"ASE out = {ase_out * 1e3:.4f} mJ, pump absorbed = {pump_abs * 1e3:.4f} mJ "
         f"[{'OK' if ase_ok else 'VIOLATION'}]"),
        ("Energy budget (rough)", bal_ok, "PASS" if bal_ok else "FAIL",
         f"pump absorbed = {pump_abs * 1e3:.4f} mJ, sig_gain+ASE = {budget * 1e3:.4f} mJ, "
         f"Stokes hnu_p/hnu_s = {stokes:.4f}, imbalance = {bal_frac * 100:.2f}% "
         f"[{'BALANCED' if bal_ok else 'IMBALANCED'}]"),
    ]

    n_pass = sum(1 for _, ok, status, _ in checks if status == "PASS")
    n_warn = sum(1 for _, ok, status, _ in checks if status == "WARN")
    n_fail = sum(1 for _, ok, status, _ in checks if status == "FAIL")

    lines = [
        "",
        "=" * 80,
        "PHYSICS SANITY CHECKS",
        "=" * 80,
    ]
    for name, _ok, status, detail in checks:
        lines.append(f"  [{status}] {name}")
        lines.append(f"              {detail}")
        lines.append("")
    lines.append(f"  Summary: {n_pass}/{len(checks)} checks passed. {n_warn} warnings. {n_fail} failures.")
    return "\n".join(lines)


def _populations_after_pump_section(
    result: FiberCPAResult,
    cfg: FiberCPAConfig,
    *,
    it_ss: int,
) -> str:
    pops = result.populations_after_pump
    if pops is None:
        return (
            "\n"
            + "=" * 80
            + "\nPOPULATIONS AFTER PUMP PASS (every z)\n"
            + "=" * 80
            + "\n  (Not stored for this run — re-run with diagnostics export enabled.)\n"
        )

    n_tot = cfg.yb_concentration_m3
    t_ss = float(result.t_s[it_ss])
    lines = [
        "",
        "=" * 80,
        "POPULATIONS AFTER PUMP PASS (every z, before signal/ASE march)",
        "=" * 80,
        f"  Sample time t_ss = {t_ss * 1e6:.4f} µs  (just before chirped packet)",
        f"  N_tot = {n_tot:.6e} m⁻³",
        "",
        "Columns: z, N0/N_tot, N2/N_tot, N3/N_tot, N0, N2, N3 (m⁻³), "
        "N2/(N0+N2), P_pump(W) at t_ss",
        "",
        f"{'z(m)':>10} {'N0_frac':>10} {'N2_frac':>10} {'N3_frac':>10} "
        f"{'N0(m^-3)':>12} {'N2(m^-3)':>12} {'N3(m^-3)':>12} {'inv':>9} {'P_pump':>10}",
    ]
    p_pump = result.pump_fwd_w[:, it_ss]
    for iz in range(result.z_m.size):
        n0f = float(pops.n0[iz, it_ss])
        n2f = float(pops.n2[iz, it_ss])
        n3f = float(pops.n3[iz, it_ss])
        inv = n2f / max(n0f + n2f, 1e-30)
        lines.append(
            f"{result.z_m[iz]:10.5f} {n0f:10.6f} {n2f:10.6f} {n3f:10.6f} "
            f"{n0f * n_tot:12.4e} {n2f * n_tot:12.4e} {n3f * n_tot:12.4e} "
            f"{inv:9.5f} {p_pump[iz]:10.3f}"
        )
    return "\n".join(lines)


def _cross_sections_per_z_section(result: FiberCPAResult, cfg: FiberCPAConfig) -> str:
    wl = result.wavelength_nm
    sa = result.sigma_abs_signal_m2
    se = result.sigma_em_signal_m2
    ic = int(np.argmin(np.abs(wl - cfg.signal.center_wavelength_nm)))
    sa_c = float(sa[ic])
    se_c = float(se[ic])
    sigma_p = result.sigma_abs_pump_m2
    sigma_ep = result.sigma_em_pump_m2 or 0.0

    lines = [
        "",
        "=" * 80,
        "ABSORPTION / EMISSION CROSS SECTIONS (used in simulation)",
        "=" * 80,
        "  Material σ(λ) tabulation; same at every z (no z-dependent bleaching in σ).",
        f"  Pump wavelength: {result.pump_wavelength_nm:.2f} nm",
        f"  Signal grid: {wl[0]:.2f} – {wl[-1]:.2f} nm, {wl.size} points",
        "",
        "--- Spectral grid (signal band, all λ steps) ---",
        f"{'wl(nm)':>10} {'sigma_abs(m^2)':>16} {'sigma_em(m^2)':>16}",
    ]
    for i in range(wl.size):
        lines.append(f"{wl[i]:10.3f} {sa[i]:16.6e} {se[i]:16.6e}")

    lines.extend(
        [
            "",
            "--- Per z slice (σ constant vs z; listed at each step for audit) ---",
            f"  Signal center λ = {cfg.signal.center_wavelength_nm:.2f} nm",
            f"  Pump: sigma_abs = {sigma_p:.6e} m², sigma_em = {sigma_ep:.6e} m²",
            "",
            f"{'iz':>4} {'z(m)':>10} {'sig_a@ctr':>14} {'sig_e@ctr':>14} "
            f"{'sig_a_pump':>14} {'sig_e_pump':>14}",
        ]
    )
    for iz in range(result.z_m.size):
        lines.append(
            f"{iz:4d} {result.z_m[iz]:10.5f} {sa_c:14.6e} {se_c:14.6e} "
            f"{sigma_p:14.6e} {sigma_ep:14.6e}"
        )
    return "\n".join(lines)


def _tables_section(tab: DiagnosticsTables, dz: float) -> str:
    lines = [
        "",
        "=" * 80,
        "PER-Z AFTER FULL RUN (pump + signal + ASE at t_ss)",
        "=" * 80,
        f"  t_ss = {tab.steady_state_time_s * 1e6:.4f} µs  (index for pump-only inversion)",
        f"  t_pulse = {tab.pulse_time_s * 1e9:.4f} ns  (first pulse peak, for G_pulse)",
        "",
        "Columns:",
        "  z, N0, N2, N3 (fractions), inv=N2/(N0+N2), P_pump(W), T_pump(step),",
        "  alpha_p(Np/m), alpha_p(dB/m), g0_center(m^-1), g0_mean(m^-1),",
        "  G0=exp(g0_mean*dz), G_pulse(at t_pulse), P_signal(W @ pulse)",
        "",
        f"{'z(m)':>10} {'N0':>9} {'N2':>9} {'N3':>9} {'inv':>9} {'P_pump':>10} {'T_pump':>9} "
        f"{'a_Np/m':>10} {'a_dB/m':>8} {'g0_ctr':>10} {'g0_avg':>10} {'G0':>9} {'G_pulse':>9} {'P_sig':>10}",
    ]
    nz = tab.z_m.size
    for iz in range(nz):
        g0 = tab.G0_power_per_step[iz] if iz < nz - 1 else float("nan")
        gp = tab.G_pulse_power_per_step[iz] if iz < nz - 1 else float("nan")
        lines.append(
            f"{tab.z_m[iz]:10.5f} {tab.n0_fraction[iz]:9.5f} {tab.n2_fraction[iz]:9.5f} "
            f"{tab.n3_fraction[iz]:9.5f} {tab.inversion_n2_over_n0_n2[iz]:9.5f} "
            f"{tab.pump_power_w[iz]:10.3f} {tab.pump_step_transmission[iz]:9.5f} "
            f"{tab.alpha_p_np_m[iz]:10.4f} {tab.alpha_p_db_m[iz]:8.3f} "
            f"{tab.g0_center_m[iz]:10.4e} {tab.g0_spectral_mean_m[iz]:10.4e} "
            f"{g0:9.5f} {gp:9.5f} {tab.pulse_power_w[iz]:10.4e}"
        )

    lines.extend(
        [
            "",
            "Integrated metrics:",
            f"  Product G0 over all steps     {np.prod(tab.G0_power_per_step[:-1]):.6g}",
            f"  Product G_pulse over steps    {np.prod(tab.G_pulse_power_per_step[:-1]):.6g}",
            f"  Pump transmission z=0 -> L   {tab.pump_power_w[-1]/max(tab.pump_power_w[0],1e-30):.6g}",
        ]
    )
    return "\n".join(lines)


def build_diagnostics_report(
    cfg: FiberCPAConfig,
    result: FiberCPAResult,
    material: Material,
    dopant: Any,
    *,
    gamma_p: float,
    gamma_s: float,
    a_core: float,
    a_pump: float,
    dz: float,
    dt_travel: float,
    run_label: str = "pulsed",
    cw_reference: FiberCPAResult | None = None,
    b_integral: BIntegralResult | None = None,
    l_passive_before_m: float = 0.0,
    l_passive_after_m: float = 0.0,
) -> str:
    tab = compute_diagnostics_tables(result, cfg, gamma_p=gamma_p, gamma_s=gamma_s, dz=dz)
    it_ss = steady_state_time_index(result.t_s, cfg.signal)
    t_ss = float(result.t_s[it_ss])
    n_tot = cfg.yb_concentration_m3
    lifetimes = result.four_level_lifetimes or cfg.four_level
    sigma_p = result.sigma_abs_pump_m2
    sigma_ep = result.sigma_em_pump_m2
    hnu_p = float(material.photon_energy_j(result.pump_wavelength_nm)[0])
    hnu_s_arr = material.photon_energy_j(result.wavelength_nm)
    sigma_a = result.sigma_abs_signal_m2
    sigma_e = result.sigma_em_signal_m2
    ic = int(np.argmin(np.abs(result.wavelength_nm - cfg.signal.center_wavelength_nm)))
    sigma_a_c = float(sigma_a[ic])
    sigma_e_c = float(sigma_e[ic])
    hnu_s_c = float(hnu_s_arr[ic])
    a_signal = result.a_signal_m2 if result.a_signal_m2 > 0 else (
        result.signal_mode_area_m2 if result.signal_mode_area_m2 > 0 else a_core
    )
    a_mode_m2 = signal_mode_area_m2(
        core_radius_m=cfg.core_diameter_um / 2 * 1e-6,
        na=cfg.core_na,
        wavelength_m=cfg.signal.center_wavelength_nm * 1e-9,
    )
    kappa_db = cfg.pump_absorption_db_per_m
    if kappa_db is None and dopant is not None and hasattr(dopant, "alpha_db_per_m"):
        kappa_db = float(dopant.alpha_db_per_m)

    parts = [
        "Laser Sim — CPA fiber amplifier diagnostics",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Run mode: {run_label}",
        "",
        _equations_section(),
        _coefficients_section(
            cfg, result, dopant,
            gamma_p=gamma_p, gamma_s=gamma_s,
            a_core=a_core, a_mode_m2=a_mode_m2, a_pump=a_pump, dz=dz, dt_travel=dt_travel,
            material=material,
        ),
        _time_grid_section(result.t_s, cfg, dt_travel, material),
        _time_dynamics_section(
            result, cfg, material, n_tot, gamma_p, gamma_s, a_pump, sigma_p, hnu_p, lifetimes
        ),
        _coefficient_audit_section(
            result, cfg, n_tot, gamma_p, gamma_s, a_pump, a_signal,
            sigma_p, sigma_ep, hnu_p, hnu_s_arr, sigma_a, sigma_e,
            lifetimes, it_ss, iz=0, dz=dz,
        ),
        _crosssection_verification_section(
            material,
            result.pump_wavelength_nm,
            cfg.signal.center_wavelength_nm,
            n_tot,
            gamma_p,
            kappa_db,
            result,
        ),
        _sanity_checks_section(
            result, cfg, n_tot, gamma_p, gamma_s, a_pump, a_signal,
            sigma_p, hnu_p, sigma_a_c, sigma_e_c, hnu_s_c, lifetimes, t_ss, it_ss, dz,
        ),
        *(
            [
                _b_integral_section(
                    b_integral,
                    l_passive_before_m=l_passive_before_m,
                    l_active_m=cfg.fiber_length_m,
                    l_passive_after_m=l_passive_after_m,
                )
            ]
            if b_integral is not None
            else []
        ),
        _populations_after_pump_section(result, cfg, it_ss=it_ss),
        _cross_sections_per_z_section(result, cfg),
        _tables_section(tab, dz),
    ]
    if cw_reference is not None:
        e_in = cw_reference.energy_packet_in_j
        e_out = cw_reference.energy_packet_out_j
        g = e_out / max(e_in, 1e-30)
        parts.extend(
            [
                "",
                "=" * 80,
                "CW AVERAGE-POWER REFERENCE RUN",
                "=" * 80,
                f"  Signal: constant P = packet_average_power = {packet_average_power_w(cfg.signal):.6g} W",
                f"  Energy in (integrated)     {e_in:.6e} J",
                f"  Energy out (integrated)    {e_out:.6e} J",
                f"  Energy gain (out/in)       {g:.6f}",
                f"  Notes: {cw_reference.notes}",
            ]
        )
    return "\n".join(parts)


def write_diagnostics_report(
    path: Path,
    text: str,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path
