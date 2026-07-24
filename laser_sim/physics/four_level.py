"""
Rate equations for temporally resolved Yb fiber amplifiers.

Default: quasi-two-level with N₁ skipped (τ₁₀ → 0, lower Stark manifold empty).
Manifolds: N₀ ground, N₃ pump band, N₂ upper laser level.
Conservation: N₀ + N₂ + N₃ = N_tot.

Datasheet pump absorption (dB/m) is used only to set N_tot via κ/σ_p — not as a
separate Beer–Lambert law in z. Pump depletion: α_p(z,t) = σ_p · N₀(z,t).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.materials.base import Material


@dataclass(frozen=True)
class FourLevelLifetimes:
    """Characteristic lifetimes (s). N₁ level omitted when ``skip_n1_level`` is True."""

    tau_32_s: float  # N₃ → N₂ (fast, ps–ns)
    tau_21_s: float  # N₂ spontaneous (Yb ~ 0.84–1 ms)
    tau_10_s: float  # unused when skip_n1_level
    beta_21: float = 1.0
    skip_n1_level: bool = True  # τ₁₀ → 0: no population in lower laser level


@dataclass
class FourLevelPopulations:
    """Population fractions at each (z, t); shape (n_z, n_t)."""

    n0: np.ndarray
    n1: np.ndarray
    n2: np.ndarray
    n3: np.ndarray

    @property
    def n2_fraction(self) -> np.ndarray:
        return self.n2


def lifetimes_from_material(material: Material) -> FourLevelLifetimes:
    return FourLevelLifetimes(
        tau_32_s=1.0e-12,
        tau_21_s=material.lifetime_s,
        tau_10_s=1.0e-9,
        beta_21=1.0,
        skip_n1_level=True,
    )


def pump_rate_per_ion(
    ip_w_m2: float,
    *,
    gamma_p: float = 1.0,
    sigma_p: float,
    sigma_ep: float,
    hnu_p: float,
) -> tuple[float, float]:
    """
    (W_p,abs, W_p,esa) per ion for dN₃/dt = W_p,abs·N₀ − W_p,esa·N₃ − N₃/τ₃₂.

    ``ip_w_m2`` is pump intensity P/A_pump (cladding: A_clad; core-pumped: A_core).
    Overlap Γ_p must not be applied again — it is already in α_p(z) via
    ``advance_pump_power_z``. Core-local rates: σ·P/(hν·A_clad) = σ·Γ_p·P/(hν·A_core).
    """
    _ = gamma_p  # kept for call-site compatibility; not multiplied into rates
    w_p = ip_w_m2 / hnu_p
    return w_p * sigma_p, w_p * sigma_ep


def steady_state_n2_fraction_pump(
    w_p_abs: float,
    w_p_esa: float,
    tau_21_s: float,
) -> float:
    """Upper-state fraction N₂/N_tot under pump only (includes pump stimulated emission)."""
    return w_p_abs / (w_p_abs + w_p_esa + 1.0 / max(tau_21_s, 1e-30))


def multichannel_pump_rate_per_ion(
    channel_intensities: list[tuple[float, float, float, float]],
) -> tuple[float, float]:
    """
    Sum pump absorption/ESA rates across pump channels.

    Each entry is (ip_w_m2, sigma_p, sigma_ep, hnu_p) with ip = P/A_pump for that channel.
  Γ_p is applied only in z-propagation (``advance_pump_power_z``), not here.
    """
    w_pa, w_pe = 0.0, 0.0
    for ip, sigma_p, sigma_ep, hnu_p in channel_intensities:
        if ip <= 0.0 or hnu_p <= 0.0:
            continue
        flux = ip / hnu_p
        w_pa += flux * sigma_p
        w_pe += flux * sigma_ep
    return w_pa, w_pe


def _pump_ip_at_index(
    it: int,
    t: np.ndarray,
    p_pf: np.ndarray,
    p_pb: np.ndarray,
    a_pump_m2: float,
) -> float:
    if a_pump_m2 <= 0.0:
        return 0.0
    if it <= 0:
        # Halved t=0 intensity: matches the warm-start seed convention of
        # march_populations_pump_qss (the validated single-channel baseline).
        return float((p_pf[0] + p_pb[0]) * 0.5 / a_pump_m2)
    # Trapezoidal average of total (fwd+bwd) power over [it-1, it] — same
    # convention as the single-channel march in march_populations_pump_qss.
    p_mid = 0.5 * (p_pf[it - 1] + p_pb[it - 1] + p_pf[it] + p_pb[it])
    return float(p_mid / a_pump_m2)


def pump_rates_from_channel_rows(
    it: int,
    t: np.ndarray,
    channel_rows: list[tuple[np.ndarray, np.ndarray, float, float, float, float]],
) -> tuple[float, float]:
    """(w_p_abs, w_p_esa) from per-channel (p_fwd, p_bwd, σ_p, σ_ep, hν, A_pump) rows at z."""
    intensities: list[tuple[float, float, float, float]] = []
    for p_f, p_b, sigma_p, sigma_ep, hnu_p, a_pump in channel_rows:
        ip = _pump_ip_at_index(it, t, p_f, p_b, a_pump)
        intensities.append((ip, sigma_p, sigma_ep, hnu_p))
    return multichannel_pump_rate_per_ion(intensities)


def guided_fraction_into_mode(eta_guided: float, gamma_s: float) -> float:
    """Fraction of N₂ spontaneous decay coupled into the guided signal/ASE mode."""
    return float(np.clip(eta_guided * gamma_s, 0.0, 1.0))


def field_rates_per_ion(
    p_row_w_nm: np.ndarray,
    dlam: np.ndarray,
    sigma_e: np.ndarray,
    sigma_a: np.ndarray,
    hnu: np.ndarray,
    *,
    gamma_s: float,
    a_signal: float,
) -> tuple[float, float]:
    """(W_se, W_abs) coupling coefficients (s⁻¹) from one power-density row (W/nm)."""
    isig = np.clip(p_row_w_nm / max(a_signal, 1e-30), 0.0, 1e18)
    w_se = gamma_s * float(np.sum(sigma_e * isig * dlam / hnu))
    w_abs = gamma_s * float(np.sum(sigma_a * isig * dlam / hnu))
    return w_se, w_abs


def combined_field_rates(
    p_sig_row_w_nm: np.ndarray,
    p_ase_row_w_nm: np.ndarray,
    dlam: np.ndarray,
    sigma_e: np.ndarray,
    sigma_a: np.ndarray,
    hnu: np.ndarray,
    *,
    gamma_s: float,
    a_signal: float,
    couple_signal: bool = True,
    couple_ase: bool = True,
) -> tuple[float, float]:
    """Stimulated-emission / absorption rates from signal and ASE (incoherent sum)."""
    w_se, w_abs = 0.0, 0.0
    if couple_signal and p_sig_row_w_nm.size:
        ws, wa = field_rates_per_ion(
            p_sig_row_w_nm, dlam, sigma_e, sigma_a, hnu, gamma_s=gamma_s, a_signal=a_signal
        )
        w_se += ws
        w_abs += wa
    if couple_ase and p_ase_row_w_nm.size:
        wa_se, wa_ab = field_rates_per_ion(
            p_ase_row_w_nm, dlam, sigma_e, sigma_a, hnu, gamma_s=gamma_s, a_signal=a_signal
        )
        w_se += wa_se
        w_abs += wa_ab
    return w_se, w_abs


def level_derivatives(
    n0: float,
    n2: float,
    n3: float,
    n_tot: float,
    *,
    w_p_abs: float,
    w_p_esa: float,
    w_se: float,
    w_abs: float,
    lifetimes: FourLevelLifetimes,
) -> tuple[float, float, float]:
    """Time derivatives (m⁻³ s⁻¹) for N₀, N₂, N₃."""
    n0 = float(np.clip(n0, 0.0, n_tot))
    n2 = float(np.clip(n2, 0.0, n_tot))
    n3 = float(np.clip(n3, 0.0, n_tot))
    lt = lifetimes

    dn3 = w_p_abs * n0 - w_p_esa * n3 - n3 / lt.tau_32_s
    if lt.skip_n1_level:
        # Stimulated emission N₂→N₀; absorption N₀→N₂
        dn2 = n3 / lt.tau_32_s - w_se * n2 + w_abs * n0 - n2 / lt.tau_21_s
        dn0 = -dn2 - dn3
    else:
        n1 = max(n_tot - n0 - n2 - n3, 0.0)
        dn2 = n3 / lt.tau_32_s - w_se * n2 - w_abs * n1 - n2 / lt.tau_21_s
        dn1 = w_se * n2 + w_abs * n1 + lt.beta_21 * n2 / lt.tau_21_s - n1 / lt.tau_10_s
        dn0 = -(dn1 + dn2 + dn3)

    return dn0, dn2, dn3


def _rk4_step_3level(
    n0: float,
    n2: float,
    n3: float,
    n_tot: float,
    dt: float,
    *,
    w_p_abs: float,
    w_p_esa: float,
    w_se: float,
    w_abs: float,
    lifetimes: FourLevelLifetimes,
) -> tuple[float, float, float]:
    def f(n0i: float, n2i: float, n3i: float) -> tuple[float, float, float]:
        return level_derivatives(
            n0i, n2i, n3i, n_tot,
            w_p_abs=w_p_abs, w_p_esa=w_p_esa, w_se=w_se, w_abs=w_abs,
            lifetimes=lifetimes,
        )

    k1 = f(n0, n2, n3)
    k2 = f(n0 + 0.5 * dt * k1[0], n2 + 0.5 * dt * k1[1], n3 + 0.5 * dt * k1[2])
    k3 = f(n0 + 0.5 * dt * k2[0], n2 + 0.5 * dt * k2[1], n3 + 0.5 * dt * k2[2])
    k4 = f(n0 + dt * k3[0], n2 + dt * k3[1], n3 + dt * k3[2])
    n0n = n0 + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    n2n = n2 + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    n3n = n3 + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return _enforce_3level(n0n, n2n, n3n, n_tot)


def _enforce_3level(n0: float, n2: float, n3: float, n_tot: float) -> tuple[float, float, float]:
    n0 = max(n0, 0.0)
    n2 = max(n2, 0.0)
    n3 = max(n3, 0.0)
    s = n0 + n2 + n3
    if s > n_tot:
        scale = n_tot / s
        n0, n2, n3 = n0 * scale, n2 * scale, n3 * scale
    return n0, n2, n3


def _substeps_for_interval(dt: float, lifetimes: FourLevelLifetimes, max_sub: int = 64) -> int:
    if dt <= 0.0:
        return 1
    tau_fast = lifetimes.tau_32_s
    if dt > 50.0 * tau_fast:
        return int(min(12, max_sub))
    n = int(np.ceil(dt / max(tau_fast / 5.0, 1e-18)))
    return int(np.clip(n, 1, max_sub))


def march_level_interval(
    n0: float,
    n2: float,
    n3: float,
    n_tot: float,
    dt: float,
    *,
    ip_w_m2: float,
    p_sig_row: np.ndarray,
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
    lifetimes: FourLevelLifetimes,
    include_signal: bool = True,
    p_ase_row: np.ndarray | None = None,
    couple_ase: bool = True,
    pump_channel_rows: list[tuple[np.ndarray, np.ndarray, float, float, float, float]]
    | None = None,
    pump_time_index: int = 0,
    pump_time_s: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Integrate rate equations over Δt at fixed z (signal + ASE + pump + spontaneous)."""
    n_sub = _substeps_for_interval(dt, lifetimes)
    dt_sub = dt / n_sub
    if pump_channel_rows is not None and pump_time_s is not None:
        w_p_abs, w_p_esa = pump_rates_from_channel_rows(
            pump_time_index, pump_time_s, pump_channel_rows
        )
    else:
        w_p_abs, w_p_esa = pump_rate_per_ion(
            ip_w_m2, gamma_p=gamma_p, sigma_p=sigma_p, sigma_ep=sigma_ep, hnu_p=hnu_p
        )
    p_ase = p_ase_row if p_ase_row is not None else np.zeros_like(p_sig_row)
    w_se, w_abs = combined_field_rates(
        p_sig_row if include_signal else np.zeros_like(p_sig_row),
        p_ase,
        dlam,
        sigma_e,
        sigma_a,
        hnu,
        gamma_s=gamma_s,
        a_signal=a_signal,
        couple_signal=include_signal,
        couple_ase=couple_ase and bool(p_ase.size),
    )

    for _ in range(n_sub):
        n0, n2, n3 = _rk4_step_3level(
            n0, n2, n3, n_tot, dt_sub,
            w_p_abs=w_p_abs, w_p_esa=w_p_esa, w_se=w_se, w_abs=w_abs,
            lifetimes=lifetimes,
        )
    return n0, n2, n3


def march_level_interval_batch(
    n0_batch: np.ndarray,
    n2_batch: np.ndarray,
    n_tot: float,
    dt: float,
    *,
    ip_batch: np.ndarray,
    p_sig_batch: np.ndarray,
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
    lifetimes: FourLevelLifetimes,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorised quasi-2L RK4 over a batch of active time bins.

    Returns updated (n0_batch, n2_batch). N3 is held at 0 (QSS during signal).
    """
    w_p = ip_batch / hnu_p
    w_pa = w_p * sigma_p

    i_sig = p_sig_batch / a_signal
    fac = gamma_s * dlam / hnu
    w_se = np.sum(sigma_e * i_sig * fac, axis=1)
    w_abs = np.sum(sigma_a * i_sig * fac, axis=1)

    tau21 = lifetimes.tau_21_s

    def deriv(n0: np.ndarray, n2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        dn2 = -w_se * n2 + (w_abs + w_pa) * n0 - n2 / tau21
        return -dn2, dn2

    n0 = n0_batch.copy()
    n2 = n2_batch.copy()
    k0_0, k0_2 = deriv(n0, n2)
    k1_0, k1_2 = deriv(n0 + 0.5 * dt * k0_0, n2 + 0.5 * dt * k0_2)
    k2_0, k2_2 = deriv(n0 + 0.5 * dt * k1_0, n2 + 0.5 * dt * k1_2)
    k3_0, k3_2 = deriv(n0 + dt * k2_0, n2 + dt * k2_2)

    n0n = n0 + (dt / 6.0) * (k0_0 + 2 * k1_0 + 2 * k2_0 + k3_0)
    n2n = n2 + (dt / 6.0) * (k0_2 + 2 * k1_2 + 2 * k2_2 + k3_2)
    n0n = np.maximum(n0n, 0.0)
    n2n = np.maximum(n2n, 0.0)
    total = n0n + n2n
    over = total > n_tot
    scale = np.where(over, n_tot / total, 1.0)
    return n0n * scale, n2n * scale


def march_populations_pump_qss(
    t: np.ndarray,
    p_pf: np.ndarray,
    p_pb: np.ndarray,
    *,
    n_tot: float,
    a_pump: float,
    gamma_p: float,
    sigma_p: float,
    sigma_ep: float,
    hnu_p: float,
    lifetimes: FourLevelLifetimes,
    initial_n2_fraction: float = 0.0,
    pump_channel_rows: list[tuple[np.ndarray, np.ndarray, float, float, float, float]]
    | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pump-only causal march; N₃ QSS; N₁ = 0."""
    nt = t.size
    lt = lifetimes
    n1 = np.zeros(nt, dtype=np.float64)
    n2 = np.zeros(nt, dtype=np.float64)
    n3 = np.zeros(nt, dtype=np.float64)
    n0 = np.zeros(nt, dtype=np.float64)

    f2 = float(np.clip(initial_n2_fraction, 0.0, 1.0))
    n0i = float(n_tot) * (1.0 - f2)
    n2i = float(n_tot) * f2
    n3i = 0.0
    if f2 <= 0.0 and nt > 0:
        if pump_channel_rows:
            w0a, w0e = pump_rates_from_channel_rows(0, t, pump_channel_rows)
        else:
            # Warm-start seed uses half the t=0 intensity (historical convention;
            # the validated baseline and the energy-budget check are calibrated
            # to it — see _pump_ip_at_index for the matching multichannel seed).
            ip0 = float((p_pf[0] + p_pb[0]) * 0.5 / a_pump)
            w0a, w0e = (
                pump_rate_per_ion(
                    ip0, gamma_p=gamma_p, sigma_p=sigma_p, sigma_ep=sigma_ep, hnu_p=hnu_p
                )
                if ip0 > 0.0
                else (0.0, 0.0)
            )
        if w0a > 0.0 or w0e > 0.0:
            n2i = steady_state_n2_fraction_pump(w0a, w0e, lt.tau_21_s) * n_tot
            n3i = w0a * n0i * lt.tau_32_s / (1.0 + w0e * lt.tau_32_s)
            n2i = min(n2i, n_tot - n3i)
            n0i = max(n_tot - n2i - n3i, 0.0)
    for it in range(nt):
        if it > 0:
            dt_i = float(t[it] - t[it - 1])
            if pump_channel_rows:
                w_p_abs, w_p_esa = pump_rates_from_channel_rows(it, t, pump_channel_rows)
            else:
                ip = float((p_pf[it - 1] + p_pb[it - 1] + p_pf[it] + p_pb[it]) * 0.5 / a_pump)
                w_p_abs, w_p_esa = pump_rate_per_ion(
                    ip, gamma_p=gamma_p, sigma_p=sigma_p, sigma_ep=sigma_ep, hnu_p=hnu_p
                )
            n0i = max(n_tot - n2i - n3i, 0.0)
            # N3 QSS (τ₃₂ ≪ Δt): unchanged, N₃ is small
            n3_ss = w_p_abs * n0i * lt.tau_32_s / (1.0 + w_p_esa * lt.tau_32_s)
            # N2 self-consistent steady state: solve N0+N2=N_tot simultaneously.
            # n2_ss = W·τ / (1 + W·τ) · N_tot  -- saturates to N_tot at high pump.
            # Old formula n2_ss = W·N0·τ is wrong when W·τ >> 1 (pegs N2 to N_tot
            # via N0→0 lag, overshooting in one step).
            n2_ss = steady_state_n2_fraction_pump(
                w_p_abs, w_p_esa, lt.tau_21_s
            ) * n_tot
            n2i = n2_ss - (n2_ss - n2i) * np.exp(-dt_i / lt.tau_21_s)
            # Guard: N₂ cannot exceed (N_tot − N₃_ss)
            n2i = min(n2i, n_tot - n3_ss)
            n3i = n3_ss
            n0i = max(n_tot - n2i - n3i, 0.0)
        n0[it], n1[it], n2[it], n3[it] = n0i, 0.0, n2i, n3i

    return n0, n1, n2, n3


def advance_pump_power_z(
    p_in: np.ndarray,
    n0: np.ndarray,
    sigma_p: float,
    dz: float,
    *,
    gamma_p: float = 1.0,
    n2: np.ndarray | None = None,
    sigma_ep: float | None = None,
) -> np.ndarray:
    """
    Deplete pump along +z (m⁻¹).

    α_p = Γ_p · (σ_p·N₀ − σ_ep·N₂) when ``n2`` and ``sigma_ep`` are given;
    otherwise α_p = Γ_p · σ_p · N₀.
    """
    alpha = gamma_p * sigma_p * np.maximum(n0, 0.0)
    if n2 is not None and sigma_ep is not None:
        alpha = alpha - gamma_p * sigma_ep * np.maximum(n2, 0.0)
    return np.maximum(p_in * np.exp(-alpha * dz), 0.0)


def gain_coefficient_m(
    n0: np.ndarray,
    n2: np.ndarray,
    sigma_a: np.ndarray,
    sigma_e: np.ndarray,
    *,
    gamma_s: float,
    n1: np.ndarray | None = None,
) -> np.ndarray:
    """g(λ) = Γ [σ_e N₂ − σ_a N₁] or quasi-2L: σ_e N₂ − σ_a N₀."""
    if n1 is not None and np.any(n1):
        return gamma_s * (sigma_e[None, :] * n2[:, None] - sigma_a[None, :] * n1[:, None])
    return gamma_s * (sigma_e[None, :] * n2[:, None] - sigma_a[None, :] * n0[:, None])


def spontaneous_power_w_per_nm(
    n2: float,
    sigma_e: np.ndarray,
    hnu: np.ndarray,
    wl_nm: np.ndarray,
    *,
    tau_21_s: float,
    eta_guided: float,
    gamma_s: float,
) -> np.ndarray:
    """
    Guided spontaneous emission spectral density (W/nm) from N₂ decay.

    ∫ P_sp dλ ≈ η Γ_s n₂ h⟨ν⟩ / τ₂₁  (W/m³); multiply by A_signal·dz for ASE increment.
    """
    norm = float(np.trapezoid(sigma_e, wl_nm))
    if norm <= 0:
        norm = float(np.sum(sigma_e)) or 1.0
    return (
        eta_guided
        * gamma_s
        * max(n2, 0.0)
        * hnu
        * sigma_e
        / (max(tau_21_s, 1e-30) * norm)
    )


def populations_to_fractions(
    n0: np.ndarray, n1: np.ndarray, n2: np.ndarray, n3: np.ndarray, n_tot: float
) -> FourLevelPopulations:
    return FourLevelPopulations(
        n0=n0 / n_tot,
        n1=n1 / n_tot,
        n2=n2 / n_tot,
        n3=n3 / n_tot,
    )
