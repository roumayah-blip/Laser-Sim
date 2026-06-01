"""
Lumped per-period steady-state for rep-rate CW-pumped fiber amplifiers.

This mirrors the standard RP-Fiber-Power-style approach:
    1. Apply pulse extraction along z (Frantz–Nodvik per slab) → N2(z) after pulse.
    2. Pump-rebuild during the rep period: N2(z) relaxes toward CW steady state
       with effective lifetime 1 / (1/τ21 + W_p).
    3. Iterate until N2(z) just-before-pulse converges between consecutive periods.

The converged N2(z) is then used to warm-start ONE time-resolved period of the
full quasi-2L simulation. This skips the long (often >40-period) startup
transient that the time-resolved march would otherwise have to walk through.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.physics.progress import ProgressCallback, emit_progress


@dataclass(frozen=True)
class RepRateWarmupResult:
    n2_pre_packet: np.ndarray         # (nz,) N2 just before the packet, steady state
    n2_post_packet: np.ndarray        # (nz,) N2 immediately after extraction
    n_iter: int
    residual: float
    converged: bool
    e_packet_out_j: float             # output packet energy in steady state (per period)
    e_packet_in_j: float              # input packet energy each period (lumped model)
    small_signal_gain_db: float       # integrated G0 (dB) using N2_pre_packet
    notes: str
    e_out_per_iter_j: np.ndarray          # low-res packet energy out, one per warmup iteration
    gain_per_iter: np.ndarray             # E_out/E_in per iteration
    residual_per_iter: np.ndarray         # ‖ΔN₂‖/‖N₂‖ after each iteration


def _frantz_nodvik_extraction(
    n2: np.ndarray,
    n_tot: float,
    dz: float,
    *,
    e_in_j: float,
    sigma_a_sig: float,
    sigma_e_sig: float,
    gamma_s: float,
    a_signal_m2: float,
    hnu_s: float,
) -> tuple[np.ndarray, float]:
    """Propagate a pulse of energy ``e_in_j`` through slabs; return (N2_after, E_out)."""
    e_sat_density = hnu_s / max((sigma_a_sig + sigma_e_sig) * gamma_s, 1e-60)  # J/m²
    e_sat = e_sat_density * a_signal_m2  # J

    n2_out = n2.copy()
    e = float(e_in_j)
    nz = n2.size
    for iz in range(nz - 1):
        n0_iz = max(n_tot - n2_out[iz], 0.0)
        g0_dz = gamma_s * (sigma_e_sig * n2_out[iz] - sigma_a_sig * n0_iz) * dz
        g0_dz = float(np.clip(g0_dz, -50.0, 50.0))
        e_norm = e / max(e_sat, 1e-60)
        if e_norm > 50.0:
            # E >> E_sat: ln(1 + (exp(x)-1)*G) ≈ x + ln(G); only stored energy can be
            # extracted in saturated regime, so cap by available inversion.
            e_out = e_sat * (e_norm + g0_dz)
        else:
            e_out = e_sat * float(np.log1p(np.expm1(e_norm) * np.exp(g0_dz)))
        e_out = max(e_out, e)  # extraction non-negative
        de = e_out - e
        # ΔN2 lost per unit volume in this slab
        dn2 = de / max(a_signal_m2 * dz * hnu_s, 1e-60)
        # Cannot extract more inversion than is present
        dn2 = min(dn2, n2_out[iz])
        # Update photon count accordingly if we clipped
        de = dn2 * a_signal_m2 * dz * hnu_s
        n2_out[iz] = max(n2_out[iz] - dn2, 0.0)
        e = e + de
    return n2_out, e


def solve_rep_rate_steady_state_inversion(
    *,
    z_m: np.ndarray,
    pump_power_z_w: np.ndarray,
    a_pump_m2: float,
    sigma_p: float,
    gamma_p: float,
    hnu_p: float,
    sigma_ep: float | None = None,
    sigma_a_sig: float,
    sigma_e_sig: float,
    gamma_s: float,
    a_signal_m2: float,
    hnu_s: float,
    packet_energy_j: float,
    rep_rate_hz: float,
    tau_21_s: float,
    n_tot: float,
    initial_n2_fraction: float = 0.0,
    max_iter: int = 300,
    tol: float = 1e-4,
    progress_callback: ProgressCallback | None = None,
    progress_base: float = 0.05,
    progress_span: float = 0.25,
) -> RepRateWarmupResult:
    """Iterate per-period dynamics until ``N2(z)`` just-before-packet converges."""
    z = np.asarray(z_m, dtype=np.float64)
    nz = z.size
    if nz < 2:
        raise ValueError("solve_rep_rate_steady_state_inversion needs >=2 z slices")
    dz = float(z[1] - z[0])
    t_rep = 1.0 / max(rep_rate_hz, 1e-30)

    pump_z = np.asarray(pump_power_z_w, dtype=np.float64)
    if pump_z.size != nz:
        raise ValueError("pump_power_z_w must have shape (nz,)")
    ip_z = pump_z / max(a_pump_m2, 1e-60)
    sigma_ep_eff = sigma_p if sigma_ep is None else sigma_ep
    w_p_flux = ip_z / max(hnu_p, 1e-60)
    w_p_abs = sigma_p * w_p_flux
    w_p_esa = sigma_ep_eff * w_p_flux
    n2_ss_z = (
        w_p_abs / (w_p_abs + w_p_esa + 1.0 / max(tau_21_s, 1e-30))
    ) * n_tot
    tau_eff = 1.0 / np.maximum(1.0 / tau_21_s + w_p_abs, 1e-30)
    rebuild = np.exp(-t_rep / tau_eff)

    if initial_n2_fraction > 0:
        n2_pre = np.full(nz, initial_n2_fraction * n_tot, dtype=np.float64)
    else:
        n2_pre = n2_ss_z.copy()

    residual = float("inf")
    n_iter = 0
    e_out = 0.0
    n2_post = n2_pre.copy()
    e_out_hist: list[float] = []
    gain_hist: list[float] = []
    resid_hist: list[float] = []

    for it_iter in range(max_iter):
        n2_prev = n2_pre.copy()

        n2_post, e_out = _frantz_nodvik_extraction(
            n2_pre,
            n_tot,
            dz,
            e_in_j=packet_energy_j,
            sigma_a_sig=sigma_a_sig,
            sigma_e_sig=sigma_e_sig,
            gamma_s=gamma_s,
            a_signal_m2=a_signal_m2,
            hnu_s=hnu_s,
        )

        # Pump rebuild during T_rep with no signal
        n2_pre = n2_ss_z + (n2_post - n2_ss_z) * rebuild

        denom = float(np.linalg.norm(n2_prev))
        residual = float(np.linalg.norm(n2_pre - n2_prev) / max(denom, 1e-30))
        n_iter = it_iter + 1
        e_out_hist.append(e_out)
        gain_hist.append(e_out / max(packet_energy_j, 1e-30))
        resid_hist.append(residual)

        if progress_callback is not None and (
            it_iter % 5 == 0 or residual < tol or it_iter == max_iter - 1
        ):
            frac = progress_base + progress_span * min(1.0, it_iter / max(max_iter - 1, 1))
            emit_progress(
                progress_callback,
                frac,
                f"Steady-state warmup iter {it_iter + 1}: residual {residual:.2e}",
            )

        if residual < tol:
            break

    g0_int_np = float(
        np.sum(
            gamma_s
            * (sigma_e_sig * n2_pre - sigma_a_sig * (n_tot - n2_pre))
            * dz
        )
    )
    g0_db = g0_int_np * (10.0 / np.log(10.0))
    converged = residual < tol
    notes = (
        f"Rep rate {rep_rate_hz / 1e3:.2f} kHz, period {t_rep * 1e6:.3f} µs. "
        f"Warmup {'converged' if converged else 'did NOT converge'} in {n_iter} iter "
        f"(residual={residual:.3e}, tol={tol:.1e}). Small-signal gain "
        f"{g0_db:.2f} dB at steady-state inversion."
    )

    return RepRateWarmupResult(
        n2_pre_packet=n2_pre,
        n2_post_packet=n2_post,
        n_iter=n_iter,
        residual=residual,
        converged=converged,
        e_packet_out_j=e_out,
        e_packet_in_j=packet_energy_j,
        small_signal_gain_db=g0_db,
        notes=notes,
        e_out_per_iter_j=np.asarray(e_out_hist, dtype=np.float64),
        gain_per_iter=np.asarray(gain_hist, dtype=np.float64),
        residual_per_iter=np.asarray(resid_hist, dtype=np.float64),
    )


def cw_pump_power_along_z(
    *,
    z_m: np.ndarray,
    pump_power_in_w: float,
    a_pump_m2: float,
    sigma_p: float,
    gamma_p: float,
    n_tot: float,
    n2_estimate_z: np.ndarray | None = None,
) -> np.ndarray:
    """
    Estimate CW pump power along z, ground-state dominated by default.

    Beer–Lambert with α(z) = Γ_p σ_p N0(z); N0 = N_tot − N2(z) if a guess is supplied,
    else N0 = N_tot (no extraction).
    """
    z = np.asarray(z_m, dtype=np.float64)
    nz = z.size
    if nz < 2:
        return np.full(nz, pump_power_in_w, dtype=np.float64)
    dz = float(z[1] - z[0])
    n0_z = (
        np.maximum(n_tot - np.asarray(n2_estimate_z, dtype=np.float64), 0.0)
        if n2_estimate_z is not None
        else np.full(nz, n_tot, dtype=np.float64)
    )
    p = np.zeros(nz, dtype=np.float64)
    p[0] = float(pump_power_in_w)
    for iz in range(nz - 1):
        alpha = gamma_p * sigma_p * n0_z[iz]
        p[iz + 1] = max(p[iz] * np.exp(-alpha * dz), 0.0)
    return p
