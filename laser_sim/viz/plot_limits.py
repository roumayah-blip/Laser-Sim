"""Default axis ranges and plot helpers for CPA pulse visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from laser_sim.pulses.chirp import ChirpedBurstSpec


def integrated_power_vs_time(signal_w_nm: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    dlam = np.gradient(wl_nm)
    return np.sum(signal_w_nm * dlam[None, :], axis=1)


def resolve_plot_spec_from_signal(
    t_s: np.ndarray,
    signal_w_nm: np.ndarray,
    wl_nm: np.ndarray,
    fallback: ChirpedBurstSpec,
    *,
    peak_fraction: float = 1e-4,
) -> ChirpedBurstSpec:
    """
    Align ``burst_start_time_s`` with where signal power actually sits on the grid.

    Use when GUI burst delay and simulation disagree (e.g. rep-rate auto delay bug).
    """
    from dataclasses import replace

    p_t = integrated_power_vs_time(signal_w_nm, wl_nm)
    peak = float(np.max(p_t)) if p_t.size else 0.0
    if peak <= 0.0:
        return fallback

    active = np.where(p_t > peak * peak_fraction)[0]
    if active.size == 0:
        return fallback

    t_start = float(t_s[active[0]])
    t_end = float(t_s[active[-1]])
    # If spec window already overlaps the active signal, keep it.
    t_lo, t_hi = temporal_pulse_window_s(fallback)
    if t_lo <= t_end and t_hi >= t_start:
        return fallback

    burst_start = max(t_start, 0.0)
    # Preserve packet span from detected extent when possible.
    span = max(t_end - burst_start, fallback.chirp_duration_s)
    spacing = fallback.burst_spacing_s
    if fallback.burst_count > 1 and span > fallback.chirp_duration_s:
        spacing = span / max(fallback.burst_count - 1, 1)

    return replace(
        fallback,
        burst_start_time_s=burst_start,
        burst_spacing_s=max(spacing, fallback.burst_spacing_s),
    )


def packet_center_time_s(spec: ChirpedBurstSpec) -> float:
    """Center of the pulse packet on the absolute time axis (s)."""
    from laser_sim.pulses.chirp import packet_time_extent_s

    return 0.5 * (spec.burst_start_time_s + packet_time_extent_s(spec))


def temporal_half_width_s(spec: ChirpedBurstSpec, time_axis_factor: float = 3.0) -> float:
    """Half-width of temporal plot window (s), ≥ 1.5× chirp and full packet."""
    from laser_sim.pulses.chirp import packet_time_extent_s

    pulse_w = max(spec.chirp_duration_s, 0.5e-9)
    pkt0 = spec.burst_start_time_s
    pkt1 = packet_time_extent_s(spec)
    half_from_pulse = 0.5 * time_axis_factor * pulse_w
    half_from_packet = 0.5 * (pkt1 - pkt0) + 0.5 * pulse_w
    return max(half_from_pulse, half_from_packet)


def temporal_pulse_window_s(
    spec: ChirpedBurstSpec,
    time_axis_factor: float = 3.0,
) -> tuple[float, float]:
    """Absolute time window (s) for cropping plots."""
    center = packet_center_time_s(spec)
    half = temporal_half_width_s(spec, time_axis_factor)
    return center - half, center + half


def temporal_relative_window_ns(
    spec: ChirpedBurstSpec,
    time_axis_factor: float = 3.0,
) -> tuple[float, float]:
    """Relative time axis (ns) centered on packet: t=0 at packet center."""
    half = temporal_half_width_s(spec, time_axis_factor) * 1e9
    return -half, half


def sample_temporal_traces(
    t_s: np.ndarray,
    p_in: np.ndarray,
    p_out: np.ndarray,
    spec: ChirpedBurstSpec,
    time_axis_factor: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """
    Return (t_rel_ns, p_in, p_out, x_limits_ns) with t=0 at packet center.

    Window spans the full packet when wider than ``time_axis_factor`` × chirp.
    """
    center = packet_center_time_s(spec)
    t_lo, t_hi = temporal_pulse_window_s(spec, time_axis_factor)
    m = (t_s >= t_lo) & (t_s <= t_hi)
    if not np.any(m) or (
        float(np.max(p_in[m])) <= 0.0 and float(np.max(p_out[m])) <= 0.0
    ):
        peak_frac = 1e-4
        peak = float(np.max(np.maximum(p_in, p_out))) if p_in.size else 0.0
        if peak > 0.0:
            active = np.where(np.maximum(p_in, p_out) > peak * peak_frac)[0]
            if active.size:
                pad = max(spec.chirp_duration_s, 0.5e-9) * time_axis_factor
                t_lo = float(t_s[active[0]]) - pad
                t_hi = float(t_s[active[-1]]) + pad
                m = (t_s >= t_lo) & (t_s <= t_hi)
                center = 0.5 * (t_lo + t_hi)
    if not np.any(m):
        m = np.ones_like(t_s, dtype=bool)
    t_rel = (t_s[m] - center) * 1e9
    half_ns = max(float(np.max(np.abs(t_rel))) * 1.05, 0.5)
    xlim = (-half_ns, half_ns)
    return t_rel, p_in[m], p_out[m], xlim


def spectrum_plot_limits(
    spec: ChirpedBurstSpec,
    margin_factor: float = 1.25,
) -> tuple[float, float]:
    half = 0.5 * max(spec.bandwidth_nm, 1.0) * margin_factor
    return spec.center_wavelength_nm - half, spec.center_wavelength_nm + half


def pump_plot_limits(z_m: np.ndarray, alpha_db_m: np.ndarray, pump_w: np.ndarray) -> dict:
    z0, z1 = float(z_m[0]), float(z_m[-1])
    a_max = float(np.nanmax(alpha_db_m)) if alpha_db_m.size else 1.0
    p_max = float(np.nanmax(pump_w)) * 1.05 if pump_w.size else 1.0
    return {
        "x": (z0, z1),
        "y_alpha": (0.0, max(a_max * 1.1, 0.1)),
        "y_pump": (0.0, max(p_max, 1e-6)),
    }


def integrate_signal_spectrum(
    signal_w_nm: np.ndarray,
    t_s: np.ndarray,
    wl_nm: np.ndarray,
    spec: ChirpedBurstSpec,
    *,
    packet: bool = True,
) -> np.ndarray:
    """Time-integrated spectrum (J/nm) over packet or single-pulse (1/e²) window."""
    from laser_sim.pulses.chirp import (
        packet_integration_window_s,
        single_pulse_1e2_window_s,
    )

    if packet:
        t_lo, t_hi = packet_integration_window_s(spec)
    else:
        t_lo, t_hi = single_pulse_1e2_window_s(spec, pulse_index=0)
    m = (t_s >= t_lo) & (t_s <= t_hi)
    if int(np.count_nonzero(m)) < 2:
        p_t = integrated_power_vs_time(signal_w_nm, wl_nm)
        peak = float(np.max(p_t)) if p_t.size else 0.0
        if peak > 0.0:
            active = np.where(p_t > peak * 1e-4)[0]
            if active.size >= 2:
                m = np.zeros(t_s.size, dtype=bool)
                m[active[0] : active[-1] + 1] = True
    if int(np.count_nonzero(m)) < 2:
        return np.zeros(wl_nm.size)
    return np.trapezoid(signal_w_nm[m], t_s[m], axis=0)


def apply_plotly_temporal_dual_axis(
    fig,
    t_rel_ns: np.ndarray,
    p_in: np.ndarray,
    p_out: np.ndarray,
    xlim_ns: tuple[float, float],
) -> None:
    in_max = float(np.max(p_in)) if p_in.size else 0.0
    out_max = float(np.max(p_out)) if p_out.size else 0.0
    in_ylim = [0.0, max(in_max * 1.15, 1e-12)] if in_max > 0 else None
    out_ylim = [0.0, max(out_max * 1.15, 1e-12)] if out_max > 0 else None

    yaxis_kw = dict(title="Signal in (W)", side="left", rangemode="tozero")
    yaxis2_kw = dict(
        title="Signal out (W)",
        side="right",
        overlaying="y",
        rangemode="tozero",
    )
    if in_ylim is not None:
        yaxis_kw["range"] = in_ylim
    if out_ylim is not None:
        yaxis2_kw["range"] = out_ylim

    fig.update_layout(
        xaxis=dict(title="Time relative to packet center (ns)", range=list(xlim_ns)),
        yaxis=yaxis_kw,
        yaxis2=yaxis2_kw,
    )


def apply_plotly_spectrum_layout(fig, spec: ChirpedBurstSpec) -> None:
    lo, hi = spectrum_plot_limits(spec)
    ymax = 0.0
    for tr in fig.data:
        y = np.asarray(tr.y)
        if y.size:
            ymax = max(ymax, float(np.max(y)))
    yaxis_kw = dict(title="Energy density (J/nm)", rangemode="tozero")
    if ymax > 0:
        yaxis_kw["range"] = [0.0, ymax * 1.15]
    fig.update_layout(
        xaxis=dict(title="Wavelength (nm)", range=[lo, hi]),
        yaxis=yaxis_kw,
    )


def apply_plotly_time_heatmap_layout(
    fig,
    t_s: np.ndarray,
    wl_nm: np.ndarray,
    spec: ChirpedBurstSpec,
) -> None:
    center = packet_center_time_s(spec)
    half = temporal_half_width_s(spec)
    t_lo, t_hi = center - half, center + half
    lo, hi = spectrum_plot_limits(spec)
    fig.update_layout(
        xaxis=dict(title="λ (nm)", range=[lo, hi]),
        yaxis=dict(title="t − t_center (ns)", range=[-half * 1e9, half * 1e9]),
    )


def dpg_axis_limits_temporal(
    t_rel_ns: np.ndarray,
    p_in: np.ndarray,
    p_out: np.ndarray,
    xlim_ns: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    in_max = float(np.max(p_in)) if p_in.size else 1.0
    out_max = float(np.max(p_out)) if p_out.size else 1.0
    return (
        xlim_ns,
        (0.0, max(in_max * 1.15, 1e-12)),
        (0.0, max(out_max * 1.15, 1e-12)),
    )
