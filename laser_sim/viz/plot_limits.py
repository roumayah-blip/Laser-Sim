"""Default axis ranges and plot helpers for CPA pulse visualization."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from laser_sim.pulses.chirp import ChirpedBurstSpec


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
    if not np.any(m):
        m = np.ones_like(t_s, dtype=bool)
    t_rel = (t_s[m] - center) * 1e9
    xlim = temporal_relative_window_ns(spec, time_axis_factor)
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


def integrated_power_vs_time(signal_w_nm: np.ndarray, wl_nm: np.ndarray) -> np.ndarray:
    dlam = np.gradient(wl_nm)
    return np.sum(signal_w_nm * dlam[None, :], axis=1)


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
        return np.zeros(wl_nm.size)
    return np.trapezoid(signal_w_nm[m], t_s[m], axis=0)


def apply_plotly_temporal_dual_axis(
    fig,
    t_rel_ns: np.ndarray,
    p_in: np.ndarray,
    p_out: np.ndarray,
    xlim_ns: tuple[float, float],
) -> None:
    in_max = float(np.max(p_in)) if p_in.size else 1.0
    out_max = float(np.max(p_out)) if p_out.size else 1.0

    fig.update_layout(
        xaxis=dict(title="Time relative to packet center (ns)", range=list(xlim_ns)),
        yaxis=dict(
            title="Signal in (W)",
            side="left",
            rangemode="tozero",
            range=[0.0, max(in_max * 1.15, 1e-12)],
        ),
        yaxis2=dict(
            title="Signal out (W)",
            side="right",
            overlaying="y",
            rangemode="tozero",
            range=[0.0, max(out_max * 1.15, 1e-12)],
        ),
    )


def apply_plotly_spectrum_layout(fig, spec: ChirpedBurstSpec) -> None:
    lo, hi = spectrum_plot_limits(spec)
    ymax = 0.0
    for tr in fig.data:
        y = np.asarray(tr.y)
        if y.size:
            ymax = max(ymax, float(np.max(y)))
    fig.update_layout(
        xaxis=dict(title="Wavelength (nm)", range=[lo, hi]),
        yaxis=dict(title="Energy density (J/nm)", rangemode="tozero", range=[0, ymax * 1.15 + 1e-30]),
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
