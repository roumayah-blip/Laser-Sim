"""Multi-channel signal and pump definitions for the fiber amplifier."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec


@dataclass
class SignalChannel:
    """One signal channel: a ChirpedBurstSpec on its own narrow λ grid."""

    name: str = "signal"
    spec: ChirpedBurstSpec = field(default_factory=ChirpedBurstSpec)
    n_lambda: int = 32
    lambda_half_window_nm: float = 4.0
    enabled: bool = True


@dataclass
class PumpChannel:
    """One pump channel at a single wavelength."""

    name: str = "pump"
    spec: PumpPulseSpec = field(default_factory=PumpPulseSpec)
    cladding_pumped: bool = True
    backward_fraction: float = 0.0
    enabled: bool = True


def enabled_signal_channels(channels: list[SignalChannel]) -> list[SignalChannel]:
    return [ch for ch in channels if ch.enabled]


def enabled_pump_channels(channels: list[PumpChannel]) -> list[PumpChannel]:
    return [ch for ch in channels if ch.enabled]


def collect_signal_wavelength_grids(
    channels: list[SignalChannel],
) -> tuple[np.ndarray, list[tuple[SignalChannel, slice]]]:
    """
    Build the union λ grid across signal channels (contiguous blocks per channel).

    Returns (wl_union_nm, [(channel, slice), ...]).
    """
    bounds: list[tuple[float, float, SignalChannel]] = []
    for ch in channels:
        if not ch.enabled:
            continue
        lo = ch.spec.center_wavelength_nm - ch.lambda_half_window_nm
        hi = ch.spec.center_wavelength_nm + ch.lambda_half_window_nm
        bounds.append((lo, hi, ch))
    bounds.sort(key=lambda b: b[0])
    for i in range(len(bounds) - 1):
        if bounds[i][1] > bounds[i + 1][0]:
            raise ValueError(
                f"Signal channels '{bounds[i][2].name}' and '{bounds[i + 1][2].name}' "
                f"have overlapping λ ranges. Reduce lambda_half_window_nm or "
                f"separate the center wavelengths."
            )
    parts: list[np.ndarray] = []
    slices: list[tuple[SignalChannel, slice]] = []
    cursor = 0
    for lo, hi, ch in bounds:
        n_pts = max(int(ch.n_lambda), 4)
        wl_ch = np.linspace(lo, hi, n_pts)
        parts.append(wl_ch)
        slices.append((ch, slice(cursor, cursor + n_pts)))
        cursor += n_pts
    if not parts:
        return np.array([1030.0], dtype=np.float64), []
    return np.concatenate(parts).astype(np.float64), slices


def build_multichannel_time_grid(
    *,
    signal_channels: list[SignalChannel],
    pump_channels: list[PumpChannel],
    points_per_chirped_pulse: int = 80,
    points_per_burst_spacing: int = 16,
    points_pump_coarse: int = 400,
) -> np.ndarray:
    """
    Union of per-channel fine windows + coarse pump windows; sorted unique t[].
    """
    from laser_sim.pulses.chirp import build_cpa_time_grid

    times: list[np.ndarray] = [np.array([0.0], dtype=np.float64)]
    n_pump_on = max(1, sum(1 for pc in pump_channels if pc.enabled))
    pump_t_max = 0.0
    for pc in pump_channels:
        if not pc.enabled:
            continue
        t0 = float(pc.spec.start_time_s)
        if pc.spec.cw:
            t1 = t0 + max(pc.spec.duration_s, 1e-6)
        else:
            t1 = t0 + max(pc.spec.duration_s, 1e-9)
        pump_t_max = max(pump_t_max, t1)
        n_coarse = max(points_pump_coarse // n_pump_on, 8)
        times.append(np.linspace(t0, t1, n_coarse))
    for sc in signal_channels:
        if not sc.enabled:
            continue
        if sc.spec.cw_average_power_mode:
            # Flat-in-time signal (build_cw_average_signal) — no packet to
            # resolve finely; a coarse window spanning start..pump_t_max is enough.
            t0 = float(sc.spec.burst_start_time_s)
            t1 = max(pump_t_max, t0 + 1e-6)
            times.append(np.linspace(t0, t1, max(points_pump_coarse // 4, 8)))
            continue
        t_sub = build_cpa_time_grid(
            pump_duration_s=max(pump_t_max, sc.spec.burst_start_time_s + 1e-6),
            spec=sc.spec,
            pump_cw=False,
            points_per_chirped_pulse=points_per_chirped_pulse,
            points_per_burst_spacing=points_per_burst_spacing,
            points_pump_coarse=0,
        )
        times.append(t_sub)
    return np.unique(np.concatenate(times)).astype(np.float64)
