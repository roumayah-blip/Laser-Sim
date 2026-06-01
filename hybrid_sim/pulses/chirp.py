"""
Chirped CPA pulses (ns+) and pump envelopes (µs–ms).

fs/ps hooks: set `ultrashort_mode=True` and small `duration_s` — same API, finer grids.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Pulse-packet spacing (seconds): default 1 ns, minimum 0.5 ns
MIN_BURST_SPACING_S = 0.5e-9
DEFAULT_BURST_SPACING_S = 1.0e-9


@dataclass
class ChirpedBurstSpec:
    """Spectrally resolved chirped signal burst (CPA stretch regime)."""

    center_wavelength_nm: float = 1030.0
    bandwidth_nm: float = 8.0
    chirp_duration_s: float = 1e-9  # stretched pulse ~1 ns+
    packet_energy_j: float = 10e-6
    burst_count: int = 5  # pulses per packet (typical CPA train)
    burst_spacing_s: float = DEFAULT_BURST_SPACING_S
    burst_start_time_s: float = 0.0  # delay first burst after pump turn-on
    spectral_shape: str = "gaussian"  # gaussian | flat
    # Hook: quadratic chirp coefficient (s²/nm) mapping λ → group delay
    chirp_parameter_s2_per_nm: float | None = None
    ultrashort_mode: bool = False  # hook for future fs/ps grids
    # Rep-rate / steady-state (CW pump): repeat packet at 1/rep_rate_hz
    rep_rate_mode: bool = False
    rep_rate_hz: float = 100e3
    n_periods: int = 40
    steady_state_tol: float = 0.02
    cw_average_power_mode: bool = False  # flat CW at packet-average power (diagnostics)
    pulse_relative_powers: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        self.burst_spacing_s = max(float(self.burst_spacing_s), MIN_BURST_SPACING_S)
        self.burst_count = int(np.clip(self.burst_count, 1, 50))
        self.n_periods = int(np.clip(self.n_periods, 3, 500))
        self.rep_rate_hz = max(float(self.rep_rate_hz), 1.0)
        if self.pulse_relative_powers is not None:
            arr = tuple(float(x) for x in self.pulse_relative_powers)
            if len(arr) != self.burst_count:
                raise ValueError(
                    f"pulse_relative_powers length {len(arr)} != burst_count {self.burst_count}"
                )
            if any(x < 0 for x in arr):
                raise ValueError("pulse_relative_powers must be non-negative")
            self.pulse_relative_powers = arr

    @property
    def energy_per_pulse_j(self) -> float:
        """Backwards-compat: energy of a single flat pulse = packet_energy_j / burst_count."""
        return self.packet_energy_j / max(self.burst_count, 1)


def packet_time_extent_s(spec: ChirpedBurstSpec) -> float:
    """End time of last pulse envelope: t_start + (N-1)*Δt + T_chirp."""
    n = max(spec.burst_count, 1)
    return spec.burst_start_time_s + (n - 1) * spec.burst_spacing_s + spec.chirp_duration_s


def packet_duration_s(spec: ChirpedBurstSpec) -> float:
    """Duration of one packet (first pulse start → last pulse end)."""
    return packet_time_extent_s(spec) - spec.burst_start_time_s


def packet_energy_expected_j(spec: ChirpedBurstSpec) -> float:
    """Total energy in one packet (J)."""
    return float(spec.packet_energy_j)


def first_pulse_center_time_s(spec: ChirpedBurstSpec) -> float:
    """Time (s) at the peak of the first pulse in the packet."""
    return spec.burst_start_time_s + 0.5 * spec.chirp_duration_s


def last_pulse_center_time_s(spec: ChirpedBurstSpec) -> float:
    """Time (s) at the peak of the last pulse in the packet."""
    n = max(spec.burst_count, 1)
    t0 = spec.burst_start_time_s + (n - 1) * spec.burst_spacing_s
    return t0 + 0.5 * spec.chirp_duration_s


def chirp_sigma_t_s(spec: ChirpedBurstSpec) -> float:
    """Temporal σ for intensity Gaussian (FWHM = chirp_duration_s)."""
    return spec.chirp_duration_s / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def _get_pulse_weights(spec: ChirpedBurstSpec) -> np.ndarray:
    """
    Return per-pulse energy weights, normalized so mean = 1.0.
    Preserves total packet energy = packet_energy_j (mean weight 1 → sum(weights)=burst_count).
    """
    n = int(np.clip(spec.burst_count, 1, 50))
    if spec.pulse_relative_powers is None:
        return np.ones(n, dtype=np.float64)
    arr = np.asarray(spec.pulse_relative_powers, dtype=np.float64)
    mean = float(np.mean(arr))
    if mean < 1e-12:
        raise ValueError("pulse_relative_powers mean is zero — all weights are zero")
    return arr / mean


def pulse_center_time_s(spec: ChirpedBurstSpec, pulse_index: int = 0) -> float:
    """Peak time (s) of pulse ``pulse_index`` in the packet."""
    spacing = max(spec.burst_spacing_s, MIN_BURST_SPACING_S)
    idx = int(np.clip(pulse_index, 0, max(spec.burst_count, 1) - 1))
    return spec.burst_start_time_s + idx * spacing + 0.5 * spec.chirp_duration_s


def single_pulse_1e2_window_s(
    spec: ChirpedBurstSpec,
    pulse_index: int = 0,
) -> tuple[float, float]:
    """
    Time window (s) for one pulse: intensity ≥ P_peak/e².

    For P(t) ∝ exp(−½((t−t₀)/σ)²), the 1/e² intensity radius is 2σ.
    """
    sigma = chirp_sigma_t_s(spec)
    t_c = pulse_center_time_s(spec, pulse_index)
    half = 2.0 * sigma
    return t_c - half, t_c + half


def packet_integration_window_s(
    spec: ChirpedBurstSpec,
    margin_chirps: float = 0.5,
) -> tuple[float, float]:
    """Time window (s) covering the full packet (first pulse start → last pulse end)."""
    pulse_w = max(spec.chirp_duration_s, 0.5e-9)
    margin = margin_chirps * pulse_w
    return spec.burst_start_time_s - margin, packet_time_extent_s(spec) + margin


def nearest_time_index(t_s: np.ndarray, t_query_s: float) -> int:
    return int(np.argmin(np.abs(t_s - t_query_s)))


def _integrate_over_time_window(
    signal_w_nm: np.ndarray,
    t_s: np.ndarray,
    wl_nm: np.ndarray,
    t_lo: float,
    t_hi: float,
) -> float:
    """∫∫ P(t,λ) dλ dt (J) via trapezoid rule on [t_lo, t_hi] only."""
    t = np.asarray(t_s, dtype=np.float64)
    wl = np.asarray(wl_nm, dtype=np.float64)
    p = np.asarray(signal_w_nm, dtype=np.float64)
    if t.size < 2 or wl.size < 1 or p.size == 0:
        return 0.0

    dlam = np.gradient(wl)
    p_t = np.sum(p * dlam[None, :], axis=1)
    m = (t >= t_lo) & (t <= t_hi)
    if int(np.count_nonzero(m)) < 2:
        return 0.0
    return float(np.trapezoid(p_t[m], t[m]))


def integrate_single_pulse_energy(
    signal_w_nm: np.ndarray,
    t_s: np.ndarray,
    wl_nm: np.ndarray,
    spec: ChirpedBurstSpec,
    pulse_index: int = 0,
) -> float:
    """Energy (J) in one pulse, integrated to the 1/e² intensity contour."""
    t_lo, t_hi = single_pulse_1e2_window_s(spec, pulse_index)
    return _integrate_over_time_window(signal_w_nm, t_s, wl_nm, t_lo, t_hi)


def integrate_packet_energy(
    signal_w_nm: np.ndarray,
    t_s: np.ndarray,
    wl_nm: np.ndarray,
    spec: ChirpedBurstSpec,
) -> float:
    """Total packet energy (J) over all pulses in the train."""
    t_lo, t_hi = packet_integration_window_s(spec)
    return _integrate_over_time_window(signal_w_nm, t_s, wl_nm, t_lo, t_hi)


def integrate_signal_energy(
    signal_w_nm: np.ndarray,
    t_s: np.ndarray,
    wl_nm: np.ndarray,
    spec: ChirpedBurstSpec | None = None,
) -> float:
    """Alias: packet energy when ``spec`` is given; else threshold mask on full grid."""
    if spec is None:
        t = np.asarray(t_s, dtype=np.float64)
        wl = np.asarray(wl_nm, dtype=np.float64)
        p = np.asarray(signal_w_nm, dtype=np.float64)
        dlam = np.gradient(wl)
        p_t = np.sum(p * dlam[None, :], axis=1)
        thresh = 1e-6 * max(float(np.max(p_t)), 1e-30)
        m = p_t > thresh
        if int(np.count_nonzero(m)) < 2:
            return 0.0
        return float(np.trapezoid(p_t[m], t[m]))
    return integrate_packet_energy(signal_w_nm, t_s, wl_nm, spec)


def rep_period_s(spec: ChirpedBurstSpec) -> float:
    return 1.0 / spec.rep_rate_hz


@dataclass
class PumpPulseSpec:
    """Pump temporal envelope at a single wavelength (resolved separately)."""

    wavelength_nm: float = 976.0
    peak_power_w: float = 50.0
    duration_s: float = 1e-3
    shape: str = "flat_top"  # flat_top | gaussian | trapezoid | cw
    start_time_s: float = 0.0
    edge_time_s: float = 50e-6  # rise/fall for trapezoid
    cw: bool = False  # steady pump power over the simulation window

    def __post_init__(self) -> None:
        if self.cw or self.shape.lower() == "cw":
            self.cw = True
            self.shape = "cw"


def _spectral_weights(wl_nm: np.ndarray, spec: ChirpedBurstSpec) -> np.ndarray:
    c = spec.center_wavelength_nm
    bw = max(spec.bandwidth_nm, 0.1)
    if spec.spectral_shape == "flat":
        w = ((wl_nm >= c - bw / 2) & (wl_nm <= c + bw / 2)).astype(float)
    else:
        sigma = bw / (2 * np.sqrt(2 * np.log(2)))
        w = np.exp(-0.5 * ((wl_nm - c) / sigma) ** 2)
    w /= np.maximum(w.sum(), 1e-30)
    return w


def _delay_from_wavelength(wl_nm: np.ndarray, spec: ChirpedBurstSpec) -> np.ndarray:
    """Map wavelength to arrival time within the chirped pulse (linear chirp)."""
    c = spec.center_wavelength_nm
    T = spec.chirp_duration_s
    if spec.chirp_parameter_s2_per_nm is not None:
        # t(λ) = t0 + C * (λ - λ0)²
        return spec.chirp_parameter_s2_per_nm * (wl_nm - c) ** 2
    # Default: spread pulse across T using linear λ→t mapping over bandwidth
    bw = max(spec.bandwidth_nm, 0.1)
    return (wl_nm - c) / bw * T


def build_chirped_signal(
    time_s: np.ndarray,
    wavelength_nm: np.ndarray,
    spec: ChirpedBurstSpec,
) -> np.ndarray:
    """Build input signal; adds periodic packets when ``rep_rate_mode`` is enabled."""
    if spec.cw_average_power_mode:
        from hybrid_sim.pulses.cw_average import build_cw_average_signal

        return build_cw_average_signal(time_s, wavelength_nm, spec)
    if not spec.rep_rate_mode:
        return build_chirped_burst(time_s, wavelength_nm, spec)

    t_period = rep_period_s(spec)
    pkt_dur = packet_duration_s(spec)
    if t_period < pkt_dur * 1.01:
        raise ValueError(
            f"Rep period {t_period*1e6:.3f} µs shorter than packet duration {pkt_dur*1e6:.3f} µs. "
            "Lower rep_rate_hz or shorten the packet."
        )

    out = np.zeros((time_s.size, wavelength_nm.size), dtype=np.float64)
    t_max = float(time_s[-1])
    k = 0
    while True:
        t0 = spec.burst_start_time_s + k * t_period
        if t0 > t_max:
            break
        shifted = ChirpedBurstSpec(
            center_wavelength_nm=spec.center_wavelength_nm,
            bandwidth_nm=spec.bandwidth_nm,
            chirp_duration_s=spec.chirp_duration_s,
            packet_energy_j=spec.packet_energy_j,
            pulse_relative_powers=spec.pulse_relative_powers,
            burst_count=spec.burst_count,
            burst_spacing_s=spec.burst_spacing_s,
            burst_start_time_s=t0,
            spectral_shape=spec.spectral_shape,
            chirp_parameter_s2_per_nm=spec.chirp_parameter_s2_per_nm,
        )
        out += build_chirped_burst(time_s, wavelength_nm, shifted)
        k += 1
    return np.maximum(out, 0.0)


def build_chirped_burst(
    time_s: np.ndarray,
    wavelength_nm: np.ndarray,
    spec: ChirpedBurstSpec,
) -> np.ndarray:
    """
    Build signal power spectral density P_s(t, λ) in W/nm.

    Each chirped pulse uses ``packet_energy_j * weight[b] / burst_count`` with weights from
    ``_get_pulse_weights`` (mean 1 → total packet energy preserved). Pulses are
    superposed in time: overlapping envelopes **add in intensity** (incoherent
    sum of power density), not in field amplitude.
    """
    nt = time_s.size
    nw = wavelength_nm.size
    out = np.zeros((nt, nw), dtype=np.float64)
    spectral_w = _spectral_weights(wavelength_nm, spec)
    delays = _delay_from_wavelength(wavelength_nm, spec)

    dlam = np.gradient(wavelength_nm)
    dlam = np.where(np.abs(dlam) < 1e-12, 1e-12, dlam)

    spacing = max(spec.burst_spacing_s, MIN_BURST_SPACING_S)
    burst_count = int(np.clip(spec.burst_count, 1, 50))
    sigma_t = spec.chirp_duration_s / (2 * np.sqrt(2 * np.log(2)))

    weights = _get_pulse_weights(spec)
    # Gaussian in time: ∫ P(t) dt = E_pulse with σ_t (FWHM = chirp_duration_s).
    for b in range(burst_count):
        t_burst0 = spec.burst_start_time_s + b * spacing
        e_pulse_b = spec.packet_energy_j * weights[b] / max(spec.burst_count, 1)
        for j in range(nw):
            t_center = t_burst0 + delays[j]
            envelope = np.exp(-0.5 * ((time_s - t_center) / sigma_t) ** 2)
            p_total = e_pulse_b / (np.sqrt(2 * np.pi) * sigma_t) * envelope
            # W/nm: sum_j P_j Δλ_j ≈ P_total when spectral_w sums to 1
            out[:, j] += p_total * spectral_w[j] / dlam[j]

    return np.maximum(out, 0.0)


def build_pump_power(time_s: np.ndarray, spec: PumpPulseSpec) -> np.ndarray:
    """Pump power P_p(t) in watts (single wavelength channel)."""
    t = time_s - spec.start_time_s
    dur = max(spec.duration_s, 1e-12)
    p = np.zeros_like(time_s, dtype=np.float64)

    if spec.cw or spec.shape == "cw":
        return np.where(t >= 0, spec.peak_power_w, 0.0).astype(np.float64)

    if spec.shape == "gaussian":
        sigma = dur / (2 * np.sqrt(2 * np.log(2)))
        p = spec.peak_power_w * np.exp(-0.5 * ((t - 0.5 * dur) / sigma) ** 2)
    elif spec.shape == "trapezoid":
        e = max(spec.edge_time_s, 1e-12)
        rise = np.clip(t / e, 0, 1)
        fall = np.clip((dur - t) / e, 0, 1)
        p = spec.peak_power_w * np.minimum(rise, fall) * (t >= 0) * (t <= dur)
    else:  # flat_top
        p = np.where((t >= 0) & (t <= dur), spec.peak_power_w, 0.0)

    return np.maximum(p, 0.0)


def build_cpa_time_grid(
    *,
    pump_duration_s: float,
    spec: ChirpedBurstSpec,
    pump_cw: bool = False,
    points_per_chirped_pulse: int = 80,
    points_per_burst_spacing: int = 16,
    points_pump_coarse: int = 400,
    margin_chirps: float = 2.0,
) -> np.ndarray:
    if spec.rep_rate_mode:
        return build_rep_rate_time_grid(
            spec=spec,
            pump_cw=pump_cw,
            pump_duration_s=pump_duration_s,
            points_per_chirped_pulse=points_per_chirped_pulse,
            points_per_burst_spacing=points_per_burst_spacing,
            points_pump_coarse=points_pump_coarse,
        )
    """
    Time samples for ms pump + ns pulse packet: fine grid around the packet,
    coarser elsewhere (sorted unique).
    """
    chirp = max(spec.chirp_duration_s, 1e-12)
    spacing = max(spec.burst_spacing_s, MIN_BURST_SPACING_S)
    span = max((spec.burst_count - 1) * spacing, 0.0)
    packet_end = packet_time_extent_s(spec)
    margin = margin_chirps * chirp

    t_pkt0 = max(0.0, spec.burst_start_time_s - margin)
    t_pkt1 = packet_end + margin
    dt_fine = min(
        chirp / max(points_per_chirped_pulse, 20),
        spacing / max(points_per_burst_spacing, 4),
        0.1e-9,
    )
    n_fine = max(int(np.ceil((t_pkt1 - t_pkt0) / dt_fine)) + 1, 32)
    t_fine = np.linspace(t_pkt0, t_pkt1, n_fine)

    if pump_cw:
        # pump_duration_s = simulation window for CW steady state
        t_end = max(pump_duration_s, packet_end) * 1.05
    else:
        t_end = max(pump_duration_s, packet_end) * 1.12 + 0.05 * pump_duration_s
    t_coarse = np.linspace(0.0, t_end, max(points_pump_coarse, 100))

    t = np.unique(np.concatenate([t_coarse, t_fine]))
    return t.astype(np.float64)


def build_rep_rate_time_grid(
    *,
    spec: ChirpedBurstSpec,
    pump_cw: bool,
    pump_duration_s: float,
    points_per_chirped_pulse: int = 80,
    points_per_burst_spacing: int = 16,
    points_pump_coarse: int = 400,
) -> np.ndarray:
    """Time grid covering many packet periods for CW + rep-rate steady-state."""
    t_rep = rep_period_s(spec)
    n_per = spec.n_periods
    t_end = spec.burst_start_time_s + n_per * t_rep + packet_duration_s(spec)
    if pump_cw:
        t_end = max(t_end, pump_duration_s, spec.burst_start_time_s + 3 * t_rep)
    else:
        t_end = max(t_end, pump_duration_s)

    # Fine grids in each period (sample at least one period finely)
    t_fine_list = []
    for k in range(min(n_per, 6)):
        sub = ChirpedBurstSpec(
            center_wavelength_nm=spec.center_wavelength_nm,
            bandwidth_nm=spec.bandwidth_nm,
            chirp_duration_s=spec.chirp_duration_s,
            packet_energy_j=spec.packet_energy_j,
            pulse_relative_powers=spec.pulse_relative_powers,
            burst_count=spec.burst_count,
            burst_spacing_s=spec.burst_spacing_s,
            burst_start_time_s=spec.burst_start_time_s + k * t_rep,
            spectral_shape=spec.spectral_shape,
        )
        tf = build_cpa_time_grid(
            pump_duration_s=t_end,
            spec=sub,
            pump_cw=True,
            points_per_chirped_pulse=points_per_chirped_pulse,
            points_per_burst_spacing=points_per_burst_spacing,
            points_pump_coarse=max(points_pump_coarse // 4, 80),
        )
        t_fine_list.append(tf)

    t_coarse = np.linspace(0.0, t_end, max(points_pump_coarse, 150))
    parts = [t_coarse, *t_fine_list]
    return np.unique(np.concatenate(parts)).astype(np.float64)
