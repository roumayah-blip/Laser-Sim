"""
Estimate wall-clock time for a CPA fiber simulation from grid resolution.

Calibrated with conservative throughput assumptions; user can set GPU model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Effective scalar updates per z-slab per (t, λ) channel
OPS_PER_CELL_PER_Z = 48  # rate + signal + ASE + spont coupling


@dataclass(frozen=True)
class RuntimeEstimate:
    """Simulation cost breakdown."""

    n_z: int
    n_t: int
    n_lambda: int
    n_burst_pulses: int
    total_cell_updates: int
    estimated_seconds: float
    estimated_minutes: float
    backend: str
    breakdown: dict[str, float]
    notes: str


# Conservative effective throughputs (scalar updates / second)
BACKEND_GFLOPS = {
    "cpu": 0.15e9,
    "cuda": 25.0e9,  # RTX 5090 class — tune after first `ti diagnose` benchmark
    "taichi_cuda": 30.0e9,
}


def recommend_time_grid(
    *,
    pump_duration_s: float,
    burst_span_s: float,
    chirped_pulse_duration_s: float,
    burst_count: int,
    burst_spacing_s: float | None = None,
    burst_start_time_s: float = 0.0,
    points_per_chirped_pulse: int = 80,
    points_per_burst_spacing: int = 8,
) -> tuple[float, float, int]:
    """
    Suggest (t_start, t_end, n_t) for CPA + pulse packet + ms pump.

    Uses a hybrid fine grid around the packet (ns spacing) and coarse pump grid.
    """
    from laser_sim.pulses.chirp import ChirpedBurstSpec, build_cpa_time_grid

    spacing = burst_spacing_s
    if spacing is None:
        spacing = burst_span_s / max(burst_count - 1, 1) if burst_count > 1 else 1e-9

    spec = ChirpedBurstSpec(
        chirp_duration_s=chirped_pulse_duration_s,
        burst_count=burst_count,
        burst_spacing_s=spacing,
        burst_start_time_s=burst_start_time_s,
    )
    t = build_cpa_time_grid(
        pump_duration_s=pump_duration_s,
        spec=spec,
        points_per_chirped_pulse=points_per_chirped_pulse,
        points_per_burst_spacing=points_per_burst_spacing,
    )
    return float(t[0]), float(t[-1]), int(t.size)


def recommend_wavelength_grid(
    center_nm: float,
    bandwidth_nm: float,
    points_per_nm: float = 2.0,
    lambda_min_nm: float | None = None,
    lambda_max_nm: float | None = None,
) -> tuple[float, float, int]:
    """Suggest spectral grid for chirped CPA signal."""
    half = 0.5 * bandwidth_nm * 1.2
    lam_min = (center_nm - half) if lambda_min_nm is None else lambda_min_nm
    lam_max = (center_nm + half) if lambda_max_nm is None else lambda_max_nm
    n_lam = int(np.ceil((lam_max - lam_min) * points_per_nm)) + 1
    return lam_min, lam_max, max(n_lam, 32)


def estimate_runtime(
    *,
    fiber_length_m: float,
    n_z: int | None = None,
    dz_m: float | None = None,
    n_t: int,
    n_lambda: int,
    n_burst_pulses: int = 1,
    include_ase: bool = True,
    include_backward_pump: bool = False,
    backend: str = "cuda",
    calibration_factor: float = 1.0,
    pump_duration_s: float = 1e-3,
    signal_chirp_duration_s: float = 1e-9,
) -> RuntimeEstimate:
    """
    Estimate completion time from discretization.

    If n_z is None, use dz_m (default 2 mm for ns CPA advective stability margin).
    """
    if n_z is None:
        dz = dz_m if dz_m is not None else 2e-3
        n_z = max(int(np.ceil(fiber_length_m / dz)) + 1, 10)
    else:
        n_z = max(n_z, 10)

    n_burst = max(1, min(n_burst_pulses, 50))

    channels = 1  # signal spectral
    channels += 1  # pump (broadband or single λ counted in n_lambda if spectrally resolved)
    if include_ase:
        channels += 2  # forward + backward ASE
    if include_backward_pump:
        channels += 1

    # Population + spont + burst replay factor
    burst_factor = 1.0 + 0.05 * (n_burst - 1)
    spont_factor = 1.25 if include_ase else 1.0

    total_updates = (
        n_z * n_t * n_lambda * channels * OPS_PER_CELL_PER_Z * burst_factor * spont_factor
    )

    gflops = BACKEND_GFLOPS.get(backend, BACKEND_GFLOPS["cpu"])
    seconds = (total_updates / gflops) * calibration_factor

    # CPA ns pulses in ms pump window: advective CFL may force sub-stepping
    c_light = 299_792_458.0
    n_group = 1.45
    v_g = c_light / n_group
    dt_est = (fiber_length_m / n_z) / v_g * (signal_chirp_duration_s / max(pump_duration_s, 1e-9))
    cfl_penalty = 1.0
    if signal_chirp_duration_s >= 1e-9 and n_t > 500:
        cfl_penalty = 1.0 + 0.3 * np.log10(max(n_t / 500, 1.0))
    seconds *= cfl_penalty

    breakdown = {
        "base_compute_s": total_updates / gflops,
        "cfl_penalty_factor": cfl_penalty,
        "calibration_factor": calibration_factor,
        "effective_gflops": gflops,
    }

    notes = (
        f"Grid {n_z}×{n_t}×{n_lambda}, {n_burst} burst slot(s), backend={backend}. "
        "Run one short job and set calibration_factor from measured wall time. "
        "fs/ps hooks add ~×2–5 if enabled later."
    )

    return RuntimeEstimate(
        n_z=n_z,
        n_t=n_t,
        n_lambda=n_lambda,
        n_burst_pulses=n_burst,
        total_cell_updates=total_updates,
        estimated_seconds=seconds,
        estimated_minutes=seconds / 60.0,
        backend=backend,
        breakdown=breakdown,
        notes=notes,
    )
