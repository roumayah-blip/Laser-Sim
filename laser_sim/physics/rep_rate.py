"""Steady-state convergence metrics for rep-rate packet trains."""

from __future__ import annotations

import numpy as np

from laser_sim.pulses.chirp import ChirpedBurstSpec, rep_period_s
from laser_sim.viz.plot_limits import integrated_power_vs_time


def _slice_period(
    t: np.ndarray,
    p_t: np.ndarray,
    t_start: float,
    t_end: float,
) -> tuple[np.ndarray, np.ndarray]:
    m = (t >= t_start) & (t < t_end)
    if np.sum(m) < 3:
        return np.array([]), np.array([])
    return t[m], p_t[m]


def analyze_rep_rate_steady_state(
    t_s: np.ndarray,
    signal_out_w_nm: np.ndarray,
    wavelength_nm: np.ndarray,
    spec: ChirpedBurstSpec,
) -> tuple[bool, float, str]:
    """
    Compare last two packet periods at fiber output.

    Returns (converged, relative_L2_change, notes).
    """
    if not spec.rep_rate_mode:
        return False, float("nan"), "Rep-rate mode off."

    t_rep = rep_period_s(spec)
    p_out = integrated_power_vs_time(signal_out_w_nm, wavelength_nm)

    n_per = spec.n_periods
    t1_start = spec.burst_start_time_s + (n_per - 2) * t_rep
    t1_end = t1_start + t_rep
    t2_start = spec.burst_start_time_s + (n_per - 1) * t_rep
    t2_end = t2_start + t_rep

    _, p1 = _slice_period(t_s, p_out, t1_start, t1_end)
    _, p2 = _slice_period(t_s, p_out, t2_start, t2_end)

    if p1.size < 3 or p2.size < 3:
        return False, float("nan"), "Insufficient samples in last periods; increase n_periods or grid."

    e1 = float(np.trapezoid(p1))
    e2 = float(np.trapezoid(p2))
    if e1 < 1e-30 and e2 < 1e-30:
        return True, 0.0, "Negligible output energy both periods."

    # Resample onto common normalized time grid within period
    n = min(p1.size, p2.size, 200)
    u = np.linspace(0, 1, n)
    p1r = np.interp(u, np.linspace(0, 1, p1.size), p1 / max(np.max(p1), 1e-30))
    p2r = np.interp(u, np.linspace(0, 1, p2.size), p2 / max(np.max(p2), 1e-30))
    rel_shape = float(np.sqrt(np.mean((p1r - p2r) ** 2)))
    rel_energy = abs(e2 - e1) / max(e1, e2)
    metric = max(rel_shape, rel_energy)
    converged = metric <= spec.steady_state_tol

    notes = (
        f"Rep rate {spec.rep_rate_hz/1e3:.1f} kHz, period {t_rep*1e6:.2f} µs, "
        f"{n_per} periods. Shape Δ={rel_shape:.4f}, energy Δ={rel_energy:.4f} "
        f"(tol={spec.steady_state_tol}). "
        + ("Steady state reached." if converged else "Not converged — increase n_periods.")
    )
    return converged, metric, notes
