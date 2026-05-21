"""
Signal mode area and guided spontaneous-emission fraction from core NA.
"""

from __future__ import annotations

import numpy as np

from laser_sim.constants import NM_TO_M


def v_number(core_radius_m: float, na: float, wavelength_m: float) -> float:
    """Normalized frequency V = (2π/λ) a_core NA."""
    return float((2.0 * np.pi / wavelength_m) * core_radius_m * na)


def signal_mode_radius_m(
    core_radius_m: float,
    na: float,
    wavelength_m: float,
) -> float:
    """
    LP01 mode-field radius for step-index fiber (Marcuse approximation).

    Uses Marcuse (1978) formula valid for 1.2 < V < 3.
    For V >= 3 (multimode), returns core_radius_m (mode fills core, Gamma_s -> 1).
    For V < 1.2 (weakly guided), clamps V to 1.2.
    """
    V = (2.0 * np.pi / wavelength_m) * core_radius_m * na
    if V >= 3.0:
        return core_radius_m
    V = max(V, 1.2)
    return core_radius_m * (0.65 + 1.619 / V**1.5 + 2.879 / V**6)


def signal_mode_area_m2(
    core_radius_m: float,
    na: float,
    wavelength_m: float,
) -> float:
    """Effective signal mode area π w² (m²)."""
    w = signal_mode_radius_m(core_radius_m, na, wavelength_m)
    return float(np.pi * w * w)


def signal_overlap_gamma(
    core_radius_m: float,
    wavelength_nm: float,
    na: float,
) -> tuple[float, float, float]:
    """
    Return (Gamma_s, A_signal_m2, A_core_m2).

    Gamma_s = min(1, A_mode / A_core).
    """
    wl_m = wavelength_nm * NM_TO_M
    a_core = float(np.pi * core_radius_m * core_radius_m)
    a_mode = signal_mode_area_m2(core_radius_m, na, wl_m)
    gamma_s = min(1.0, a_mode / max(a_core, 1e-30))
    return gamma_s, a_mode, a_core


def guided_spontaneous_fraction(v: float) -> float:
    """
    Fraction of fluorescence coupled into guided LP₀₁ (power in core).

    η ≈ V² / (V² + 2)  (Marcuse-style step-index confinement).
    Clamped to [0, 1].
    """
    v2 = float(v) ** 2
    return float(np.clip(v2 / (v2 + 2.0), 0.0, 1.0))
