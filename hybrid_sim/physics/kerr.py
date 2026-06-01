"""Kerr nonlinear phase and B-integral for 2D fields."""

from __future__ import annotations

import numpy as np

PI_HALF = 0.5 * np.pi
PI = np.pi


def kerr_phase_2d(
    intensity_xy: np.ndarray,
    n2_m2_per_w: float,
    k0: float,
    dz: float,
) -> np.ndarray:
    """ΔΦ_Kerr(x,y) = k0 * n2 * I(x,y) * dz [rad]."""
    return k0 * n2_m2_per_w * np.maximum(intensity_xy, 0.0) * dz


def b_integral_2d(
    intensity_xyz: np.ndarray,
    n2: float,
    k0: float,
    dz: float,
) -> np.ndarray:
    """B(x,y) = k0 * n2 * ∫ I(x,y,z) dz."""
    return k0 * n2 * np.sum(intensity_xyz, axis=0) * dz


def check_b_integral(B_max: float) -> str:
    if B_max > PI:
        return f"CRITICAL: B_max={B_max:.3f} rad > π — self-focusing likely invalidates simulation"
    if B_max > PI_HALF:
        return f"WARNING: B_max={B_max:.3f} rad > π/2 — significant nonlinear phase"
    return f"OK: B_max={B_max:.3f} rad"
