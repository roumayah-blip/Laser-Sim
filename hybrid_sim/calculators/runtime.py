"""Runtime estimates for hybrid 2D grid."""

from __future__ import annotations


def estimate_solid_runtime_s(
    n_z: int,
    n_t: int,
    n_x: int,
    n_y: int,
    *,
    backend: str = "numpy",
) -> float:
    """Rough CPU seconds for solid amplifier pass."""
    ops = n_z * n_t * n_x * n_y
    per_op = 2e-8 if backend == "numpy" else 5e-9
    return ops * per_op


def recommend_transverse_grid(beam_waist_m: float, points_per_waist: int = 8) -> int:
    aperture = 4.0 * beam_waist_m
    return max(32, int(points_per_waist * aperture / beam_waist_m))
