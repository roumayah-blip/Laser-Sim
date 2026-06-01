"""2D thermal diffusion and thermal lens focal length."""

from __future__ import annotations

import numpy as np


def solve_heat_2d(
    Q_xyz: np.ndarray,
    dt_s: float,
    kappa_t: float,
    rho: float,
    cp: float,
    T_ambient_k: float = 300.0,
    bc: str = "dirichlet",
    n_steps: int | None = None,
) -> np.ndarray:
    """
    Explicit 2D heat equation on (nx, ny) at each z slab.

    Q_xyz: W/m³, shape (nz, nx, ny). Integrated source over pump duration.
    Returns ΔT above ambient, shape (nz, nx, ny).
    """
    nz, nx, ny = Q_xyz.shape
    alpha = kappa_t / (rho * cp)
    dx = dy = 1.0  # normalized; Q already includes physical scaling in caller

    if dt_s <= 0 or alpha <= 0:
        return Q_xyz * dt_s / (rho * cp)

    # Quasi-steady for short pulses: one step if stability allows
    dt_stable = 0.25 * dx * dx / max(alpha, 1e-30)
    n_steps = n_steps or max(1, int(np.ceil(dt_s / dt_stable)))
    dt_sub = dt_s / n_steps

    T = np.zeros((nz, nx, ny), dtype=np.float64)
    lap = np.zeros((nx, ny), dtype=np.float64)
    for iz in range(nz):
        Tz = T[iz].copy()
        for _ in range(n_steps):
            lap[1:-1, 1:-1] = (
                Tz[2:, 1:-1]
                + Tz[:-2, 1:-1]
                + Tz[1:-1, 2:]
                + Tz[1:-1, :-2]
                - 4.0 * Tz[1:-1, 1:-1]
            ) / (dx * dx)
            Tz += dt_sub * (alpha * lap + Q_xyz[iz] / (rho * cp))
            if bc == "dirichlet":
                Tz[0, :] = Tz[-1, :] = Tz[:, 0] = Tz[:, -1] = 0.0
        T[iz] = Tz
    return T


def dn_thermal_field(dndt: float, T_rise_xyz: np.ndarray) -> np.ndarray:
    return dndt * T_rise_xyz


def thermal_lens_focal_length(
    dn_dT: float,
    T_profile_xy: np.ndarray,
    dx: float,
    dy: float,
    wavelength_m: float,
    crystal_length_m: float,
) -> float:
    """
    Paraxial thermal lens from quadratic fit to ΔT(r²).

    f = 1 / (k0 * d²(Δn)/dr²) with Δn = dn/dT * ΔT.
    """
    nx, ny = T_profile_xy.shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    r2 = xx**2 + yy**2
    T_flat = T_profile_xy.ravel()
    r2_flat = r2.ravel()
    if np.max(T_flat) < 1e-12:
        return float("inf")
    # Fit ΔT ≈ a + b*r²
    A = np.column_stack([np.ones_like(r2_flat), r2_flat])
    coeff, _, _, _ = np.linalg.lstsq(A, T_flat, rcond=None)
    T0 = float(coeff[0])
    b = float(coeff[1])
    if T0 <= 0 or abs(b) < 1e-30:
        return float("inf")
    # Peaked profile: b < 0, w_eff from ΔT(r) ≈ T0(1 − 2r²/w²) → f ≈ w²/(2L·(dn/dT)·T0)
    w_eff = float(np.sqrt(max(-T0 / (2.0 * b), 1e-30))) if b < 0 else float(np.sqrt(nx * dx * ny * dy))
    return w_eff**2 / (2.0 * max(crystal_length_m, 1e-9) * max(dn_dT, 1e-30) * T0)
