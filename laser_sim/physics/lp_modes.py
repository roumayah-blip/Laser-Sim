"""
LP mode eigenvalues and core overlap (Petermann I) for step-index fiber.

Used at setup time for multimode groundwork; not part of the Taichi GPU path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq
from scipy.special import jv, kv


@dataclass(frozen=True)
class LPMode:
    name: str
    l: int
    m: int
    u: float
    w: float
    n_eff: float
    gamma_overlap: float
    degeneracy: int


def _n_clad(n_core: float, na: float) -> float:
    return float(np.sqrt(max(n_core**2 - na**2, 1.0)))


def _v_number(core_radius_m: float, na: float, wavelength_m: float) -> float:
    k0 = 2.0 * np.pi / wavelength_m
    return k0 * core_radius_m * na


def _characteristic_residual(u: float, l: int, v: float) -> float:
    if u <= 1e-8 or u >= v - 1e-8:
        return 1e6
    w = float(np.sqrt(max(v * v - u * u, 1e-30)))
    ju = float(jv(l, u))
    if abs(ju) < 1e-14:
        return 1e6
    jlp = float(jv(l + 1, u))
    kl = float(kv(l, w))
    if abs(kl) < 1e-14:
        return 1e6
    klp = float(kv(l + 1, w))
    lhs = u * jlp / ju
    rhs = -w * klp / kl
    return lhs - rhs


def gamma_petermann_i(u: float, w: float, l: int, v: float) -> float:
    """
    Core power overlap Γ (Petermann I) for LP_lm on a step-index fiber.

    Γ = 1 − (u/v)² · K_{l−1}(w) K_{l+1}(w) / K_l(w)²
    """
    if v <= 0:
        return 0.0
    kl = float(kv(l, w))
    if abs(kl) < 1e-14:
        return 1.0
    km1 = float(kv(l - 1, w))
    kp1 = float(kv(l + 1, w))
    fac = (u / v) ** 2
    gamma = 1.0 - fac * km1 * kp1 / (kl * kl)
    return float(np.clip(gamma, 0.0, 1.0))


def mode_group_index(
    u: float,
    core_radius_m: float,
    wavelength_m: float,
    n_core: float,
) -> float:
    """Group index n_g = n_eff − λ dn_eff/dλ (finite difference)."""
    dwl = wavelength_m * 1e-4
    wl_p = wavelength_m + dwl
    wl_m = wavelength_m - dwl
    k0p = 2.0 * np.pi / wl_p
    k0m = 2.0 * np.pi / wl_m
    beta_p = np.sqrt(max((k0p * n_core) ** 2 - (u / core_radius_m) ** 2, 0.0))
    beta_m = np.sqrt(max((k0m * n_core) ** 2 - (u / core_radius_m) ** 2, 0.0))
    n_eff_p = beta_p / k0p
    n_eff_m = beta_m / k0m
    n_eff = np.sqrt(max((2.0 * np.pi / wavelength_m * n_core) ** 2 - (u / core_radius_m) ** 2, 0.0)) / (
        2.0 * np.pi / wavelength_m
    )
    return float(n_eff - wavelength_m * (n_eff_p - n_eff_m) / (2.0 * dwl))


def _mode_name(l: int, m: int) -> str:
    return f"LP{l}{m + 1}"


def _find_roots_for_l(l: int, v: float, n_scan: int = 800) -> list[float]:
    """Scan u ∈ (0, V) for sign changes of the LP characteristic equation."""
    if v <= 1e-6:
        return []
    us = np.linspace(1e-6, v - 1e-6, n_scan)
    vals = np.array([_characteristic_residual(float(u), l, v) for u in us])
    roots: list[float] = []
    for i in range(len(us) - 1):
        if vals[i] * vals[i + 1] > 0:
            continue
        if not np.isfinite(vals[i]) or not np.isfinite(vals[i + 1]):
            continue
        try:
            root = brentq(
                lambda uu: _characteristic_residual(uu, l, v),
                float(us[i]),
                float(us[i + 1]),
                xtol=1e-10,
                rtol=1e-10,
            )
        except ValueError:
            continue
        if all(abs(root - r) > 1e-5 for r in roots):
            roots.append(float(root))
    return sorted(roots)


def find_lp_modes(
    core_radius_m: float,
    na: float,
    wavelength_m: float,
    n_core: float = 1.45,
) -> list[LPMode]:
    """
    Find guided LP modes by solving the step-index characteristic equation.

    Returns modes sorted by n_eff descending (fundamental first).
    """
    v = _v_number(core_radius_m, na, wavelength_m)
    if v < 0.5:
        return []

    k0 = 2.0 * np.pi / wavelength_m
    modes: list[LPMode] = []
    l_max = int(np.ceil(v)) + 2

    for l in range(l_max):
        roots = _find_roots_for_l(l, v)
        if not roots and l > 0:
            break
        for m, u in enumerate(roots):
            w = float(np.sqrt(max(v * v - u * u, 1e-30)))
            beta = np.sqrt(max((k0 * n_core) ** 2 - (u / core_radius_m) ** 2, 0.0))
            n_eff = float(beta / k0)
            gamma = gamma_petermann_i(u, w, l, v)
            modes.append(
                LPMode(
                    name=_mode_name(l, m),
                    l=l,
                    m=m,
                    u=u,
                    w=w,
                    n_eff=n_eff,
                    gamma_overlap=gamma,
                    degeneracy=1 if l == 0 else 2,
                )
            )

    modes = [m for m in modes if m.gamma_overlap > 1e-6]
    modes.sort(key=lambda m: m.n_eff, reverse=True)
    return modes
