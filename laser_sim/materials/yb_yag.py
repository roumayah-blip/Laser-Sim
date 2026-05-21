"""Yb:YAG — narrow lines (300 K, π polarization approximate)."""

import numpy as np

from laser_sim.materials.base import Material

_WL = np.linspace(850.0, 1150.0, 1200)


def _lorentz(wl_nm: np.ndarray, center: float, fwhm_nm: float, peak_m2: float) -> np.ndarray:
    gamma = 0.5 * fwhm_nm
    return peak_m2 * (gamma**2) / ((wl_nm - center) ** 2 + gamma**2)


_sigma_abs = (
    _lorentz(_WL, 940.0, 1.2, 0.75e-24)
    + _lorentz(_WL, 969.0, 0.9, 2.2e-24)
    + _lorentz(_WL, 1020.0, 2.0, 0.15e-24)
)
_sigma_em = _lorentz(_WL, 1030.0, 1.5, 2.5e-24) + _lorentz(_WL, 1050.0, 2.0, 1.2e-24)

YB_YAG = Material(
    name="Yb:YAG",
    wavelength_nm=_WL,
    sigma_abs_m2=_sigma_abs,
    sigma_em_m2=_sigma_em,
    lifetime_s=950e-6,
    n_group=1.82,
)
