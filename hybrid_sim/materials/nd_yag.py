"""Nd:YAG — 4-level pump at 808 nm, emission at 1064 nm."""

import numpy as np

from hybrid_sim.constants import (
    ND_YAG_DAMAGE_FLUENCE_J_M2,
    ND_YAG_DENSITY_KG_M3,
    ND_YAG_DNDT_PER_K,
    ND_YAG_N2_M2_PER_W,
    ND_YAG_N_GROUP,
    ND_YAG_THERMAL_COND_W_MK,
)
from hybrid_sim.materials.base import Material

_WL = np.linspace(750.0, 1200.0, 1200)


def _lorentz(wl_nm: np.ndarray, center: float, fwhm_nm: float, peak_m2: float) -> np.ndarray:
    gamma = 0.5 * fwhm_nm
    return peak_m2 * (gamma**2) / ((wl_nm - center) ** 2 + gamma**2)


_sigma_abs = _lorentz(_WL, 808.0, 2.0, 6.5e-24) + _lorentz(_WL, 880.0, 15.0, 0.3e-24)
_sigma_em = _lorentz(_WL, 1064.0, 0.8, 2.8e-24) + _lorentz(_WL, 1338.0, 3.0, 0.5e-24)

ND_YAG = Material(
    name="Nd:YAG",
    wavelength_nm=_WL,
    sigma_abs_m2=_sigma_abs,
    sigma_em_m2=_sigma_em,
    lifetime_s=230e-6,
    n_group=ND_YAG_N_GROUP,
    n2_m2_per_w=ND_YAG_N2_M2_PER_W,
    dndt_per_k=ND_YAG_DNDT_PER_K,
    thermal_cond_w_mk=ND_YAG_THERMAL_COND_W_MK,
    cp_j_kg_k=590.0,
    density_kg_m3=ND_YAG_DENSITY_KG_M3,
    damage_fluence_j_m2=ND_YAG_DAMAGE_FLUENCE_J_M2,
    skip_n1_level=False,
    default_pump_wavelength_nm=808.0,
    default_signal_wavelength_nm=1064.0,
)
