"""Yb:YAG — narrow lines (300 K, π polarization approximate)."""

import numpy as np

from hybrid_sim.constants import (
    YAG_BETA2_S2_PER_M,
    YAG_CP_J_KG_K,
    YAG_DAMAGE_FLUENCE_J_M2,
    YAG_DENSITY_KG_M3,
    YAG_DNDT_PER_K,
    YAG_MOLAR_MASS_G_MOL,
    YAG_N2_M2_PER_W,
    YAG_N_GROUP,
    YAG_THERMAL_COND_W_MK,
)
from hybrid_sim.materials.base import Material

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
    n_group=YAG_N_GROUP,
    beta2_s2_per_m=YAG_BETA2_S2_PER_M,
    n2_m2_per_w=YAG_N2_M2_PER_W,
    dndt_per_k=YAG_DNDT_PER_K,
    thermal_cond_w_mk=YAG_THERMAL_COND_W_MK,
    cp_j_kg_k=YAG_CP_J_KG_K,
    density_kg_m3=YAG_DENSITY_KG_M3,
    damage_fluence_j_m2=YAG_DAMAGE_FLUENCE_J_M2,
    molar_mass_g_mol=YAG_MOLAR_MASS_G_MOL,
    default_pump_wavelength_nm=940.0,
    default_signal_wavelength_nm=1030.0,
)
