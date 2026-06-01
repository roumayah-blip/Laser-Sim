"""Yb:YLF — structured lines (E∥c approximate, 300 K)."""

import numpy as np

from hybrid_sim.constants import (
    YLF_BETA2_S2_PER_M,
    YLF_CP_J_KG_K,
    YLF_DAMAGE_FLUENCE_J_M2,
    YLF_DENSITY_KG_M3,
    YLF_DNDT_PER_K,
    YLF_MOLAR_MASS_G_MOL,
    YLF_N2_M2_PER_W,
    YLF_N_GROUP,
    YLF_THERMAL_COND_W_MK,
)
from hybrid_sim.materials.base import Material

_WL = np.linspace(850.0, 1150.0, 1200)


def _lorentz(wl_nm: np.ndarray, center: float, fwhm_nm: float, peak_m2: float) -> np.ndarray:
    gamma = 0.5 * fwhm_nm
    return peak_m2 * (gamma**2) / ((wl_nm - center) ** 2 + gamma**2)


_sigma_abs = (
    _lorentz(_WL, 916.0, 1.5, 0.9e-24)
    + _lorentz(_WL, 940.0, 1.2, 0.5e-24)
    + _lorentz(_WL, 978.0, 1.0, 1.8e-24)
)
_sigma_em = (
    _lorentz(_WL, 1013.0, 1.8, 1.6e-24)
    + _lorentz(_WL, 1022.0, 2.0, 2.2e-24)
    + _lorentz(_WL, 1048.0, 2.5, 0.8e-24)
)

YB_YLF = Material(
    name="Yb:YLF",
    wavelength_nm=_WL,
    sigma_abs_m2=_sigma_abs,
    sigma_em_m2=_sigma_em,
    lifetime_s=2.0e-3,
    n_group=YLF_N_GROUP,
    beta2_s2_per_m=YLF_BETA2_S2_PER_M,
    n2_m2_per_w=YLF_N2_M2_PER_W,
    dndt_per_k=YLF_DNDT_PER_K,
    thermal_cond_w_mk=YLF_THERMAL_COND_W_MK,
    cp_j_kg_k=YLF_CP_J_KG_K,
    density_kg_m3=YLF_DENSITY_KG_M3,
    damage_fluence_j_m2=YLF_DAMAGE_FLUENCE_J_M2,
    molar_mass_g_mol=YLF_MOLAR_MASS_G_MOL,
    default_pump_wavelength_nm=960.0,
    default_signal_wavelength_nm=1020.0,
)
