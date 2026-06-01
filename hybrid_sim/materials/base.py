from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_sim.constants import C0, H, NM_TO_M


@dataclass(frozen=True)
class Material:
    """Doped crystal or glass with tabulated cross-sections and bulk properties."""

    name: str
    wavelength_nm: np.ndarray
    sigma_abs_m2: np.ndarray
    sigma_em_m2: np.ndarray
    lifetime_s: float
    n_group: float = 1.45
    beta2_s2_per_m: float = 0.0
    gamma_nonlinear_per_w_m: float = 0.0
    n2_m2_per_w: float = 0.0
    dndt_per_k: float = 0.0
    thermal_cond_w_mk: float = 0.0
    cp_j_kg_k: float = 0.0
    density_kg_m3: float = 0.0
    damage_fluence_j_m2: float = 5e4
    skip_n1_level: bool = True
    molar_mass_g_mol: float = 0.0
    default_pump_wavelength_nm: float = 940.0
    default_signal_wavelength_nm: float = 1030.0

    def sigma_abs_at(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        return np.interp(
            np.atleast_1d(wavelength_nm),
            self.wavelength_nm,
            self.sigma_abs_m2,
        )

    def sigma_em_at(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        return np.interp(
            np.atleast_1d(wavelength_nm),
            self.wavelength_nm,
            self.sigma_em_m2,
        )

    def photon_energy_j(self, wavelength_nm: float | np.ndarray) -> np.ndarray:
        lam = np.atleast_1d(wavelength_nm) * NM_TO_M
        return H * C0 / lam


def load_material(key: str) -> Material:
    from hybrid_sim.materials import nd_yag, yb_glass, yb_yag, yb_ylf

    table = {
        "yb_glass": yb_glass.YB_GLASS,
        "yb_yag": yb_yag.YB_YAG,
        "yb_ylf": yb_ylf.YB_YLF,
        "nd_yag": nd_yag.ND_YAG,
    }
    key = key.lower().replace("-", "_").replace(" ", "_")
    if key not in table:
        raise KeyError(f"Unknown material {key!r}. Choose from {list(table)}")
    return table[key]


def overlap_cladding_pump(core_radius_m: float, cladding_radius_m: float) -> float:
    if cladding_radius_m <= core_radius_m:
        return 1.0
    return (core_radius_m / cladding_radius_m) ** 2


def core_area_m2(core_radius_m: float) -> float:
    return np.pi * core_radius_m**2


def cladding_area_m2(cladding_radius_m: float) -> float:
    return np.pi * cladding_radius_m**2
