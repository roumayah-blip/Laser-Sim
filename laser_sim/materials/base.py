from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from laser_sim.constants import C0, H, NM_TO_M


@dataclass(frozen=True)
class Material:
    """Ytterbium-doped host with tabulated cross-sections."""

    name: str
    wavelength_nm: np.ndarray
    sigma_abs_m2: np.ndarray
    sigma_em_m2: np.ndarray
    lifetime_s: float
    # Hooks for future fs/ps NLSE work (unused in CPA ns regime)
    n_group: float = 1.45
    beta2_s2_per_m: float = 0.0
    gamma_nonlinear_per_w_m: float = 0.0

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
    from laser_sim.materials import yb_glass, yb_yag, yb_ylf

    table = {
        "yb_glass": yb_glass.YB_GLASS,
        "yb_yag": yb_yag.YB_YAG,
        "yb_ylf": yb_ylf.YB_YLF,
    }
    key = key.lower().replace("-", "_").replace(" ", "_")
    if key not in table:
        raise KeyError(f"Unknown material {key!r}. Choose from {list(table)}")
    return table[key]


def overlap_cladding_pump(core_radius_m: float, cladding_radius_m: float) -> float:
    """Power filling factor for cladding pump (step-index, Yb in core only)."""
    if cladding_radius_m <= core_radius_m:
        return 1.0
    return (core_radius_m / cladding_radius_m) ** 2


def core_area_m2(core_radius_m: float) -> float:
    return np.pi * core_radius_m**2


def cladding_area_m2(cladding_radius_m: float) -> float:
    return np.pi * cladding_radius_m**2
