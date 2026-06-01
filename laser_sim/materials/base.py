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

    @property
    def wavelength_range_nm(self) -> tuple[float, float]:
        wl = self.wavelength_nm
        return float(wl[0]), float(wl[-1])


def require_pump_cross_sections(
    material: Material,
    pump_wavelength_nm: float,
    *,
    min_sigma_abs_m2: float = 1e-27,
) -> tuple[float, float]:
    """
    Require tabulated pump cross-sections at ``pump_wavelength_nm`` (no extrapolation).

    Raises ``ValueError`` if the wavelength is outside the material table or if
    σ_abs is negligible (no usable pump band).
    """
    wl = float(pump_wavelength_nm)
    lo, hi = material.wavelength_range_nm
    if wl < lo or wl > hi:
        raise ValueError(
            f"Simulation pump wavelength {wl:.2f} nm is outside the cross-section "
            f"table for {material.name} ({lo:.1f}–{hi:.1f} nm). "
            f"Pick a wavelength within the glass data range or choose another material."
        )
    sigma_abs = float(material.sigma_abs_at(wl)[0])
    sigma_em = float(material.sigma_em_at(wl)[0])
    if sigma_abs < min_sigma_abs_m2:
        raise ValueError(
            f"No usable pump absorption at {wl:.2f} nm in {material.name}: "
            f"σ_abs = {sigma_abs:.3e} m² (tabulated range {lo:.1f}–{hi:.1f} nm). "
            f"Try a wavelength on the Yb pump band (e.g. 915–980 nm for Yb:glass)."
        )
    return sigma_abs, sigma_em


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
