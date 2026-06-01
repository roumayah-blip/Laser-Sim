"""Crystal dopant concentration from atomic percent."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hybrid_sim.constants import N_AVOGADRO
from hybrid_sim.materials.base import Material


@dataclass(frozen=True)
class CrystalDopantEstimate:
    concentration_m3: float
    at_percent: float
    density_kg_m3: float
    molar_mass_g_mol: float
    notes: str


def concentration_from_at_percent(
    at_percent: float,
    material: Material,
) -> CrystalDopantEstimate:
    """
    N_dopant [m⁻³] from atomic percent in host.

    N = (at%/100) × ρ × N_A / M_host
    """
    if material.density_kg_m3 <= 0 or material.molar_mass_g_mol <= 0:
        raise ValueError(
            f"Material {material.name} missing density or molar_mass for at.% conversion"
        )
    rho = material.density_kg_m3
    M = material.molar_mass_g_mol * 1e-3  # kg/mol
    n = (at_percent / 100.0) * rho * N_AVOGADRO / M
    return CrystalDopantEstimate(
        concentration_m3=float(n),
        at_percent=float(at_percent),
        density_kg_m3=rho,
        molar_mass_g_mol=material.molar_mass_g_mol,
        notes=f"{at_percent:.3g} at.% in {material.name}",
    )


def validate_cross_sections(material: Material, pump_nm: float, signal_nm: float) -> None:
    sa = float(material.sigma_abs_at(pump_nm)[0])
    se = float(material.sigma_em_at(signal_nm)[0])
    if sa < 1e-25:
        raise ValueError(f"sigma_abs({pump_nm} nm)={sa:.2e} m² — check units/bands")
    if se < 1e-25:
        raise ValueError(f"sigma_em({signal_nm} nm)={se:.2e} m² — check units/bands")
