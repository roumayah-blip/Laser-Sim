"""
Back-calculate Yb³⁺ concentration from fiber pump absorption specification.

Small-signal, ground-state dominated (N₁ ≈ N₀ ≈ N):

    α_dB/m  — datasheet pump attenuation (measured on the fiber)
    α_np/m  = α_dB/m / (10·log₁₀ e)     [Napier/m],  DB_PER_NP = 10/ln(10)

Cladding-pumped (Yb in core, pump in cladding):

    Γ_p = (r_core / r_clad)²
    α_np = Γ_p · σ_abs(λ_p) · N

Core-pumped:

    Γ_p = 1
    α_np = σ_abs(λ_p) · N

Solve for ion density in the core:

    N = α_np / (Γ_p · σ_abs(λ_p))

So for fixed κ (dB/m) and cladding, a larger core ⇒ larger Γ_p ⇒ lower N.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.constants import DB_PER_NP
from laser_sim.materials.base import (
    Material,
    cladding_area_m2,
    core_area_m2,
    overlap_cladding_pump,
)


@dataclass(frozen=True)
class DopantEstimate:
    """Result of dopant concentration back-calculation."""

    concentration_m3: float
    concentration_ppm_wt: float  # approximate for silica, Yb₂O₃ equivalent scale
    gamma_pump: float
    sigma_abs_pump_m2: float
    alpha_np_per_m: float
    alpha_db_per_m: float
    pump_area_m2: float
    concentration_for_rates_m3: float
    notes: str


def estimate_dopant_concentration(
    *,
    pump_absorption_db_per_m: float,
    core_diameter_um: float,
    cladding_diameter_um: float,
    pump_wavelength_nm: float,
    material: Material,
    cladding_pumped: bool = True,
    overlap_gamma_pump: float | None = None,
    ignore_overlap_for_concentration: bool = False,
    apply_overlap_factor: bool | None = None,
) -> DopantEstimate:
    """
    Estimate Yb³⁺ ion density N (m⁻³) from specified pump absorption.

    Parameters
    ----------
    pump_absorption_db_per_m
        Small-signal pump absorption coefficient (dB/m), e.g. 6 dB/m at 976 nm.
    core_diameter_um, cladding_diameter_um
        Fiber core and pump-cladding diameters (µm); set Γ_p for cladding pump.
    ignore_overlap_for_concentration
        If True, use N = α_np/σ only (legacy). Default False uses N = α_np/(Γ_p σ).
    apply_overlap_factor
        Deprecated alias: True ≡ ignore_overlap False.
    """
    if apply_overlap_factor is not None:
        ignore_overlap_for_concentration = not apply_overlap_factor

    r_core = 0.5 * core_diameter_um * 1e-6
    r_clad = 0.5 * cladding_diameter_um * 1e-6

    sigma = float(material.sigma_abs_at(pump_wavelength_nm)[0])
    if sigma < 1e-25:
        raise ValueError(
            f"sigma_abs at pump wavelength is suspiciously small "
            f"({sigma:.2e} m^2). Check that the material cross-sections are in m^2 "
            f"not cm^2, and that the pump wavelength overlaps an absorption band."
        )

    if overlap_gamma_pump is not None:
        gamma = overlap_gamma_pump
        if cladding_pumped:
            pump_area = cladding_area_m2(r_clad)
        else:
            pump_area = core_area_m2(r_core)
    elif cladding_pumped:
        gamma = overlap_cladding_pump(r_core, r_clad)
        pump_area = cladding_area_m2(r_clad)
    else:
        gamma = 1.0
        pump_area = core_area_m2(r_core)

    alpha_np = pump_absorption_db_per_m / DB_PER_NP
    denom = sigma if ignore_overlap_for_concentration else gamma * sigma
    concentration = alpha_np / denom

    molar_mass_sio2 = 60.08e-3
    n_silica = 2200.0
    n_sites = n_silica / molar_mass_sio2 * 6.022e23
    ppm_wt = 1e6 * concentration / n_sites

    if ignore_overlap_for_concentration:
        overlap_note = "N = κ_np/σ (Γ_p ignored — not recommended for cladding pump)."
    else:
        overlap_note = "N = κ_np/(Γ_p·σ); κ is measured fiber attenuation."

    notes = (
        f"Small-signal, N₁≈N. Γ_p={gamma:.4f}, σ_abs={sigma:.3e} m², "
        f"A_pump={pump_area*1e12:.1f} µm². {overlap_note} "
        "Bleaching and co-pumping may require lower effective N."
    )

    return DopantEstimate(
        concentration_m3=concentration,
        concentration_ppm_wt=ppm_wt,
        gamma_pump=gamma,
        sigma_abs_pump_m2=sigma,
        alpha_np_per_m=alpha_np,
        alpha_db_per_m=pump_absorption_db_per_m,
        pump_area_m2=pump_area,
        concentration_for_rates_m3=concentration,
        notes=notes,
    )


def concentration_from_total_absorption_db(
    total_absorption_db: float,
    fiber_length_m: float,
    **kwargs,
) -> DopantEstimate:
    """Convert total pump absorption over length L to dB/m, then estimate N."""
    if fiber_length_m <= 0:
        raise ValueError("fiber_length_m must be positive")
    db_per_m = total_absorption_db / fiber_length_m
    return estimate_dopant_concentration(pump_absorption_db_per_m=db_per_m, **kwargs)
