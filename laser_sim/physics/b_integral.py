"""B-integral (nonlinear phase) calculator for fiber amplifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Nonlinear index of fused silica at 1030 nm [m²/W]
N2_SILICA_M2_PER_W = 2.6e-20


@dataclass(frozen=True)
class BIntegralResult:
    b_passive_before_rad: float
    b_active_rad: float
    b_passive_after_rad: float
    b_total_rad: float
    l_nl_passive_before_m: float
    l_nl_active_m: float
    l_nl_passive_after_m: float
    p_peak_in_w: float
    p_peak_out_w: float
    a_eff_m2: float
    severity: str
    notes: str


def _severity(b_total_rad: float) -> str:
    if b_total_rad < 1.0:
        return "excellent"
    if b_total_rad < 3.0:
        return "moderate"
    if b_total_rad < 5.0:
        return "significant"
    return "severe"


def _nonlinear_length_m(
    wavelength_m: float,
    n2_m2_per_w: float,
    p_peak_w: float,
    a_eff_m2: float,
) -> float:
    return wavelength_m * a_eff_m2 / (2.0 * np.pi * n2_m2_per_w * max(p_peak_w, 1e-30))


def compute_b_integral(
    *,
    wavelength_m: float,
    a_eff_m2: float,
    p_peak_in_w: float,
    p_peak_out_w: float,
    l_active_m: float,
    l_passive_before_m: float = 0.0,
    l_passive_after_m: float = 0.0,
    n2_m2_per_w: float = N2_SILICA_M2_PER_W,
) -> BIntegralResult:
    """
    Compute B-integral for active fiber flanked by optional passive sections.

    Uses average power approximation for the active section:
        B_active = (2π/λ) * n2 * ((P_in + P_out)/2) / A_eff * L_active

    Uses input peak power for passive-before and output peak power for passive-after:
        B_passive_before = (2π/λ) * n2 * P_in  / A_eff * L_passive_before
        B_passive_after  = (2π/λ) * n2 * P_out / A_eff * L_passive_after
    """
    a_eff = max(float(a_eff_m2), 1e-30)
    coeff = (2.0 * np.pi / wavelength_m) * n2_m2_per_w / a_eff

    b_passive_before = coeff * p_peak_in_w * l_passive_before_m
    p_avg_active = 0.5 * (p_peak_in_w + p_peak_out_w)
    b_active = coeff * p_avg_active * l_active_m
    b_passive_after = coeff * p_peak_out_w * l_passive_after_m
    b_total = b_passive_before + b_active + b_passive_after

    l_nl_before = _nonlinear_length_m(wavelength_m, n2_m2_per_w, p_peak_in_w, a_eff)
    l_nl_active = _nonlinear_length_m(wavelength_m, n2_m2_per_w, p_peak_out_w, a_eff)
    l_nl_after = _nonlinear_length_m(wavelength_m, n2_m2_per_w, p_peak_out_w, a_eff)

    sev = _severity(b_total)
    notes = (
        f"B_total={b_total:.4f} rad ({sev}); "
        f"P_peak {p_peak_in_w:.4g}→{p_peak_out_w:.4g} W; A_eff={a_eff*1e12:.2f} µm²"
    )

    return BIntegralResult(
        b_passive_before_rad=b_passive_before,
        b_active_rad=b_active,
        b_passive_after_rad=b_passive_after,
        b_total_rad=b_total,
        l_nl_passive_before_m=l_nl_before,
        l_nl_active_m=l_nl_active,
        l_nl_passive_after_m=l_nl_after,
        p_peak_in_w=p_peak_in_w,
        p_peak_out_w=p_peak_out_w,
        a_eff_m2=a_eff,
        severity=sev,
        notes=notes,
    )
