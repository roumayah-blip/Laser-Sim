"""
Yb-doped silica/glass cross-sections for fiber CPA simulation.

Default material ``YB_GLASS`` uses tabulated Liekki Yb1200-* data from
``Liekki_Yb.inc`` (wavelength in m, σ_abs and σ_em in m² per row).

The legacy Gaussian model ``YB_GLASS_GAUSSIAN`` is retained for comparison only;
its σ_abs(976 nm) ≈ 2.4×10⁻²⁵ m² was ~10× below the Liekki measurement (~2.5×10⁻²⁴ m²),
which inflated N_tot from dB/m specs and pump power needed for transparency.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from hybrid_sim.materials.base import Material

_INC_PATH = Path(__file__).resolve().parent / "Liekki_Yb.inc"
_TAU_21_S = 0.88e-3  # tau_Yb := 0.88 ms in Liekki_Yb.inc


def _gaussian(wl_nm: np.ndarray, center: float, fwhm_nm: float, peak_m2: float) -> np.ndarray:
    sigma = fwhm_nm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    return peak_m2 * np.exp(-0.5 * ((wl_nm - center) / sigma) ** 2)


def _parse_liekki_yb_inc(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse RP Photonics readlist table: wavelength_m, sigma_abs_m2, sigma_em_m2."""
    text = path.read_text(encoding="utf-8", errors="replace")
    in_table = False
    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("readlist"):
            in_table = True
            continue
        if not in_table or not stripped or stripped.startswith("(*"):
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) < 3:
            continue
        try:
            lam_m = float(parts[0])
            sigma_abs = float(parts[1])
            sigma_em = float(parts[2])
        except ValueError:
            continue
        if lam_m <= 0:
            continue
        # Exclude trailing (1.140E-06, 0, 0) sentinel row
        if sigma_abs == 0.0 and sigma_em == 0.0:
            continue
        rows.append((lam_m * 1e9, sigma_abs, sigma_em))

    if not rows:
        raise ValueError(f"No spectral rows parsed from {path}")

    arr = np.array(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    return arr[:, 0], arr[:, 1], arr[:, 2]


def _sigma_at(wl_nm: float, wl_grid: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.interp(wl_nm, wl_grid, sigma))


_WL_NM, _SIGMA_ABS, _SIGMA_EM = _parse_liekki_yb_inc(_INC_PATH)

YB_GLASS_LIEKKI = Material(
    name="Yb:glass (Liekki)",
    wavelength_nm=_WL_NM,
    sigma_abs_m2=_SIGMA_ABS,
    sigma_em_m2=_SIGMA_EM,
    lifetime_s=_TAU_21_S,
    n_group=1.45,
)

# Deprecated: analytical Gaussians (σ_abs at 976 nm ~10× too low vs Liekki table).
_WL_GAUSS = np.linspace(850.0, 1150.0, 600)
_SIGMA_EM_G = (
    _gaussian(_WL_GAUSS, 1030.0, 35.0, 2.0e-25)
    + _gaussian(_WL_GAUSS, 1060.0, 40.0, 1.4e-25)
)
_SIGMA_ABS_G = (
    _gaussian(_WL_GAUSS, 976.0, 8.0, 2.4e-25)
    + _gaussian(_WL_GAUSS, 915.0, 12.0, 0.6e-25)
)
YB_GLASS_GAUSSIAN = Material(
    name="Yb:glass (Gaussian, deprecated)",
    wavelength_nm=_WL_GAUSS,
    sigma_abs_m2=_SIGMA_ABS_G,
    sigma_em_m2=_SIGMA_EM_G,
    lifetime_s=896e-6,
    n_group=1.45,
)

YB_GLASS = YB_GLASS_LIEKKI

_sa976 = _sigma_at(976.0, _WL_NM, _SIGMA_ABS)
_se976 = _sigma_at(976.0, _WL_NM, _SIGMA_EM)
_sa1030 = _sigma_at(1030.0, _WL_NM, _SIGMA_ABS)
_se1030 = _sigma_at(1030.0, _WL_NM, _SIGMA_EM)

assert 1e-24 <= _sa976 <= 5e-24, f"σ_abs(976 nm)={_sa976:.3e} m² outside [1e-24, 5e-24]"
assert 3e-25 <= _se1030 <= 1.5e-24, f"σ_em(1030 nm)={_se1030:.3e} m² outside [3e-25, 1.5e-24]"
assert _sa1030 > 0.0, f"σ_abs(1030 nm) underflow: {_sa1030:.3e} m²"
_ratio_976 = _sa976 / max(_se976, 1e-40)
assert 0.8 <= _ratio_976 <= 1.3, f"σ_abs/σ_em at 976 nm = {_ratio_976:.3f}, expected quasi-2L resonance ≈ 1"
