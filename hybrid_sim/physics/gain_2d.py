"""2D gain coefficient wrapper (quasi-2-level)."""

from __future__ import annotations

import numpy as np


def gain_coefficient_xy(
    n0: np.ndarray,
    n2: np.ndarray,
    sigma_a: float,
    sigma_e: float,
    *,
    gamma_s: float = 1.0,
    n_tot: float = 1.0,
) -> np.ndarray:
    """
    Small-signal gain g [m⁻¹] on 2D population fractions.

    g = Γ [σ_e N₂ − σ_a N₀], quasi-2L with N₀ ground depleted.
    """
    n0_d = np.asarray(n0, dtype=np.float64) * n_tot
    n2_d = np.asarray(n2, dtype=np.float64) * n_tot
    return gamma_s * (sigma_e * n2_d - sigma_a * n0_d)
