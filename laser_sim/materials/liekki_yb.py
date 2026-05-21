"""
Load Liekki Yb spectroscopic tables from RP Photonics ``*.inc`` format.

The file lists wavelength λ (m), σ_abs (m²), σ_em (m²) on each line after a
``readlist`` header. Upper-state lifetime is read from ``tau_Yb:=0.88 ms``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def load_liekki_yb_cross_sections(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Parse ``Liekki Yb.inc`` (or compatible) into arrays for ``Material``.

    Returns
    -------
    wavelength_nm, sigma_abs_m2, sigma_em_m2, lifetime_s
        Sorted by wavelength; rows with both cross-sections zero are dropped.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")

    lifetime_s = 880e-6
    m = re.search(r"tau_Yb\s*:=\s*([\d.]+)\s*ms", text, re.IGNORECASE)
    if m:
        lifetime_s = float(m.group(1)) * 1e-3

    in_table = False
    rows: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.lower().startswith("readlist"):
            in_table = True
            continue
        if not in_table:
            continue
        if not stripped or stripped.startswith("(*") or stripped.startswith("*"):
            continue
        parts = [p.strip() for p in stripped.split(",")]
        if len(parts) < 3:
            continue
        try:
            lam_m = float(parts[0])
            sa = float(parts[1])
            se = float(parts[2])
        except ValueError:
            continue
        if lam_m <= 0:
            continue
        wl_nm = lam_m * 1e9
        rows.append((wl_nm, sa, se))

    if not rows:
        raise ValueError(f"No spectral rows parsed from {path}")

    arr = np.array(rows, dtype=np.float64)
    order = np.argsort(arr[:, 0])
    arr = arr[order]
    wl = arr[:, 0]
    sa = arr[:, 1]
    se = arr[:, 2]
    nonzero = (sa > 0) | (se > 0)
    if not np.all(nonzero):
        wl, sa, se = wl[nonzero], sa[nonzero], se[nonzero]

    return wl, sa, se, lifetime_s
