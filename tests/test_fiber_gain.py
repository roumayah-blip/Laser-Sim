"""Fiber CPA should show gain when CW-pumped before the packet."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.gui.runner import SimInputs, run_simulation
from laser_sim.materials.yb_glass import YB_GLASS


def test_liekki_absorption_nonzero_signal_band():
    wl = YB_GLASS.wavelength_nm
    sa = YB_GLASS.sigma_abs_m2
    mask = (wl >= 1000) & (wl <= 1100)
    assert np.all(sa[mask] > 0), f"σ_abs must be positive in signal band; min={sa[mask].min()}"


def test_transparency_threshold():
    idx = np.argmin(np.abs(YB_GLASS.wavelength_nm - 1030.0))
    sa = YB_GLASS.sigma_abs_m2[idx]
    se = YB_GLASS.sigma_em_m2[idx]
    threshold = sa / (sa + se)
    assert 0.05 < threshold < 0.25, (
        f"Transparency threshold {threshold:.3f} out of expected range [0.05, 0.25]"
    )
    assert sa < se, f"σ_abs={sa:.3e} >= σ_em={se:.3e} at 1030nm — check tabulated spectra"


def test_liekki_pump_band_near_resonance_976():
    """Liekki table: σ_abs and σ_em are comparable near 976 nm (zero-phonon region)."""
    idx = np.argmin(np.abs(YB_GLASS.wavelength_nm - 976.0))
    sa = YB_GLASS.sigma_abs_m2[idx]
    se = YB_GLASS.sigma_em_m2[idx]
    ratio = sa / max(se, 1e-40)
    assert 0.8 < ratio < 1.2, f"σ_abs/σ_em at 976 nm = {ratio:.3f}, expected ≈ 1 (Liekki table)"


def test_cw_pumped_packet_shows_gain():
    # McCumber σ_abs at 1030 nm requires ~8% inversion for transparency; use
    # core-pumped 30 µm / 250 µm with enough pump to exceed that threshold.
    inp = SimInputs(
        pump_cw=True,
        pump_peak_power_w=100_000.0,
        core_diameter_um=30.0,
        cladding_diameter_um=250.0,
        cladding_pumped=False,
        burst_start_s=200e-6,
        burst_count=5,
        burst_spacing_s=2.5e-9,
        chirp_duration_s=0.8e-9,
        n_z=60,
        include_ase=False,
    )
    out = run_simulation(inp)
    assert out.ok, out.error_message
    r = out.result
    gain = r.energy_packet_out_j / max(r.energy_packet_in_j, 1e-30)
    assert gain > 1.001, f"expected measurable packet gain, got {gain}"
    assert r.n2_fraction.max() > 1e-4, "inversion should build under CW pump"
