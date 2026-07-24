"""Pump absorption vs specified dB/m."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.constants import DB_PER_NP
from laser_sim.gui.runner import SimInputs, run_simulation


def test_pump_absorption_17_db_per_m_over_2m():
    length_m = 2.0
    db_per_m = 17.0
    inp = SimInputs(
        pump_cw=True,
        pump_peak_power_w=300.0,
        pump_absorption_db_per_m=db_per_m,
        fiber_length_m=length_m,
        n_z=150,
        burst_start_s=200e-6,
    )
    out = run_simulation(inp)
    assert out.ok, out.error_message
    r = out.result
    kappa = db_per_m / DB_PER_NP
    expected = 1.0 - np.exp(-kappa * length_m)
    assert r.pump_power_absorbed_fraction > 0.85, (
        f"expected >85% absorbed for {db_per_m} dB/m × {length_m} m, "
        f"got {r.pump_power_absorbed_fraction*100:.1f}%"
    )
    assert r.populations.n0.mean() > 0.5, "N0 should remain majority under CW cladding pump"


def test_backward_pump_propagates_along_fiber():
    """Backward pump must reach the whole fiber, not just the last z-slices.

    Regression test: a single forward z-sweep left the backward pump nonzero
    only at the last two slices, so a backward-pumped amplifier had almost no
    inversion along the fiber.
    """
    from laser_sim.materials import load_material
    from laser_sim.physics.fiber_cpa import FiberCPAConfig, run_fiber_cpa
    from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec

    mat = load_material("yb_glass")
    spec = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        packet_energy_j=1e-6,
        burst_count=1,
        burst_start_time_s=50e-6,
    )
    pump = PumpPulseSpec(wavelength_nm=976.0, peak_power_w=20.0, cw=True, duration_s=2e-3)
    base = dict(
        material=mat,
        fiber_length_m=0.3,
        n_z=12,
        yb_concentration_m3=5e25,
        include_ase=False,
    )
    bwd = run_fiber_cpa(
        FiberCPAConfig(
            signal=spec,
            pump=pump,
            forward_pump=False,
            backward_pump=True,
            backward_pump_fraction=1.0,
            **base,
        ),
        backend="cpu",
    )
    fwd = run_fiber_cpa(FiberCPAConfig(signal=spec, pump=pump, **base), backend="cpu")

    p_bwd_z = np.mean(bwd.pump_bwd_w, axis=1)
    # Present along the whole fiber, monotonically decaying from z=L toward z=0.
    assert p_bwd_z[0] > 0.5 * p_bwd_z[-1]
    assert np.all(np.diff(p_bwd_z) >= -1e-9)
    # Inversion roughly uniform, not confined to the launch end.
    n2_z = np.mean(bwd.n2_fraction, axis=1)
    assert n2_z[0] > 0.5 * n2_z[-1]
    # Weak absorption here → backward pumping ≈ forward pumping by symmetry.
    assert bwd.energy_packet_out_j == pytest.approx(fwd.energy_packet_out_j, rel=0.05)
