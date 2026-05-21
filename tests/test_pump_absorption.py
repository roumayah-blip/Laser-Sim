"""Pump absorption vs specified dB/m."""

import sys
from pathlib import Path

import numpy as np

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
