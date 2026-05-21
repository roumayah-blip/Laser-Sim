"""Rep-rate steady-state mode tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.gui.runner import SimInputs, run_simulation


def test_rep_rate_cw_runs():
    out = run_simulation(
        SimInputs(
            pump_cw=True,
            pump_duration_s=2e-3,
            burst_start_s=0.0,
            rep_rate_mode=True,
            rep_rate_hz=200e3,
            n_periods=15,
            n_z=60,
        )
    )
    assert out.ok, out.traceback_text
    assert out.result.rep_rate_hz == 200e3
    assert out.result.n_periods_simulated == 15


if __name__ == "__main__":
    test_rep_rate_cw_runs()
    print("rep rate OK")
