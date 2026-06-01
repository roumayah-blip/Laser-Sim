"""Rep-rate steady-state mode tests."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.gui.runner import SimInputs, run_simulation


def test_rep_rate_cw_runs_no_warmup():
    """Legacy mode: resolve every period explicitly (warmup disabled)."""
    out = run_simulation(
        SimInputs(
            pump_cw=True,
            pump_duration_s=2e-3,
            burst_start_s=0.0,
            rep_rate_mode=True,
            rep_rate_hz=200e3,
            n_periods=15,
            n_z=60,
            steady_state_warmup=False,
        )
    )
    assert out.ok, out.traceback_text
    assert out.result.rep_rate_hz == 200e3
    assert out.result.n_periods_simulated == 15
    assert out.steady_state_warmup_used is False


def test_rep_rate_cw_runs_with_warmup():
    """RP-style warmup: lumped per-period iteration, then 1 time-resolved period."""
    out = run_simulation(
        SimInputs(
            pump_cw=True,
            pump_duration_s=2e-3,
            burst_start_s=0.0,
            rep_rate_mode=True,
            rep_rate_hz=200e3,
            n_periods=15,
            n_z=60,
            steady_state_warmup=True,
        )
    )
    assert out.ok, out.traceback_text
    assert out.result.rep_rate_hz == 200e3
    # Warmup collapses the time-resolved sim to a single period.
    assert out.result.n_periods_simulated == 1
    assert out.steady_state_warmup_used is True
    assert out.steady_state_warmup_iter is not None and out.steady_state_warmup_iter >= 1


if __name__ == "__main__":
    test_rep_rate_cw_runs_no_warmup()
    test_rep_rate_cw_runs_with_warmup()
    print("rep rate OK")
