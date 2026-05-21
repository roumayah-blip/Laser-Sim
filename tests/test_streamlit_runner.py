"""Regression tests matching Streamlit run-tab scenarios."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.gui.runner import SimInputs, run_simulation


def _run(inp: SimInputs) -> None:
    out = run_simulation(inp)
    assert out.ok, f"{out.error_type}: {out.error_message}\n{out.traceback_text}"


def test_defaults_pulsed():
    _run(SimInputs())


def test_cw_pump():
    _run(SimInputs(pump_cw=True, pump_duration_s=3e-3))


def test_yb_yag_total_db():
    _run(
        SimInputs(
            material_key="yb_yag",
            abs_mode_db_per_m=False,
            total_absorption_db=15.0,
            burst_spacing_s=0.5e-9,
            burst_count=10,
        )
    )


def test_yb_ylf_high_power():
    _run(
        SimInputs(
            material_key="yb_ylf",
            pump_peak_power_w=500.0,
            n_z=200,
            include_ase=False,
        )
    )


if __name__ == "__main__":
    test_defaults_pulsed()
    test_cw_pump()
    test_yb_yag_total_db()
    test_yb_ylf_high_power()
    print("all streamlit scenarios OK")
