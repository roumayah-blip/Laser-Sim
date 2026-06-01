"""Pump wavelength: datasheet N-calc vs simulation, with table bounds."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.gui.runner import SimInputs, simulation_pump_wavelength_nm
from laser_sim.materials import load_material
from laser_sim.materials.base import require_pump_cross_sections


def test_simulation_pump_defaults_to_datasheet():
    inp = SimInputs(pump_wavelength_nm=976.0, simulation_pump_wavelength_nm=None)
    assert simulation_pump_wavelength_nm(inp) == 976.0
    inp2 = SimInputs(pump_wavelength_nm=976.0, simulation_pump_wavelength_nm=940.0)
    assert simulation_pump_wavelength_nm(inp2) == 940.0


def test_require_pump_in_table_range():
    mat = load_material("yb_glass")
    sa, se = require_pump_cross_sections(mat, 976.0)
    assert sa > 1e-25
    assert se >= 0.0


def test_require_pump_outside_table_raises():
    mat = load_material("yb_glass")
    lo, hi = mat.wavelength_range_nm
    with pytest.raises(ValueError, match="outside the cross-section table"):
        require_pump_cross_sections(mat, hi + 5.0)
    with pytest.raises(ValueError, match="outside the cross-section table"):
        require_pump_cross_sections(mat, lo - 5.0)


def test_run_rejects_out_of_range_sim_pump():
    from laser_sim.gui.runner import run_simulation

    out = run_simulation(
        SimInputs(
            material_key="yb_glass",
            simulation_pump_wavelength_nm=1200.0,
            pump_wavelength_nm=976.0,
            n_z=30,
            include_ase=False,
            backend="cpu",
        )
    )
    assert not out.ok
    assert "outside the cross-section table" in out.error_message


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
