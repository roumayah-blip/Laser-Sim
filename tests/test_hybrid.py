import pytest

from hybrid_sim.gui.runner import HybridSimInputs, run_hybrid_safe


@pytest.mark.slow
def test_hybrid_end_to_end():
    inp = HybridSimInputs(
        fiber_length_m=0.5,
        fiber_pump_duration_us=200.0,
        ss_n_z=30,
        ss_n_x=24,
        ss_n_y=24,
        use_cavity_pump=False,
        include_thermal=False,
        include_kerr=False,
    )
    out = run_hybrid_safe(inp)
    assert out.ok, out.error_message
    assert out.result is not None
    assert out.result.solid.energy_signal_out_j > 0
