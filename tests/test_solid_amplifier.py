import numpy as np

from hybrid_sim.calculators.dopant import concentration_from_at_percent
from hybrid_sim.materials import load_material
from hybrid_sim.physics.solid_amplifier import SolidAmplifierConfig, run_solid_amplifier
from hybrid_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec


def test_solid_amplifier_runs_and_conserves():
    mat = load_material("yb_yag")
    dop = concentration_from_at_percent(1.0, mat)
    cfg = SolidAmplifierConfig(
        material=mat,
        crystal_length_m=0.01,
        crystal_diameter_m=0.01,
        yb_concentration_m3=dop.concentration_m3,
        pump_pulse=PumpPulseSpec(peak_power_w=200.0, duration_s=500e-6),
        signal=ChirpedBurstSpec(packet_energy_j=1e-9, burst_start_time_s=400e-6),
        n_z=40,
        n_x=32,
        n_y=32,
        include_thermal=True,
        include_kerr=True,
    )
    res = run_solid_amplifier(cfg)
    assert res.energy_signal_out_j >= 0
    assert res.pump_absorbed_frac > 0.01
    assert res.B_integral_max >= 0
    n0n2 = res.N0_xyz + res.N2_xyz
    assert np.nanmax(np.abs(n0n2 - np.clip(n0n2, 0, 1))) < 0.5
