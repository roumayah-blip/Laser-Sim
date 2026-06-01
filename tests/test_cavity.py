import numpy as np

from hybrid_sim.materials import load_material
from hybrid_sim.physics.cavity import CavityConfig, frantz_nodvik_energy, run_cavity_simulation


def test_cavity_threshold_and_pulse():
    mat = load_material("nd_yag")
    cfg = CavityConfig(
        crystal=mat,
        crystal_length_m=0.01,
        cavity_length_m=0.15,
        yb_concentration_m3=1e26,
        pump_power_w=500.0,
        n_roundtrips=2000,
        q_switch_on_time_s=50e-6,
    )
    res = run_cavity_simulation(cfg)
    assert res.pulse_energy_j > 0 or np.max(res.e_intra_j) > cfg.seed_energy_j * 10
    assert np.max(res.g0_per_pass) > 0

    g0 = float(np.max(res.g0_per_pass))
    e_fn = frantz_nodvik_energy(
        g0=g0,
        length_m=cfg.crystal_length_m,
        sigma_e=float(mat.sigma_em_at(1064)[0]),
        sigma_a=float(mat.sigma_abs_at(1064)[0]),
        area_m2=np.pi * 0.005**2,
        wavelength_m=1064e-9,
        e_in_j=cfg.seed_energy_j,
    )
    if e_fn > 1e-12:
        rel = abs(res.pulse_energy_j - e_fn) / e_fn
        assert rel < 0.5, f"FN mismatch {rel:.2f}"
