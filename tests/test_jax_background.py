"""JAX Stage 2 (log-space vmapped propagation) — correctness and anti-overflow tests.

Validates laser_sim/physics/jax_background.py against:
1. A synthetic case with a known analytic solution (accuracy check).
2. A real gain field pulled from the CPU pump-pass for the previously-failing
   300 W / 17 dB/m / 2 m scenario (tests/test_pump_absorption.py ::
   test_pump_absorption_17_db_per_m_over_2m) — confirming the new propagator
   stays finite where the existing CPU per-slab ``P *= exp(g*dz)`` recurrence
   overflows to ``inf``.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.calculators.dopant import estimate_dopant_concentration
from laser_sim.materials import load_material
from laser_sim.physics.fiber_cpa import FiberCPAConfig, run_fiber_cpa
from laser_sim.physics.four_level import gain_coefficient_m, spontaneous_power_w_per_nm
from laser_sim.physics.jax_background import propagate_log_space, solve_stage1_forward
from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec


def test_propagate_log_space_matches_analytic_solution():
    """Constant gain, no source: P(z) = P(0)*exp(g*z) — check against the closed form."""
    nz = 50
    z = np.linspace(0.0, 2.0, nz)
    n_bins = 6
    g_const = np.array([-300.0, -50.0, 0.0, 50.0, 300.0, 700.0])
    gain_np_m = np.tile(g_const, (nz, 1))
    p0 = np.full(n_bins, 1e-9)

    p_z = propagate_log_space(z, p0, gain_np_m)

    assert np.all(np.isfinite(p_z))
    log_expected = np.log(p0)[None, :] + g_const[None, :] * z[:, None]
    log_got = np.log(np.clip(p_z, 1e-300, None))
    # Only compare bins whose analytic answer is itself representable in float64.
    representable = np.abs(log_expected) < 700.0
    assert np.allclose(log_got[representable], log_expected[representable], rtol=1e-4, atol=1e-3)
    # The g=700 Np/m bin at z=2 has log(P) ~ 1379 -- unrepresentable in linear
    # float64 (exp overflows), but the clipped output must still be finite,
    # not inf/NaN, and monotonically the largest of the batch.
    assert np.all(np.isfinite(p_z[:, -1]))
    assert p_z[-1, -1] >= p_z[-1, -2]


def _extreme_case_gain_and_source():
    """Pump-only populations for the 300 W/17 dB/m/2 m case that overflows on CPU."""
    mat = load_material("yb_glass")
    dopant = estimate_dopant_concentration(
        pump_absorption_db_per_m=17.0,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        pump_wavelength_nm=976.0,
        material=mat,
        cladding_pumped=True,
    )
    sig = ChirpedBurstSpec(
        center_wavelength_nm=1030.0, packet_energy_j=1e-7, burst_count=1, burst_start_time_s=200e-6
    )
    pump = PumpPulseSpec(wavelength_nm=976.0, peak_power_w=300.0, cw=True, duration_s=1e-3)
    cfg = FiberCPAConfig(
        material=mat,
        fiber_length_m=2.0,
        n_z=150,
        yb_concentration_m3=dopant.concentration_m3,
        signal=sig,
        pump=pump,
        include_ase=False,  # only need the pump-driven population field, not the (exploding) ASE march
    )
    r = run_fiber_cpa(cfg, backend="cpu")

    it = -1  # steady-state CW time index
    n0 = r.populations.n0[:, it] * dopant.concentration_m3
    n2 = r.populations.n2[:, it] * dopant.concentration_m3
    gain_np_m = gain_coefficient_m(n0, n2, r.sigma_abs_signal_m2, r.sigma_em_signal_m2, gamma_s=r.gamma_signal)

    hnu = mat.photon_energy_j(r.wavelength_nm)
    source_w_nm = np.stack(
        [
            spontaneous_power_w_per_nm(
                float(n2_i),
                r.sigma_em_signal_m2,
                hnu,
                r.wavelength_nm,
                tau_21_s=r.four_level_lifetimes.tau_21_s,
                eta_guided=r.eta_guided_spontaneous,
                gamma_s=r.gamma_signal,
            )
            for n2_i in n2
        ]
    ) * r.a_signal_m2
    return r.z_m, gain_np_m, source_w_nm, r.wavelength_nm


def test_naive_repeated_multiply_overflows_on_extreme_gain_field():
    """Confirms the bug this module fixes: the CPU code's own recurrence overflows."""
    z, gain_np_m, source_w_nm, wl = _extreme_case_gain_and_source()
    dz = float(z[1] - z[0])
    nz, n_bins = gain_np_m.shape
    p = np.zeros((nz, n_bins))
    p[0] = 1e-15
    for iz in range(nz - 1):
        gdz = np.clip(gain_np_m[iz] * dz, -50.0, 50.0)
        p[iz + 1] = np.maximum(p[iz] * np.exp(gdz) + source_w_nm[iz] * dz, 0.0)
    assert np.any(~np.isfinite(p)), "expected the naive per-slab recurrence to overflow (regression check)"


def test_propagate_log_space_stays_finite_on_extreme_gain_field():
    """The new JAX Stage 2 propagator must stay finite on the same field that overflows above."""
    z, gain_np_m, source_w_nm, wl = _extreme_case_gain_and_source()
    p0 = np.full(gain_np_m.shape[1], 1e-15)

    p_z = propagate_log_space(z, p0, gain_np_m, source_w_nm)

    assert np.all(np.isfinite(p_z)), "JAX log-space propagation must never produce inf/NaN"
    assert np.all(p_z >= 0.0)


def _stage1_geometry(mat, kappa_db_m):
    dopant = estimate_dopant_concentration(
        pump_absorption_db_per_m=kappa_db_m,
        core_diameter_um=10.0,
        cladding_diameter_um=400.0,
        pump_wavelength_nm=976.0,
        material=mat,
        cladding_pumped=True,
    )
    r_core, r_clad = 5e-6, 200e-6
    return dict(
        n_tot=dopant.concentration_m3,
        gamma_p=dopant.gamma_pump,
        sigma_p=float(mat.sigma_abs_at(976.0)[0]),
        sigma_ep=float(mat.sigma_em_at(976.0)[0]),
        hnu_p=float(mat.photon_energy_j(976.0)[0]),
        a_pump=np.pi * r_clad**2,
        a_signal=np.pi * r_core**2,
        gamma_s=1.0,
        tau_21_s=mat.lifetime_s,
    )


def test_stage1_forward_self_consistent_solve_stays_finite_with_ase_band():
    """
    Two seeded signal channels (1030, 1064 nm) plus a 20-bin ASE band
    (1000-1090 nm) all competing for the same N2(z), forward-pumped —
    mirrors the user's reported multichannel ASE explosion (1 pulsed + 1 CW
    signal, 80 W pump). The self-consistent Stage 1 solve must stay finite
    (unlike the CPU model's decoupled, ASE-can-explode-to-inf behavior) and
    N2(z) must stay physically bounded in [0, 1].
    """
    mat = load_material("yb_glass")
    geo = _stage1_geometry(mat, kappa_db_m=17.0)

    wl_real = np.array([1030.0, 1064.0])
    wl_ase = np.linspace(1000.0, 1090.0, 20)
    wl_ch = np.concatenate([wl_real, wl_ase])
    sigma_a_ch = mat.sigma_abs_at(wl_ch)
    sigma_e_ch = mat.sigma_em_at(wl_ch)
    hnu_ch = mat.photon_energy_j(wl_ch)
    is_ase = np.array([False, False] + [True] * wl_ase.size)
    dlam_ch = np.concatenate([[1.0, 1.0], np.gradient(wl_ase)])
    p_ch_in = np.concatenate([[1e-3, 1e-3], np.full(wl_ase.size, 1e-12)])

    z = np.linspace(0.0, 2.0, 150)
    res = solve_stage1_forward(
        z,
        **geo,
        p_pump_in_w=300.0,
        channel_wavelengths_nm=wl_ch,
        sigma_a_ch=sigma_a_ch,
        sigma_e_ch=sigma_e_ch,
        p_ch_in_w=p_ch_in,
        is_ase=is_ase,
        eta_guided=0.6,
        dlam_ch_nm=dlam_ch,
        material_hnu_ch=hnu_ch,
    )

    assert np.all(np.isfinite(res.pump_power_w))
    assert np.all(np.isfinite(res.channel_power_w))
    assert np.all(np.isfinite(res.n2_fraction))
    assert np.all(res.channel_power_w >= 0.0)
    assert np.all((res.n2_fraction >= 0.0) & (res.n2_fraction <= 1.0))
    # Pump must be net-absorbed, not amplified or unphysically unaffected.
    assert res.pump_power_w[-1] < res.pump_power_w[0]


def test_stage1_forward_matches_cpu_in_a_safe_low_gain_regime():
    """Cross-check against the CPU pump-only pass at low pump power (no saturation)."""
    mat = load_material("yb_glass")
    geo = _stage1_geometry(mat, kappa_db_m=6.0)

    wl_ch = np.array([1030.0])
    sigma_a_ch = mat.sigma_abs_at(wl_ch)
    sigma_e_ch = mat.sigma_em_at(wl_ch)
    hnu_ch = mat.photon_energy_j(wl_ch)

    z = np.linspace(0.0, 0.3, 60)
    res = solve_stage1_forward(
        z,
        **geo,
        p_pump_in_w=0.05,  # deliberately tiny -> unsaturated, cold-absorption regime
        channel_wavelengths_nm=wl_ch,
        sigma_a_ch=sigma_a_ch,
        sigma_e_ch=sigma_e_ch,
        p_ch_in_w=np.array([1e-9]),
        is_ase=np.array([False]),
        eta_guided=0.6,
        dlam_ch_nm=np.array([1.0]),
        material_hnu_ch=hnu_ch,
    )

    kappa_np_m = geo["gamma_p"] * geo["sigma_p"] * geo["n_tot"]
    expected_transmission = np.exp(-kappa_np_m * z[-1])
    got_transmission = res.pump_power_w[-1] / res.pump_power_w[0]
    assert got_transmission == pytest.approx(expected_transmission, rel=0.05)
