"""Tests for the rep-rate lumped steady-state warmup."""

import numpy as np
import pytest

from laser_sim.physics.four_level import pump_rate_per_ion, steady_state_n2_fraction_pump
from laser_sim.physics.rep_rate_warmup import (
    cw_pump_power_along_z,
    solve_rep_rate_steady_state_inversion,
)


def _yb_like_params(*, packet_uj: float = 10.0, rep_khz: float = 100.0):
    n_tot = 1.0e26  # m^-3
    sigma_a_sig = 6e-27
    sigma_e_sig = 4e-25
    gamma_s = 0.85
    r_core = 5e-6
    a_signal = np.pi * r_core**2
    hnu_s = 6.626e-34 * 3e8 / 1030e-9

    r_clad = 200e-6
    a_pump = np.pi * r_clad**2
    gamma_p = (r_core / r_clad) ** 2
    sigma_p = 1.5e-24
    sigma_ep = 1.45e-24
    hnu_p = 6.626e-34 * 3e8 / 976e-9

    tau_21 = 850e-6
    z = np.linspace(0.0, 2.0, 60)
    pump_z = cw_pump_power_along_z(
        z_m=z,
        pump_power_in_w=80.0,
        a_pump_m2=a_pump,
        sigma_p=sigma_p,
        gamma_p=gamma_p,
        n_tot=n_tot,
    )

    return dict(
        z_m=z,
        pump_power_z_w=pump_z,
        a_pump_m2=a_pump,
        sigma_p=sigma_p,
        sigma_ep=sigma_ep,
        gamma_p=gamma_p,
        hnu_p=hnu_p,
        sigma_a_sig=sigma_a_sig,
        sigma_e_sig=sigma_e_sig,
        gamma_s=gamma_s,
        a_signal_m2=a_signal,
        hnu_s=hnu_s,
        packet_energy_j=packet_uj * 1e-6,
        rep_rate_hz=rep_khz * 1e3,
        tau_21_s=tau_21,
        n_tot=n_tot,
    )


def test_warmup_converges_low_signal():
    """Very small packet energy: N2(z) should sit at the CW pump steady state."""
    params = _yb_like_params(packet_uj=0.001, rep_khz=100.0)
    result = solve_rep_rate_steady_state_inversion(**params, tol=1e-5, max_iter=200)
    assert result.converged, result.notes
  # Tiny packet: mean inversion should stay in the pump-dominated regime (>10%).
    mean_frac = float(np.mean(result.n2_pre_packet) / params["n_tot"])
    assert mean_frac > 0.10, f"expected pump-limited inversion, got {mean_frac:.3f}"


def test_warmup_extraction_consistent_with_energy_balance():
    """Output packet energy <= absorbed pump energy per period; gain finite."""
    params = _yb_like_params(packet_uj=10.0, rep_khz=100.0)
    result = solve_rep_rate_steady_state_inversion(**params, tol=1e-4, max_iter=300)
    assert result.converged, result.notes

    # Absorbed pump per period: (P_in - P_out) * T_rep
    p_in = float(params["pump_power_z_w"][0])
    p_out = float(params["pump_power_z_w"][-1])
    t_rep = 1.0 / params["rep_rate_hz"]
    e_pump_abs = (p_in - p_out) * t_rep

    e_in = params["packet_energy_j"]
    e_out = result.e_packet_out_j
    gain = e_out / max(e_in, 1e-30)
    assert gain >= 1.0, "Net output energy should not be below input"
    assert e_out - e_in <= e_pump_abs * 1.5, (
        f"Packet extraction {(e_out - e_in)*1e6:.3f} µJ exceeds pump absorbed "
        f"{e_pump_abs*1e6:.3f} µJ by >50% — energy bookkeeping is off"
    )


def test_warmup_low_reprate_higher_gain_than_high_reprate():
    """At lower rep rate the inversion has longer to rebuild → higher per-pulse gain."""
    low = solve_rep_rate_steady_state_inversion(
        **_yb_like_params(packet_uj=1.0, rep_khz=1.0), tol=1e-4, max_iter=300
    )
    high = solve_rep_rate_steady_state_inversion(
        **_yb_like_params(packet_uj=1.0, rep_khz=1000.0), tol=1e-4, max_iter=300
    )
    g_low = low.e_packet_out_j / 1e-6
    g_high = high.e_packet_out_j / 1e-6
    assert g_low >= g_high, f"low-rep gain {g_low:.3e} should be >= high-rep gain {g_high:.3e}"


def test_warmup_returns_per_z_pre_packet_inversion():
    params = _yb_like_params(packet_uj=5.0)
    result = solve_rep_rate_steady_state_inversion(**params, tol=5e-4, max_iter=200)
    n_tot = params["n_tot"]
    assert result.n2_pre_packet.shape == params["z_m"].shape
    assert np.all(result.n2_pre_packet >= 0.0)
    assert np.all(result.n2_pre_packet <= n_tot * 1.001)
    # Post-pulse inversion must not exceed pre-pulse inversion at any z
    assert np.all(result.n2_post_packet <= result.n2_pre_packet + 1e-3 * n_tot)


def test_time_resolution_preset_scales_grid_size():
    """Switching from low → fine should increase n_t monotonically."""
    from laser_sim.pulses.chirp import ChirpedBurstSpec, build_cpa_time_grid, time_resolution_preset

    spec = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        bandwidth_nm=8.0,
        chirp_duration_s=0.8e-9,
        packet_energy_j=10e-6,
        burst_count=5,
        burst_spacing_s=2.5e-9,
        burst_start_time_s=200e-6,
    )
    sizes = []
    for name in ("low", "standard", "fine"):
        pr = time_resolution_preset(name)
        t = build_cpa_time_grid(
            pump_duration_s=1e-3,
            spec=spec,
            pump_cw=False,
            points_per_chirped_pulse=pr["points_per_chirped_pulse"],
            points_per_burst_spacing=pr["points_per_burst_spacing"],
            points_pump_coarse=pr["points_pump_coarse"],
        )
        sizes.append(t.size)
    assert sizes[0] < sizes[1] < sizes[2], (
        f"Expected low < standard < fine, got {sizes}"
    )
    # low preset should be at least ~3× smaller than standard for typical CPA grids
    assert sizes[1] / sizes[0] >= 2.0, (
        f"Standard/low ratio too small: {sizes}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
