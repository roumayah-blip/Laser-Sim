"""Multi-channel signal and pump support."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.materials import load_material
from laser_sim.physics.fiber_cpa import FiberCPAConfig, run_fiber_cpa
from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec
from laser_sim.pulses.multichannel import (
    PumpChannel,
    SignalChannel,
    build_multichannel_time_grid,
    collect_signal_wavelength_grids,
)


def test_collect_signal_wavelength_grids_nonoverlap():
    ch1 = SignalChannel(name="a", spec=ChirpedBurstSpec(center_wavelength_nm=1030.0), n_lambda=20)
    ch2 = SignalChannel(name="b", spec=ChirpedBurstSpec(center_wavelength_nm=1064.0), n_lambda=20)
    wl, slices = collect_signal_wavelength_grids([ch1, ch2])
    assert wl.size == 40
    assert len(slices) == 2
    assert slices[0][1].stop <= slices[1][1].start


def test_collect_signal_wavelength_grids_overlap_raises():
    ch1 = SignalChannel(
        name="a",
        spec=ChirpedBurstSpec(center_wavelength_nm=1030.0),
        lambda_half_window_nm=20.0,
    )
    ch2 = SignalChannel(
        name="b",
        spec=ChirpedBurstSpec(center_wavelength_nm=1040.0),
        lambda_half_window_nm=20.0,
    )
    with pytest.raises(ValueError, match="overlapping"):
        collect_signal_wavelength_grids([ch1, ch2])


def test_chronological_ordering_automatic():
    sc_early = SignalChannel(
        name="early",
        spec=ChirpedBurstSpec(
            burst_start_time_s=10e-6,
            burst_count=1,
            chirp_duration_s=1e-9,
        ),
    )
    sc_late = SignalChannel(
        name="late",
        spec=ChirpedBurstSpec(
            burst_start_time_s=50e-6,
            burst_count=1,
            chirp_duration_s=1e-9,
        ),
    )
    pc = PumpChannel(spec=PumpPulseSpec(cw=True, duration_s=100e-6, peak_power_w=10.0))
    t = build_multichannel_time_grid(signal_channels=[sc_early, sc_late], pump_channels=[pc])
    assert t.min() <= 10e-6
    assert t.max() >= 50e-6 + 1e-9


def _base_cfg(**kwargs):
    mat = load_material("yb_glass")
    defaults = dict(
        material=mat,
        fiber_length_m=0.3,
        n_z=12,
        yb_concentration_m3=5e25,
        include_ase=False,
    )
    defaults.update(kwargs)
    return FiberCPAConfig(**defaults)


def test_single_channel_backward_compat():
    from laser_sim.physics.fiber_cpa import _multichannel_mode

    spec = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        packet_energy_j=1e-6,
        burst_count=3,
        burst_start_time_s=50e-6,
    )
    pump = PumpPulseSpec(wavelength_nm=976.0, peak_power_w=20.0, cw=True, duration_s=2e-3)
    legacy = _base_cfg(signal=spec, pump=pump)
    assert not _multichannel_mode(legacy)
    out_legacy = run_fiber_cpa(legacy, backend="cpu")

    empty_lists = _base_cfg(signal=spec, pump=pump, signal_channels=[], pump_channels=[])
    assert not _multichannel_mode(empty_lists)
    out_empty = run_fiber_cpa(empty_lists, backend="cpu")
    assert out_empty.energy_packet_out_j == pytest.approx(out_legacy.energy_packet_out_j, rel=0.02)

    mc = _base_cfg(
        signal_channels=[SignalChannel(name="seed", spec=spec)],
        pump_channels=[PumpChannel(name="p976", spec=pump)],
    )
    assert _multichannel_mode(mc)
    out_mc = run_fiber_cpa(mc, backend="cpu")
    assert out_mc.energy_packet_in_j == pytest.approx(out_legacy.energy_packet_in_j, rel=0.25)


def test_single_channel_matches_legacy_tight():
    """One signal + one pump channel on identical grids must reproduce legacy exactly.

    Regression test: the multichannel path once used half the pump intensity
    (0.25 vs 0.5 trapezoid factor) and free-ran the RK4 population march
    through pump-only stretches, inflating packet-out energy 1.8× and pinning
    N₂/N_tot ≈ 1 (unphysical for 976 nm quasi-two-level Yb pumping).
    """
    spec = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        packet_energy_j=1e-6,
        burst_count=3,
        burst_start_time_s=50e-6,
    )
    pump = PumpPulseSpec(wavelength_nm=976.0, peak_power_w=20.0, cw=True, duration_s=2e-3)
    out_legacy = run_fiber_cpa(_base_cfg(signal=spec, pump=pump), backend="cpu")

    mc = _base_cfg(
        signal_channels=[SignalChannel(name="seed", spec=spec)],
        pump_channels=[PumpChannel(name="p976", spec=pump)],
        time_s=out_legacy.t_s,
        wavelength_nm=out_legacy.wavelength_nm,
    )
    out_mc = run_fiber_cpa(mc, backend="cpu")

    assert out_mc.energy_packet_in_j == pytest.approx(out_legacy.energy_packet_in_j, rel=0.02)
    assert out_mc.energy_packet_out_j == pytest.approx(out_legacy.energy_packet_out_j, rel=0.02)
    assert out_mc.pump_power_absorbed_fraction == pytest.approx(
        out_legacy.pump_power_absorbed_fraction, rel=0.02
    )
    peak_n2_mc = float(np.max(out_mc.n2_fraction))
    assert peak_n2_mc == pytest.approx(float(np.max(out_legacy.n2_fraction)), rel=0.02)
    # 976 nm quasi-2-level cap: N₂/N_tot ≤ σ_a/(σ_a+σ_e) ≈ 0.51 (+ margin).
    assert peak_n2_mc < 0.55


def test_single_channel_matches_legacy_tight_with_ase():
    """Same equivalence with ASE on — exercises the multi-pump ASE depletion path."""
    spec = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        packet_energy_j=1e-6,
        burst_count=3,
        burst_start_time_s=50e-6,
    )
    pump = PumpPulseSpec(wavelength_nm=976.0, peak_power_w=20.0, cw=True, duration_s=2e-3)
    out_legacy = run_fiber_cpa(_base_cfg(signal=spec, pump=pump, include_ase=True), backend="cpu")

    mc = _base_cfg(
        signal_channels=[SignalChannel(name="seed", spec=spec)],
        pump_channels=[PumpChannel(name="p976", spec=pump)],
        time_s=out_legacy.t_s,
        wavelength_nm=out_legacy.wavelength_nm,
        include_ase=True,
    )
    out_mc = run_fiber_cpa(mc, backend="cpu")

    assert out_mc.energy_packet_out_j == pytest.approx(out_legacy.energy_packet_out_j, rel=0.02)
    assert out_mc.energy_ase_out_j == pytest.approx(out_legacy.energy_ase_out_j, rel=0.02, abs=1e-15)


def test_two_signal_channels_independent():
    mat = load_material("yb_glass")
    e1 = 2e-6
    e2 = 3e-6
    sc1 = SignalChannel(
        name="1030",
        spec=ChirpedBurstSpec(center_wavelength_nm=1030.0, packet_energy_j=e1, burst_count=2),
    )
    sc2 = SignalChannel(
        name="1064",
        spec=ChirpedBurstSpec(center_wavelength_nm=1064.0, packet_energy_j=e2, burst_count=2),
    )
    pump = PumpChannel(spec=PumpPulseSpec(peak_power_w=30.0, cw=True, duration_s=1e-3))
    cfg = _base_cfg(signal_channels=[sc1, sc2], pump_channels=[pump])
    out = run_fiber_cpa(cfg, backend="cpu")
    assert len(out.signal_channels_info) == 2
    e_sum = sum(ch["energy_in_j"] for ch in out.signal_channels_info)
    assert e_sum == pytest.approx(out.energy_packet_in_j, rel=0.05)
    assert out.energy_packet_in_j == pytest.approx(e1 + e2, rel=0.30)


def test_cw_signal_cross_saturation():
    """A CW 1030 nm channel measurably boosts a co-propagating pulsed 1064 nm channel's gain.

    yb_glass transparency inversion (N2/Ntot where net gain=0) is ~6.6% at 1030 nm
    but only ~1.2% at 1064 nm (sigma_abs_at/sigma_em_at). In the typical operating
    range between those two thresholds, 1030 nm is still net-absorbing (photons
    promote ions N0->N2, acting like an auxiliary pump) while 1064 nm already has
    net gain. So a CW 1030 nm channel should *raise* N2 above what the 976 nm pump
    alone builds, increasing the 1064 nm channel's output energy versus a control
    run without the 1030 nm channel -- this is the shared-inversion reabsorption
    effect the multichannel physics is meant to capture.
    """
    from laser_sim.pulses.chirp import packet_duration_s
    from laser_sim.pulses.cw_average import packet_average_power_w

    pump = PumpPulseSpec(wavelength_nm=976.0, peak_power_w=20.0, cw=True, duration_s=1e-3)
    sig1064 = ChirpedBurstSpec(
        center_wavelength_nm=1064.0, packet_energy_j=1e-7, burst_count=2, burst_start_time_s=200e-6
    )

    control = _base_cfg(
        signal_channels=[SignalChannel(name="1064", spec=sig1064)],
        pump_channels=[PumpChannel(name="p976", spec=pump)],
        fiber_length_m=0.5,
        n_z=16,
    )
    out_ctrl = run_fiber_cpa(control, backend="cpu")
    e_ctrl = next(c for c in out_ctrl.signal_channels_info if c["name"] == "1064")["energy_out_j"]

    # 1 W CW average power at 1030 nm (well below the 20 W pump).
    dummy = ChirpedBurstSpec(center_wavelength_nm=1030.0, cw_average_power_mode=True)
    effective_period_s = max(packet_duration_s(dummy), dummy.burst_spacing_s, 10 * dummy.chirp_duration_s)
    cw1030 = ChirpedBurstSpec(
        center_wavelength_nm=1030.0,
        cw_average_power_mode=True,
        packet_energy_j=1.0 * effective_period_s,
        burst_start_time_s=0.0,
    )
    assert packet_average_power_w(cw1030) == pytest.approx(1.0, rel=1e-6)

    test_cfg = _base_cfg(
        signal_channels=[
            SignalChannel(name="1064", spec=sig1064),
            SignalChannel(name="1030cw", spec=cw1030, lambda_half_window_nm=4.0),
        ],
        pump_channels=[PumpChannel(name="p976", spec=pump)],
        fiber_length_m=0.5,
        n_z=16,
    )
    out_test = run_fiber_cpa(test_cfg, backend="cpu")
    e_test = next(c for c in out_test.signal_channels_info if c["name"] == "1064")["energy_out_j"]

    assert e_test > e_ctrl * 2.0


def test_dual_wavelength_pump():
    """Two pump wavelengths vs one channel at combined power (qualitative)."""
    mat = load_material("yb_glass")
    spec = ChirpedBurstSpec(packet_energy_j=0.5e-6, burst_count=1)
    p_total = 40.0
    dual = _base_cfg(
        signal_channels=[SignalChannel(spec=spec)],
        pump_channels=[
            PumpChannel(name="976", spec=PumpPulseSpec(wavelength_nm=976.0, peak_power_w=p_total / 2)),
            PumpChannel(name="915", spec=PumpPulseSpec(wavelength_nm=915.0, peak_power_w=p_total / 2)),
        ],
    )
    single = _base_cfg(
        signal_channels=[SignalChannel(spec=spec)],
        pump_channels=[
            PumpChannel(name="976", spec=PumpPulseSpec(wavelength_nm=976.0, peak_power_w=p_total)),
        ],
    )
    out_d = run_fiber_cpa(dual, backend="cpu")
    out_s = run_fiber_cpa(single, backend="cpu")
    assert len(out_d.pump_channels_info) == 2
    assert out_d.energy_packet_out_j > out_s.energy_packet_out_j * 0.5
