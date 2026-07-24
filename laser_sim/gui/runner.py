"""Build simulation config, run, and format errors for GUI layers."""

from __future__ import annotations

import traceback
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from laser_sim.constants import C0
from laser_sim.materials.base import (
    cladding_area_m2,
    core_area_m2,
    overlap_cladding_pump,
)

from laser_sim.calculators.dopant import (
    concentration_from_total_absorption_db,
    estimate_dopant_concentration,
)
from laser_sim.calculators.runtime import estimate_runtime, recommend_wavelength_grid
from laser_sim.materials import load_material
from laser_sim.materials.base import Material
from laser_sim.gui.backend_status import normalize_sim_backend
from laser_sim.physics.fiber_cpa import FiberCPAConfig, FiberCPAResult, run_fiber_cpa
from laser_sim.physics.progress import ProgressCallback
from laser_sim.physics.rep_rate_warmup import RepRateWarmupResult
from laser_sim.pulses.chirp import (
    ChirpedBurstSpec,
    PumpPulseSpec,
    build_cpa_time_grid,
    time_resolution_preset,
)


@dataclass
class SimInputs:
    material_key: str = "yb_glass"
    core_diameter_um: float = 10.0
    core_na: float = 0.06
    cladding_diameter_um: float = 400.0
    fiber_length_m: float = 2.0
    cladding_pumped: bool = True
    ignore_gamma_for_n: bool = False
    abs_mode_db_per_m: bool = True
    pump_absorption_db_per_m: float = 6.0
    total_absorption_db: float = 12.0
    pump_wavelength_nm: float = 976.0  # datasheet λ for N from κ (dB/m); not the sim pump unless matched
    simulation_pump_wavelength_nm: float | None = None  # physics pump λ; None → use pump_wavelength_nm
    yb_concentration_m3: float | None = None  # deprecated: use yb_concentration_override_m3
    yb_concentration_override_m3: float | None = None
    # When set, used as N_tot regardless of pump geometry or pump_absorption_db_per_m.
    # pump_absorption_db_per_m is informational only (diagnostics / consistency warn).
    pump_peak_power_w: float = 200.0
    pump_cw: bool = False
    pump_shape: str = "flat_top"
    pump_duration_s: float = 1e-3
    pump_start_s: float = 0.0
    signal_center_nm: float = 1030.0
    signal_bandwidth_nm: float = 8.0
    chirp_duration_s: float = 0.8e-9
    packet_energy_j: float = 10e-6
    burst_count: int = 5
    burst_spacing_s: float = 2.5e-9
    burst_start_s: float = 200e-6
    n_z: int = 120
    include_ase: bool = True
    rep_rate_mode: bool = False
    rep_rate_hz: float = 100e3
    n_periods: int = 40
    steady_state_tol: float = 0.02
    time_resolution: str = "standard"           # "low" | "standard" | "fine"
    steady_state_warmup: bool = True            # RP-style lumped pre-iteration for rep-rate CW
    export_diagnostics: bool = False
    cw_signal_average_power: bool = False
    backend: str = "cuda"
    passive_fiber_before_m: float = 0.0
    passive_fiber_after_m: float = 0.0
    n2_override_m2_per_w: float | None = None
    pulse_relative_powers: list[float] | None = None
    multichannel_mode: bool = False
    signal_channels_inputs: list | None = None  # list[dict] → SignalChannel
    pump_channels_inputs: list | None = None  # list[dict] → PumpChannel


@dataclass
class SimRunOutcome:
    ok: bool
    result: FiberCPAResult | None = None
    dopant: Any = None
    cw_average_result: FiberCPAResult | None = None
    diagnostics_report_path: str | None = None
    diagnostics_report_text: str | None = None
    error_type: str = ""
    error_message: str = ""
    traceback_text: str = ""
    runtime_estimate_s: float | None = None
    backend_requested: str = ""
    backend_used: str = ""
    taichi_arch: str | None = None
    wall_time_s: float | None = None
    b_integral: object | None = None
    packet_energy_target_j: float | None = None
    suggested_flat_weights: list[float] | None = None
    n_t: int | None = None
    n_lambda: int | None = None
    time_resolution: str | None = None
    steady_state_warmup_used: bool = False
    steady_state_warmup_iter: int | None = None
    steady_state_warmup_residual: float | None = None
    steady_state_warmup_converged: bool | None = None
    warmup_result: RepRateWarmupResult | None = None
    signal_spec: ChirpedBurstSpec | None = None
    pump_wavelength_datasheet_nm: float | None = None
    powerpoint_export_path: str | None = None


def _channels_from_inputs(inp: SimInputs) -> tuple[list, list]:
    """Build ``SignalChannel`` / ``PumpChannel`` lists from GUI dict payloads."""
    from laser_sim.pulses.chirp import ChirpedBurstSpec, PumpPulseSpec
    from laser_sim.pulses.multichannel import PumpChannel, SignalChannel

    sig_chs: list = []
    for d in inp.signal_channels_inputs or []:
        spec_d = dict(d.get("spec", {}))
        spec = ChirpedBurstSpec(**spec_d)
        sig_chs.append(
            SignalChannel(
                name=str(d.get("name", "signal")),
                spec=spec,
                n_lambda=int(d.get("n_lambda", 32)),
                lambda_half_window_nm=float(d.get("lambda_half_window_nm", 4.0)),
                enabled=bool(d.get("enabled", True)),
            )
        )
    pump_chs: list = []
    for d in inp.pump_channels_inputs or []:
        pspec_d = dict(d.get("spec", {}))
        pspec = PumpPulseSpec(**pspec_d)
        pump_chs.append(
            PumpChannel(
                name=str(d.get("name", "pump")),
                spec=pspec,
                cladding_pumped=bool(d.get("cladding_pumped", inp.cladding_pumped)),
                backward_fraction=float(d.get("backward_fraction", 0.0)),
                enabled=bool(d.get("enabled", True)),
            )
        )
    return sig_chs, pump_chs


def simulation_pump_wavelength_nm(inp: SimInputs) -> float:
    """Pump wavelength used in the CPA simulation (validated against material σ table)."""
    if inp.simulation_pump_wavelength_nm is not None:
        return float(inp.simulation_pump_wavelength_nm)
    return float(inp.pump_wavelength_nm)


def _resolve_dopant(inp: SimInputs, material: Material):
    if inp.abs_mode_db_per_m:
        return estimate_dopant_concentration(
            pump_absorption_db_per_m=inp.pump_absorption_db_per_m,
            core_diameter_um=inp.core_diameter_um,
            cladding_diameter_um=inp.cladding_diameter_um,
            pump_wavelength_nm=inp.pump_wavelength_nm,
            material=material,
            cladding_pumped=inp.cladding_pumped,
            ignore_overlap_for_concentration=inp.ignore_gamma_for_n,
        )
    return concentration_from_total_absorption_db(
        total_absorption_db=inp.total_absorption_db,
        fiber_length_m=inp.fiber_length_m,
        core_diameter_um=inp.core_diameter_um,
        cladding_diameter_um=inp.cladding_diameter_um,
        pump_wavelength_nm=inp.pump_wavelength_nm,
        material=material,
        cladding_pumped=inp.cladding_pumped,
        ignore_overlap_for_concentration=inp.ignore_gamma_for_n,
    )


def run_simulation(
    inp: SimInputs,
    progress_callback: ProgressCallback | None = None,
) -> SimRunOutcome:
    try:
        import time

        backend_used = normalize_sim_backend(inp.backend)
        material = load_material(inp.material_key)
        sim_pump_nm = simulation_pump_wavelength_nm(inp)
        from laser_sim.materials.base import require_pump_cross_sections

        require_pump_cross_sections(material, sim_pump_nm)
        n_override = inp.yb_concentration_override_m3
        if n_override is None and inp.yb_concentration_m3 is not None:
            n_override = inp.yb_concentration_m3

        if n_override is not None:
            n_yb = n_override
            dopant = _resolve_dopant(inp, material)
        else:
            dopant = _resolve_dopant(inp, material)
            n_yb = dopant.concentration_for_rates_m3

        if inp.rep_rate_mode and not inp.pump_cw:
            raise ValueError("Rep-rate steady-state mode requires CW pump.")

        pump_shape = "cw" if inp.pump_cw else inp.pump_shape
        pump_spec = PumpPulseSpec(
            wavelength_nm=sim_pump_nm,
            peak_power_w=inp.pump_peak_power_w,
            duration_s=inp.pump_duration_s,
            shape=pump_shape,
            cw=inp.pump_cw,
            start_time_s=inp.pump_start_s,
        )

        prp: tuple[float, ...] | None = None
        if inp.pulse_relative_powers is not None:
            prp = tuple(float(x) for x in inp.pulse_relative_powers)

        burst_start_s = inp.burst_start_s
        if inp.rep_rate_mode and inp.pump_cw:
            burst_start_s = 1.0 / inp.rep_rate_hz

        use_warmup = bool(inp.rep_rate_mode and inp.pump_cw and inp.steady_state_warmup)
        # When warmup is active, we still tag the run as rep-rate (so diagnostics and
        # downstream consumers know rep_rate_hz) but only simulate ONE period
        # time-resolved; the steady-state transient is handled by the lumped warmup.
        sig_rep_rate_mode = inp.rep_rate_mode
        sig_n_periods = 1 if use_warmup else inp.n_periods

        sig_spec = ChirpedBurstSpec(
            center_wavelength_nm=inp.signal_center_nm,
            bandwidth_nm=inp.signal_bandwidth_nm,
            chirp_duration_s=inp.chirp_duration_s,
            packet_energy_j=inp.packet_energy_j,
            burst_count=inp.burst_count,
            burst_spacing_s=inp.burst_spacing_s,
            burst_start_time_s=burst_start_s,
            rep_rate_mode=sig_rep_rate_mode,
            rep_rate_hz=inp.rep_rate_hz,
            n_periods=sig_n_periods,
            steady_state_tol=inp.steady_state_tol,
            pulse_relative_powers=prp,
        )

        lam_min, lam_max, n_lam = recommend_wavelength_grid(
            center_nm=sig_spec.center_wavelength_nm,
            bandwidth_nm=sig_spec.bandwidth_nm,
            points_per_nm=2.0,
        )
        wl = np.linspace(lam_min, lam_max, max(n_lam, 48))
        pump_window = inp.pump_duration_s
        if inp.rep_rate_mode:
            from laser_sim.pulses.chirp import packet_duration_s, rep_period_s

            t_rep_value = rep_period_s(sig_spec)
            pkt_dur = packet_duration_s(sig_spec)
            if use_warmup:
                # Warmup already provides the steady-state inversion, so the
                # time-resolved pass only needs a short pre-roll + one period.
                pre_roll = max(burst_start_s, 2 * t_rep_value)
                pump_window = pre_roll + t_rep_value + pkt_dur
            else:
                pump_window = max(
                    inp.pump_duration_s,
                    burst_start_s
                    + sig_spec.n_periods * t_rep_value
                    + pkt_dur,
                )

        tr_preset = time_resolution_preset(inp.time_resolution)
        # When warmup is active, treat the single time-resolved period like a normal
        # CPA burst (non-rep-rate grid) so the coarse pump grid scales with the
        # ~10 µs window instead of being inflated to 400 points across the period.
        grid_spec = sig_spec
        if use_warmup:
            grid_spec = replace(sig_spec, rep_rate_mode=False)
        t = build_cpa_time_grid(
            pump_duration_s=pump_window,
            spec=grid_spec,
            pump_cw=inp.pump_cw,
            points_per_chirped_pulse=tr_preset["points_per_chirped_pulse"],
            points_per_burst_spacing=tr_preset["points_per_burst_spacing"],
            points_pump_coarse=tr_preset["points_pump_coarse"],
        )

        # Geometry shared by warmup + main run
        _r_core = inp.core_diameter_um / 2 * 1e-6
        _r_clad = inp.cladding_diameter_um / 2 * 1e-6
        _sigma_p = float(material.sigma_abs_at(sim_pump_nm)[0])
        _sigma_ep = float(material.sigma_em_at(sim_pump_nm)[0])
        _hnu_p = 6.626e-34 * 3e8 / (sim_pump_nm * 1e-9)
        _hnu_s = 6.626e-34 * 3e8 / (inp.signal_center_nm * 1e-9)
        if inp.cladding_pumped:
            _gamma_p = (_r_core / _r_clad) ** 2 if _r_clad > _r_core else 1.0
            _a_pump = np.pi * _r_clad**2
        else:
            _gamma_p = 1.0
            _a_pump = np.pi * _r_core**2
        _a_core = np.pi * _r_core**2
        _tau21 = material.lifetime_s

        initial_n2_fraction = 0.0
        initial_n2_fraction_z = None
        warmup_iter = None
        warmup_residual = None
        warmup_converged = None
        warmup_notes = ""
        warmup_result: RepRateWarmupResult | None = None

        if inp.rep_rate_mode and inp.pump_cw and not use_warmup:
            from laser_sim.physics.four_level import pump_rate_per_ion, steady_state_n2_fraction_pump

            _i_p = inp.pump_peak_power_w / _a_pump
            _w_abs, _w_esa = pump_rate_per_ion(
                _i_p, sigma_p=_sigma_p, sigma_ep=_sigma_ep, hnu_p=_hnu_p
            )
            initial_n2_fraction = steady_state_n2_fraction_pump(
                _w_abs, _w_esa, _tau21
            )

        if use_warmup:
            from laser_sim.physics.rep_rate_warmup import (
                cw_pump_power_along_z,
                solve_rep_rate_steady_state_inversion,
            )
            from laser_sim.physics.modes import signal_overlap_gamma

            _gamma_s_warm, _a_sig_warm, _ = signal_overlap_gamma(
                _r_core, inp.signal_center_nm, inp.core_na
            )
            _sigma_a_s = float(material.sigma_abs_at(inp.signal_center_nm)[0])
            _sigma_e_s = float(material.sigma_em_at(inp.signal_center_nm)[0])

            z_warm = np.linspace(0.0, inp.fiber_length_m, max(inp.n_z, 10))
            # Two passes: first with N0=N_tot, then refine pump with current N2 estimate.
            p_pump_z = cw_pump_power_along_z(
                z_m=z_warm,
                pump_power_in_w=inp.pump_peak_power_w,
                a_pump_m2=_a_pump,
                sigma_p=_sigma_p,
                gamma_p=_gamma_p,
                n_tot=n_yb,
            )
            warm = solve_rep_rate_steady_state_inversion(
                z_m=z_warm,
                pump_power_z_w=p_pump_z,
                a_pump_m2=_a_pump,
                sigma_p=_sigma_p,
                gamma_p=_gamma_p,
                hnu_p=_hnu_p,
                sigma_ep=_sigma_ep,
                sigma_a_sig=_sigma_a_s,
                sigma_e_sig=_sigma_e_s,
                gamma_s=_gamma_s_warm,
                a_signal_m2=_a_sig_warm,
                hnu_s=_hnu_s,
                packet_energy_j=inp.packet_energy_j,
                rep_rate_hz=inp.rep_rate_hz,
                tau_21_s=_tau21,
                n_tot=n_yb,
                progress_callback=progress_callback,
                progress_base=0.02,
                progress_span=0.20,
            )
            # Refine pump with steady-state N2 estimate then re-solve once.
            p_pump_z = cw_pump_power_along_z(
                z_m=z_warm,
                pump_power_in_w=inp.pump_peak_power_w,
                a_pump_m2=_a_pump,
                sigma_p=_sigma_p,
                gamma_p=_gamma_p,
                n_tot=n_yb,
                n2_estimate_z=warm.n2_pre_packet,
            )
            warm = solve_rep_rate_steady_state_inversion(
                z_m=z_warm,
                pump_power_z_w=p_pump_z,
                a_pump_m2=_a_pump,
                sigma_p=_sigma_p,
                gamma_p=_gamma_p,
                hnu_p=_hnu_p,
                sigma_ep=_sigma_ep,
                sigma_a_sig=_sigma_a_s,
                sigma_e_sig=_sigma_e_s,
                gamma_s=_gamma_s_warm,
                a_signal_m2=_a_sig_warm,
                hnu_s=_hnu_s,
                packet_energy_j=inp.packet_energy_j,
                rep_rate_hz=inp.rep_rate_hz,
                tau_21_s=_tau21,
                n_tot=n_yb,
                initial_n2_fraction=float(np.mean(warm.n2_pre_packet) / n_yb),
                progress_callback=progress_callback,
                progress_base=0.22,
                progress_span=0.10,
            )
            initial_n2_fraction_z = (warm.n2_pre_packet / n_yb).astype(np.float64)
            warmup_result = warm
            warmup_iter = warm.n_iter
            warmup_residual = warm.residual
            warmup_converged = warm.converged
            warmup_notes = warm.notes

        sig_chs, pump_chs = (
            _channels_from_inputs(inp) if inp.multichannel_mode else ([], [])
        )
        cfg = FiberCPAConfig(
            material=material,
            fiber_length_m=inp.fiber_length_m,
            core_diameter_um=inp.core_diameter_um,
            core_na=inp.core_na,
            cladding_diameter_um=inp.cladding_diameter_um,
            cladding_pumped=inp.cladding_pumped,
            yb_concentration_m3=n_yb,
            pump_absorption_db_per_m=inp.pump_absorption_db_per_m,
            n_z=inp.n_z,
            time_s=t,
            wavelength_nm=wl,
            signal=sig_spec,
            pump=pump_spec,
            include_ase=inp.include_ase,
            initial_n2_fraction=initial_n2_fraction,
            initial_n2_fraction_z=initial_n2_fraction_z,
            signal_channels=sig_chs,
            pump_channels=pump_chs,
        )

        t0 = time.perf_counter()

        def _run_main() -> FiberCPAResult:
            return run_fiber_cpa(cfg, backend=backend_used, progress_callback=progress_callback)

        taichi_arch = None
        backend_effective = backend_used
        try:
            result = _run_main()
        except RuntimeError as exc:
            err = str(exc).lower()
            cuda_dead = "invalid_context" in err or "invalid device context" in err
            worker_failed = (
                "taichi gpu worker" in err
                or "exited without writing" in err
                or "exited with code" in err
            )
            if backend_used == "taichi" and (cuda_dead or worker_failed):
                from laser_sim.physics import taichi_kernels as tk

                tk.abandon_taichi_runtime()
                warnings.warn(
                    f"GPU run failed ({exc}); rerunning on CPU. "
                    "Reduce grid size if this was an out-of-memory kill.",
                    stacklevel=2,
                )
                backend_effective = "cpu"
                result = run_fiber_cpa(cfg, backend="cpu", progress_callback=progress_callback)
            else:
                raise

        cw_result = None
        if inp.cw_signal_average_power:

            def _cw_progress(frac: float, msg: str) -> None:
                if progress_callback is not None:
                    progress_callback(0.5 + 0.5 * frac, f"CW reference: {msg}")

            sig_cw = replace(
                sig_spec,
                cw_average_power_mode=True,
                rep_rate_mode=False,
            )
            cfg_cw = replace(cfg, signal=sig_cw)
            cw_result = run_fiber_cpa(
                cfg_cw,
                backend=backend_used,
                progress_callback=_cw_progress,
            )
            from laser_sim.pulses.cw_average import packet_average_power_w

            p_avg = packet_average_power_w(sig_spec)
            hnu_s = 6.626e-34 * 3e8 / (sig_spec.center_wavelength_nm * 1e-9)
            sa = float(material.sigma_abs_at(sig_spec.center_wavelength_nm)[0])
            se = float(material.sigma_em_at(sig_spec.center_wavelength_nm)[0])
            a_sig = np.pi * (inp.core_diameter_um * 0.5e-6) ** 2
            p_sat = hnu_s * a_sig / ((sa + se) * material.lifetime_s)
            if p_avg > 10 * p_sat:
                warnings.warn(
                    f"CW reference power ({p_avg:.1f} W) >> P_sat ({p_sat:.3f} W). "
                    "The CW reference run is deeply saturated and not meaningful. "
                    "It shows signal saturation, not the small-signal gain. "
                    "Reduce pulse energy or increase burst_spacing to get a useful CW reference.",
                    stacklevel=2,
                )

        from laser_sim.physics.b_integral import N2_SILICA_M2_PER_W, compute_b_integral
        from laser_sim.pulses.chirp import chirp_sigma_t_s

        sigma_t = chirp_sigma_t_s(sig_spec)
        e_per_pulse_flat = sig_spec.packet_energy_j / max(sig_spec.burst_count, 1)
        p_peak_in = float(e_per_pulse_flat / (np.sqrt(2.0 * np.pi) * sigma_t))
        gain_ratio = result.energy_packet_out_j / max(result.energy_packet_in_j, 1e-30)
        p_peak_out = p_peak_in * gain_ratio
        a_eff = result.a_signal_m2 if result.a_signal_m2 > 0 else (
            result.signal_mode_area_m2
            if result.signal_mode_area_m2 > 0
            else float(np.pi * (inp.core_diameter_um * 0.5e-6) ** 2)
        )
        n2 = inp.n2_override_m2_per_w if inp.n2_override_m2_per_w is not None else N2_SILICA_M2_PER_W
        b_integral = compute_b_integral(
            wavelength_m=inp.signal_center_nm * 1e-9,
            a_eff_m2=a_eff,
            p_peak_in_w=p_peak_in,
            p_peak_out_w=p_peak_out,
            l_active_m=inp.fiber_length_m,
            l_passive_before_m=inp.passive_fiber_before_m,
            l_passive_after_m=inp.passive_fiber_after_m,
            n2_m2_per_w=n2,
        )

        wall_s = time.perf_counter() - t0
        if backend_effective == "taichi":
            from laser_sim.physics import taichi_kernels as tk

            taichi_arch = tk.active_arch_name()
        elif backend_effective == "cpu" and backend_used == "taichi":
            taichi_arch = "CPU fallback"

        diagnostics_path = None
        diagnostics_text = None
        if inp.export_diagnostics:
            from laser_sim.physics.diagnostics import (
                build_diagnostics_report,
                write_diagnostics_report,
            )
            from laser_sim.physics.progress import emit_progress

            emit_progress(progress_callback, 0.96, "Building diagnostics report…")

            r_core = 0.5 * inp.core_diameter_um * 1e-6
            r_clad = 0.5 * inp.cladding_diameter_um * 1e-6
            if inp.cladding_pumped:
                gamma_p = overlap_cladding_pump(r_core, r_clad)
                a_pump = cladding_area_m2(r_clad)
            else:
                gamma_p = 1.0
                a_pump = core_area_m2(r_core)
            a_core = core_area_m2(r_core)
            dz = inp.fiber_length_m / max(inp.n_z - 1, 1)
            v_g = C0 / material.n_group
            dt_travel = dz / v_g
            label = "pulsed"
            if sig_spec.cw_average_power_mode:
                label = "cw_average"
            diagnostics_text = build_diagnostics_report(
                cfg,
                result,
                material,
                dopant,
                gamma_p=gamma_p,
                gamma_s=result.gamma_signal,
                a_core=a_core,
                a_pump=a_pump,
                dz=dz,
                dt_travel=dt_travel,
                run_label=label,
                cw_reference=cw_result,
                b_integral=b_integral,
                l_passive_before_m=inp.passive_fiber_before_m,
                l_passive_after_m=inp.passive_fiber_after_m,
            )
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            out_dir = Path(__file__).resolve().parents[2] / "diagnostics_output"
            diagnostics_path = str(
                write_diagnostics_report(
                    out_dir / f"cpa_diagnostics_{stamp}.txt",
                    diagnostics_text,
                )
            )

        rt = estimate_runtime(
            fiber_length_m=inp.fiber_length_m,
            n_z=inp.n_z,
            n_t=t.size,
            n_lambda=wl.size,
            n_burst_pulses=inp.burst_count,
            backend=inp.backend,
        )

        suggested_flat_weights = None
        if (
            result.pulse_energies_in_j is not None
            and result.pulse_energies_out_j is not None
            and len(result.pulse_energies_in_j) > 1
        ):
            from laser_sim.physics.equalization import estimate_flat_packet_weights

            try:
                w = estimate_flat_packet_weights(
                    result.pulse_energies_in_j,
                    result.pulse_energies_out_j,
                )
                suggested_flat_weights = w.tolist()
            except Exception:
                suggested_flat_weights = None

        return SimRunOutcome(
            ok=True,
            result=result,
            dopant=dopant,
            cw_average_result=cw_result,
            diagnostics_report_path=diagnostics_path,
            diagnostics_report_text=diagnostics_text,
            runtime_estimate_s=rt.estimated_seconds,
            backend_requested=inp.backend,
            backend_used=backend_effective,
            taichi_arch=taichi_arch,
            wall_time_s=wall_s,
            b_integral=b_integral,
            packet_energy_target_j=inp.packet_energy_j,
            suggested_flat_weights=suggested_flat_weights,
            n_t=int(t.size),
            n_lambda=int(wl.size),
            time_resolution=inp.time_resolution,
            steady_state_warmup_used=use_warmup,
            steady_state_warmup_iter=warmup_iter,
            steady_state_warmup_residual=warmup_residual,
            steady_state_warmup_converged=warmup_converged,
            warmup_result=warmup_result,
            signal_spec=sig_spec,
            pump_wavelength_datasheet_nm=inp.pump_wavelength_nm,
        )
    except Exception as exc:
        return SimRunOutcome(
            ok=False,
            error_type=type(exc).__name__,
            error_message=str(exc),
            traceback_text=traceback.format_exc(),
        )


def format_energy_summary(out: SimRunOutcome) -> str:
    if not out.ok or out.result is None:
        return ""
    r = out.result
    g_pkt = r.energy_packet_out_j / max(r.energy_packet_in_j, 1e-30)
    g_pls = r.energy_pulse_out_j / max(r.energy_pulse_in_j, 1e-30)
    e_exp = r.energy_packet_expected_j
    target_uj = (
        out.packet_energy_target_j * 1e6
        if out.packet_energy_target_j is not None
        else e_exp * 1e6
    )
    lines = [
        f"Packet gain (energy): {g_pkt:.4f}",
        f"Single-pulse gain (1/e² window, 1st pulse): {g_pls:.4f}",
        f"Packet energy in:  {r.energy_packet_in_j*1e6:.3f} µJ  (target: {target_uj:.3f} µJ)",
        f"Packet energy expected: {e_exp*1e6:.3f} µJ",
        f"Single-pulse energy in (1/e²): {r.energy_pulse_in_j*1e6:.4f} µJ",
        f"Single-pulse energy out (1/e²): {r.energy_pulse_out_j*1e6:.4f} µJ",
        f"Pump absorbed (energy fraction): {r.pump_power_absorbed_fraction*100:.1f}%",
        f"Pump in / out (mJ): {r.energy_pump_in_j*1e3:.4f} / {r.energy_pump_out_j*1e3:.4f}",
        f"Packet in / out (µJ): {r.energy_packet_in_j*1e6:.4f} / {r.energy_packet_out_j*1e6:.4f}",
    ]
    a_sig = r.a_signal_m2 if r.a_signal_m2 > 0 else r.signal_mode_area_m2
    if a_sig > 0:
        mfd_um = float(np.sqrt(4 * a_sig / np.pi) * 1e6)
        lines.append(
            f"Mode area A_eff: {a_sig * 1e12:.2f} µm²  (MFD {mfd_um:.2f} µm)"
        )
    lines.append(f"ASE out (µJ): {r.energy_ase_out_j*1e6:.4f}")
    if r.ase_fraction_of_emission > 0 or r.energy_ase_out_j > 0:
        lines.append(
            f"ASE fraction of emission: {r.ase_fraction_of_emission*100:.2f}%  "
            f"(unguided spont {r.energy_unguided_spont_j*1e6:.4f} µJ)"
        )
    bal = "OK" if r.energy_balance_ok else "OVER PUMP"
    lines.append(
        f"Energy budget ({bal}): pump abs {max(r.energy_pump_in_j - r.energy_pump_out_j, 0)*1e3:.4f} mJ, "
        f"emit {((r.energy_packet_out_j - r.energy_packet_in_j) + r.energy_ase_out_j + r.energy_unguided_spont_j)*1e3:.4f} mJ"
    )
    if out.b_integral is not None:
        b = out.b_integral
        lines.append(
            f"B-integral: {b.b_passive_before_rad:.3f} (pre) + {b.b_active_rad:.3f} (active)"
            f" + {b.b_passive_after_rad:.3f} (post) = {b.b_total_rad:.3f} rad  [{b.severity}]"
        )
        lines.append(f"L_NL at output: {b.l_nl_passive_after_m * 100:.1f} cm")
    if out.backend_used:
        arch = f", Taichi arch={out.taichi_arch}" if out.taichi_arch else ""
        lines.append(f"Backend: {out.backend_requested} → {out.backend_used}{arch}")
    if out.wall_time_s is not None:
        lines.append(f"Wall time: {out.wall_time_s:.2f} s")
    if out.runtime_estimate_s is not None:
        lines.append(f"Est. runtime ({out.backend_requested}): {out.runtime_estimate_s:.2f} s")
    if r.rep_rate_hz is not None:
        lines.append(
            f"Rep rate: {r.rep_rate_hz/1e3:.2f} kHz, periods={r.n_periods_simulated}, "
            f"steady-state={'yes' if r.steady_state_reached else 'no'} (metric={r.steady_state_metric:.4f})"
        )
    if out.steady_state_warmup_used:
        conv = "converged" if out.steady_state_warmup_converged else "did NOT converge"
        lines.append(
            f"Steady-state warmup: {conv} in {out.steady_state_warmup_iter} iter "
            f"(residual={out.steady_state_warmup_residual:.2e}); 1 period time-resolved"
        )
        w = out.warmup_result
        if w is not None and w.e_out_per_iter_j.size:
            e0 = float(w.e_out_per_iter_j[0]) * 1e6
            ef = float(w.e_out_per_iter_j[-1]) * 1e6
            g0 = float(w.gain_per_iter[0])
            gf = float(w.gain_per_iter[-1])
            lines.append(
                f"Warmup packet energy (lumped): {e0:.4f} → {ef:.4f} µJ  "
                f"(gain {g0:.3f} → {gf:.3f})"
            )
    if out.n_t is not None:
        lines.append(
            f"Grid: n_t={out.n_t}, n_lambda={out.n_lambda}, "
            f"time resolution={out.time_resolution or 'standard'}"
        )
    if out.result is not None and out.pump_wavelength_datasheet_nm is not None:
        ds = out.pump_wavelength_datasheet_nm
        sim = out.result.pump_wavelength_nm
        if abs(ds - sim) > 0.5:
            lines.append(
                f"Pump λ: simulation {sim:.2f} nm, datasheet (N calc) {ds:.2f} nm"
            )
        else:
            lines.append(f"Pump λ: {sim:.2f} nm")
    if out.suggested_flat_weights is not None:
        w = out.suggested_flat_weights
        w_str = ", ".join(f"{x:.3f}" for x in w)
        lines.append(f"Suggested flat-packet weights: [{w_str}]")
        lines.append(
            f"  (peak/mean ratio: {max(w)/min(w):.2f}x; copy to pulse weights for next run)"
        )
    return "\n".join(lines)


def format_cw_reference_summary(out: SimRunOutcome) -> str:
    if out.cw_average_result is None:
        return ""
    from laser_sim.viz.plot_limits import integrated_power_vs_time

    r = out.cw_average_result
    g_e = r.energy_packet_out_j / max(r.energy_packet_in_j, 1e-30)
    p_in_t = integrated_power_vs_time(r.signal_fwd_w_nm[0], r.wavelength_nm)
    p_out_t = integrated_power_vs_time(r.signal_fwd_w_nm[-1], r.wavelength_nm)
    it = int(np.argmax(p_in_t))
    g_p = float(p_out_t[it] / max(p_in_t[it], 1e-30))
    p_avg = float(p_in_t[it])
    return (
        f"CW avg-power reference: P_in≈{p_avg:.4g} W, "
        f"P(L)/P(0)={g_p:.4f} at peak time; "
        f"packet-window energy gain={g_e:.4f} "
        f"({r.energy_packet_in_j*1e6:.4f}→{r.energy_packet_out_j*1e6:.4f} µJ)"
    )
