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
from laser_sim.pulses.chirp import (
    ChirpedBurstSpec,
    PumpPulseSpec,
    build_cpa_time_grid,
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
    pump_wavelength_nm: float = 976.0
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
    energy_per_pulse_j: float = 1e-6
    burst_count: int = 5
    burst_spacing_s: float = 2.5e-9
    burst_start_s: float = 200e-6
    n_z: int = 120
    include_ase: bool = True
    rep_rate_mode: bool = False
    rep_rate_hz: float = 100e3
    n_periods: int = 40
    steady_state_tol: float = 0.02
    export_diagnostics: bool = False
    cw_signal_average_power: bool = False
    backend: str = "cuda"
    passive_fiber_before_m: float = 0.0
    passive_fiber_after_m: float = 0.0
    n2_override_m2_per_w: float | None = None
    pulse_relative_powers: list[float] | None = None


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
            wavelength_nm=inp.pump_wavelength_nm,
            peak_power_w=inp.pump_peak_power_w,
            duration_s=inp.pump_duration_s,
            shape=pump_shape,
            cw=inp.pump_cw,
            start_time_s=inp.pump_start_s,
        )

        prp: tuple[float, ...] | None = None
        if inp.pulse_relative_powers is not None:
            prp = tuple(float(x) for x in inp.pulse_relative_powers)

        sig_spec = ChirpedBurstSpec(
            center_wavelength_nm=inp.signal_center_nm,
            bandwidth_nm=inp.signal_bandwidth_nm,
            chirp_duration_s=inp.chirp_duration_s,
            energy_per_pulse_j=inp.energy_per_pulse_j,
            burst_count=inp.burst_count,
            burst_spacing_s=inp.burst_spacing_s,
            burst_start_time_s=inp.burst_start_s,
            rep_rate_mode=inp.rep_rate_mode,
            rep_rate_hz=inp.rep_rate_hz,
            n_periods=inp.n_periods,
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

            pump_window = max(
                inp.pump_duration_s,
                inp.burst_start_s + sig_spec.n_periods * rep_period_s(sig_spec) + packet_duration_s(sig_spec),
            )

        t = build_cpa_time_grid(
            pump_duration_s=pump_window,
            spec=sig_spec,
            pump_cw=inp.pump_cw,
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
        p_peak_in = float(sig_spec.energy_per_pulse_j / (np.sqrt(2.0 * np.pi) * sigma_t))
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
    lines = [
        f"Packet gain (energy): {g_pkt:.4f}",
        f"Single-pulse gain (1/e² window, 1st pulse): {g_pls:.4f}",
        f"Packet energy expected / in: {e_exp*1e6:.3f} / {r.energy_packet_in_j*1e6:.3f} µJ",
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
