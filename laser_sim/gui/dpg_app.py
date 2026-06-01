"""
Laser Sim — Dear PyGui desktop interface.

Run: python -m laser_sim.gui.dpg_app
     or ./run_dpg.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dearpygui.dearpygui as dpg

from laser_sim.calculators.dopant import estimate_dopant_concentration
from laser_sim.gui.runner import SimInputs, SimRunOutcome, format_energy_summary, run_simulation
from laser_sim.materials import load_material
from laser_sim.pulses.chirp import ChirpedBurstSpec
from laser_sim.viz.plot_limits import (
    dpg_axis_limits_temporal,
    integrated_power_vs_time,
    packet_center_time_s,
    pump_plot_limits,
    resolve_plot_spec_from_signal,
    sample_temporal_traces,
    spectrum_plot_limits,
)

# Plot series tags
_PUMP_SERIES = "pump_abs_series"
_SIG_IN_SERIES = "sig_in_series"
_SIG_OUT_SERIES = "sig_out_series"
_SPEC_IN_SERIES = "spec_in_series"
_SPEC_OUT_SERIES = "spec_out_series"

_last_outcome: dict[str, SimRunOutcome | None] = {"value": None}


def _log(msg: str, clear: bool = False) -> None:
    if clear:
        dpg.set_value("log_text", msg)
    else:
        prev = dpg.get_value("log_text") or ""
        dpg.set_value("log_text", prev + msg)


_MAT_KEYS = {"Yb glass": "yb_glass", "Yb YAG": "yb_yag", "Yb YLF": "yb_ylf"}


def _read_inputs() -> SimInputs:
    mat_key = _MAT_KEYS[dpg.get_value("material_combo")]
    pump_mode = dpg.get_value("pump_mode_combo")
    pump_cw = pump_mode == "CW (steady)"

    pump_dur = dpg.get_value("pump_duration_ms") * 1e-3
    if pump_cw:
        pump_dur = dpg.get_value("sim_window_ms") * 1e-3

    shape_map = {
        "Flat top": "flat_top",
        "Gaussian": "gaussian",
        "Trapezoid": "trapezoid",
    }
    pump_shape = shape_map.get(dpg.get_value("pump_shape_combo"), "flat_top")

    manual_n = dpg.get_value("manual_n_check")
    n_manual = dpg.get_value("n_manual") if manual_n else None

    pulse_weights: list[float] | None = None
    pw_text = (dpg.get_value("pulse_weights_text") or "").strip()
    if pw_text:
        try:
            pulse_weights = [float(x.strip()) for x in pw_text.split(",") if x.strip()]
        except ValueError:
            pulse_weights = None

    return SimInputs(
        material_key=mat_key,
        core_diameter_um=dpg.get_value("core_um"),
        cladding_diameter_um=dpg.get_value("clad_um"),
        fiber_length_m=dpg.get_value("fiber_len_m"),
        cladding_pumped=dpg.get_value("clad_pump_check"),
        ignore_gamma_for_n=dpg.get_value("gamma_n_check"),
        abs_mode_db_per_m=(dpg.get_value("abs_db_per_m_radio") == "dB/m"),
        pump_absorption_db_per_m=dpg.get_value("pump_db_per_m"),
        total_absorption_db=dpg.get_value("pump_total_db"),
        pump_wavelength_nm=dpg.get_value("pump_wl"),
        simulation_pump_wavelength_nm=(
            None
            if dpg.get_value("sim_pump_match_check")
            else float(dpg.get_value("sim_pump_wl"))
        ),
        yb_concentration_override_m3=n_manual,
        pump_peak_power_w=dpg.get_value("pump_power_w"),
        pump_cw=pump_cw,
        pump_shape=pump_shape,
        pump_duration_s=pump_dur,
        pump_start_s=dpg.get_value("pump_start_us") * 1e-6,
        signal_center_nm=dpg.get_value("sig_center_nm"),
        signal_bandwidth_nm=dpg.get_value("sig_bw_nm"),
        chirp_duration_s=dpg.get_value("chirp_ns") * 1e-9,
        packet_energy_j=dpg.get_value("packet_energy_uj") * 1e-6,
        burst_count=int(dpg.get_value("burst_count")),
        burst_spacing_s=dpg.get_value("burst_spacing_ns") * 1e-9,
        burst_start_s=(
            (1.0 / (dpg.get_value("rep_rate_khz") * 1e3))
            if dpg.get_value("rep_rate_check") and pump_cw
            else dpg.get_value("burst_start_us") * 1e-6
        ),
        n_z=int(dpg.get_value("n_z")),
        include_ase=dpg.get_value("ase_check"),
        rep_rate_mode=dpg.get_value("rep_rate_check") and pump_cw,
        rep_rate_hz=dpg.get_value("rep_rate_khz") * 1e3,
        n_periods=int(dpg.get_value("n_periods")),
        steady_state_tol=dpg.get_value("steady_tol"),
        pulse_relative_powers=pulse_weights,
        time_resolution=dpg.get_value("time_resolution") or "standard",
        steady_state_warmup=bool(dpg.get_value("steady_warmup_check")),
    )


def _plot_spec_from_inputs(inp: SimInputs) -> ChirpedBurstSpec:
    return ChirpedBurstSpec(
        center_wavelength_nm=inp.signal_center_nm,
        bandwidth_nm=inp.signal_bandwidth_nm,
        chirp_duration_s=inp.chirp_duration_s,
        burst_count=inp.burst_count,
        burst_spacing_s=inp.burst_spacing_s,
        burst_start_time_s=inp.burst_start_s,
        rep_rate_mode=inp.rep_rate_mode,
        rep_rate_hz=inp.rep_rate_hz,
        n_periods=inp.n_periods,
    )


def _update_dopant_display() -> None:
    try:
        inp = _read_inputs()
        mat = load_material(inp.material_key)
        if inp.abs_mode_db_per_m:
            est = estimate_dopant_concentration(
                pump_absorption_db_per_m=inp.pump_absorption_db_per_m,
                core_diameter_um=inp.core_diameter_um,
                cladding_diameter_um=inp.cladding_diameter_um,
                pump_wavelength_nm=inp.pump_wavelength_nm,
                material=mat,
                cladding_pumped=inp.cladding_pumped,
                ignore_overlap_for_concentration=inp.ignore_gamma_for_n,
            )
        else:
            from laser_sim.calculators.dopant import concentration_from_total_absorption_db

            est = concentration_from_total_absorption_db(
                total_absorption_db=inp.total_absorption_db,
                fiber_length_m=inp.fiber_length_m,
                core_diameter_um=inp.core_diameter_um,
                cladding_diameter_um=inp.cladding_diameter_um,
                pump_wavelength_nm=inp.pump_wavelength_nm,
                material=mat,
                cladding_pumped=inp.cladding_pumped,
                ignore_overlap_for_concentration=inp.ignore_gamma_for_n,
            )
        dpg.set_value(
            "dopant_text",
            f"N (rates): {est.concentration_for_rates_m3:.4e} m⁻³\n"
            f"N (label): {est.concentration_m3:.4e} m⁻³\n"
            f"ppm≈{est.concentration_ppm_wt:.0f}  κ={est.alpha_db_per_m:.2f} dB/m",
        )
    except Exception as exc:
        dpg.set_value("dopant_text", f"Dopant error: {type(exc).__name__}: {exc}")


def _on_pump_mode_change() -> None:
    cw = dpg.get_value("pump_mode_combo") == "CW (steady)"
    dpg.configure_item("pulsed_pump_group", show=not cw)
    dpg.configure_item("cw_window_group", show=cw)
    dpg.configure_item("rep_rate_check", enabled=cw)
    _on_rep_rate_ui_change()


def _on_rep_rate_ui_change() -> None:
    cw = dpg.get_value("pump_mode_combo") == "CW (steady)"
    rep = dpg.get_value("rep_rate_check") and cw
    dpg.configure_item("burst_start_us", show=not rep)
    dpg.configure_item("burst_auto_text", show=rep)
    if rep:
        t_rep_ms = 1000.0 / max(dpg.get_value("rep_rate_khz"), 1e-6)
        dpg.set_value(
            "burst_auto_text",
            f"Burst delay: auto (1 rep period = {t_rep_ms:.4f} ms) — CW warm-start",
        )


def _on_apply_flat_weights() -> None:
    out = _last_outcome.get("value")
    if out is None or out.suggested_flat_weights is None:
        _log("No flat-packet weights available — run a simulation first.\n")
        return
    w_str = ", ".join(f"{x:.4f}" for x in out.suggested_flat_weights)
    dpg.set_value("pulse_weights_text", w_str)
    _log(f"Applied flat-packet weights to next run: [{w_str}]\n")


def _update_plots(outcome, plot_spec: ChirpedBurstSpec) -> None:
    r = outcome.result
    z = r.z_m
    t = r.t_s
    wl = r.wavelength_nm
    sig_in = r.signal_fwd_w_nm[0]
    sig_out = r.signal_fwd_w_nm[-1]
    base = outcome.signal_spec if outcome.signal_spec is not None else plot_spec
    plot_spec = resolve_plot_spec_from_signal(t, sig_in, wl, base)
    p_in_t = integrated_power_vs_time(sig_in, wl)
    p_out_t = integrated_power_vs_time(sig_out, wl)
    from laser_sim.viz.plot_limits import integrate_signal_spectrum

    spec_in = integrate_signal_spectrum(sig_in, t, wl, plot_spec)
    spec_out = integrate_signal_spectrum(sig_out, t, wl, plot_spec)

    t_rel, p_in_w, p_out_w, xlim_ns = sample_temporal_traces(t, p_in_t, p_out_t, plot_spec)
    lam_lo, lam_hi = spectrum_plot_limits(plot_spec)
    m_l = (wl >= lam_lo) & (wl <= lam_hi)
    t_center = packet_center_time_s(plot_spec)

    plim = pump_plot_limits(z, r.pump_absorption_db_per_m, np.mean(r.pump_fwd_w, axis=1))
    dpg.set_axis_limits("pump_x", *plim["x"])
    dpg.set_axis_limits("pump_y", *plim["y_alpha"])

    dpg.set_value(_PUMP_SERIES, [z.tolist(), r.pump_absorption_db_per_m.tolist()])
    dpg.set_value(_SIG_IN_SERIES, [t_rel.tolist(), p_in_w.tolist()])
    dpg.set_value(_SIG_OUT_SERIES, [t_rel.tolist(), p_out_w.tolist()])
    dpg.set_value(_SPEC_IN_SERIES, [wl[m_l].tolist(), spec_in[m_l].tolist()])
    dpg.set_value(_SPEC_OUT_SERIES, [wl[m_l].tolist(), spec_out[m_l].tolist()])

    x_ns, y_in, y_out = dpg_axis_limits_temporal(t_rel, p_in_w, p_out_w, xlim_ns)
    dpg.set_axis_limits("time_x", *x_ns)
    dpg.set_axis_limits("time_y", *y_in)
    dpg.set_axis_limits("time_y2", *y_out)
    dpg.set_axis_limits("lam_x", lam_lo, lam_hi)
    ymax = float(max(spec_in[m_l].max() if m_l.any() else 0, spec_out[m_l].max() if m_l.any() else 0, 1e-30))
    dpg.set_axis_limits("lam_y", 0.0, ymax * 1.15)


def _on_run() -> None:
    _log("\n--- Run ---\n", clear=False)
    dpg.set_value("status_text", "Running…")
    inp = _read_inputs()
    outcome = run_simulation(inp)

    if not outcome.ok:
        _log(f"ERROR [{outcome.error_type}]: {outcome.error_message}\n\n")
        _log(outcome.traceback_text)
        dpg.set_value("status_text", f"Failed: {outcome.error_type}")
        dpg.set_value("metrics_text", "")
        return

    _update_plots(outcome, outcome.signal_spec or _plot_spec_from_inputs(inp))
    summary = format_energy_summary(outcome)
    dpg.set_value("metrics_text", summary)
    dpg.set_value("status_text", "OK")
    _last_outcome["value"] = outcome
    _log("Simulation finished successfully.\n")
    _log(summary + "\n")
    if outcome.suggested_flat_weights is not None:
        w_str = ", ".join(f"{x:.4f}" for x in outcome.suggested_flat_weights)
        _log(f"Suggested flat-packet weights: [{w_str}]\n")
    if outcome.dopant is not None:
        d = outcome.dopant
        _log(f"Dopant: N={d.concentration_for_rates_m3:.4e} m⁻³\n")


def _on_clear_log() -> None:
    dpg.set_value("log_text", "")


def _build_ui() -> None:
    with dpg.window(tag="primary", label="Laser Sim — Yb fiber CPA", no_close=True):
        with dpg.group(horizontal=True):
            # ---- Left: inputs ----
            with dpg.child_window(width=420, height=-1, border=True):
                dpg.add_text("Fiber & material")
                dpg.add_combo(
                    ["Yb glass", "Yb YAG", "Yb YLF"],
                    default_value="Yb glass",
                    tag="material_combo",
                    width=200,
                )
                dpg.add_input_float(label="Core (µm)", default_value=10.0, tag="core_um", width=150)
                dpg.add_input_float(label="Cladding (µm)", default_value=400.0, tag="clad_um", width=150)
                dpg.add_input_float(label="Length (m)", default_value=2.0, tag="fiber_len_m", width=150)
                dpg.add_checkbox(label="Cladding pumped", default_value=True, tag="clad_pump_check")
                dpg.add_checkbox(
                    label="Ignore Γ_p when solving N",
                    default_value=False,
                    tag="gamma_n_check",
                )

                dpg.add_separator()
                dpg.add_text("Dopant from pump absorption")
                dpg.add_radio_button(
                    ["dB/m", "Total dB"],
                    default_value="dB/m",
                    tag="abs_db_per_m_radio",
                    horizontal=True,
                )
                dpg.add_input_float(label="Pump abs (dB/m)", default_value=6.0, tag="pump_db_per_m", width=150)
                dpg.add_input_float(label="Total abs (dB)", default_value=12.0, tag="pump_total_db", width=150)
                dpg.add_input_float(
                    label="Pump λ for N calc (nm)",
                    default_value=976.0,
                    tag="pump_wl",
                    width=150,
                )
                dpg.add_button(label="Update dopant estimate", callback=_update_dopant_display)
                dpg.add_text("", tag="dopant_text", wrap=380)
                dpg.add_checkbox(label="Override N manually", tag="manual_n_check")
                dpg.add_input_float(
                    label="N (m⁻³)",
                    default_value=6e24,
                    tag="n_manual",
                    width=200,
                    format="%.4e",
                )

                dpg.add_separator()
                dpg.add_text("Pump")
                dpg.add_checkbox(
                    label="Sim pump λ = datasheet λ",
                    default_value=True,
                    tag="sim_pump_match_check",
                )
                dpg.add_input_float(
                    label="Simulation pump λ (nm)",
                    default_value=976.0,
                    tag="sim_pump_wl",
                    width=150,
                )
                dpg.add_combo(
                    ["Pulsed", "CW (steady)"],
                    default_value="Pulsed",
                    tag="pump_mode_combo",
                    width=200,
                    callback=_on_pump_mode_change,
                )
                dpg.add_input_float(label="Pump power (W)", default_value=200.0, tag="pump_power_w", width=150)
                dpg.add_input_float(label="Pump start (µs)", default_value=0.0, tag="pump_start_us", width=150)

                with dpg.group(tag="pulsed_pump_group"):
                    dpg.add_combo(
                        ["Flat top", "Gaussian", "Trapezoid"],
                        default_value="Flat top",
                        tag="pump_shape_combo",
                        width=200,
                    )
                    dpg.add_input_float(
                        label="Pump duration (ms)",
                        default_value=1.0,
                        tag="pump_duration_ms",
                        width=150,
                    )

                with dpg.group(tag="cw_window_group", show=False):
                    dpg.add_input_float(
                        label="Sim window (ms)",
                        default_value=5.0,
                        tag="sim_window_ms",
                        width=150,
                    )
                    dpg.add_text("CW: constant power over window", color=(180, 180, 180))

                dpg.add_separator()
                dpg.add_text("Signal packet (CPA)")
                dpg.add_input_float(label="Center λ (nm)", default_value=1030.0, tag="sig_center_nm", width=150)
                dpg.add_input_float(label="Bandwidth (nm)", default_value=8.0, tag="sig_bw_nm", width=150)
                dpg.add_input_float(label="Chirp (ns)", default_value=0.8, tag="chirp_ns", width=150)
                dpg.add_input_float(label="Packet energy (µJ)", default_value=10.0, tag="packet_energy_uj", width=150)
                dpg.add_input_int(label="Pulses in packet", default_value=5, tag="burst_count", width=150)
                dpg.add_input_float(
                    label="Spacing (ns)",
                    default_value=2.5,
                    tag="burst_spacing_ns",
                    width=150,
                )
                dpg.add_input_float(
                    label="Packet delay (µs)",
                    default_value=200.0,
                    tag="burst_start_us",
                    width=150,
                )
                dpg.add_text("", tag="burst_auto_text", show=False, color=(180, 180, 180))
                dpg.add_input_text(
                    label="Packet power weights",
                    default_value="",
                    tag="pulse_weights_text",
                    width=250,
                )
                dpg.add_input_int(label="z steps", default_value=120, tag="n_z", width=150)
                dpg.add_checkbox(label="ASE + spontaneous", default_value=True, tag="ase_check")

                dpg.add_separator()
                dpg.add_text("Rep-rate steady state (CW)")
                dpg.add_checkbox(
                    label="Rep-rate mode",
                    tag="rep_rate_check",
                    default_value=False,
                    enabled=False,
                    callback=_on_rep_rate_ui_change,
                )
                dpg.add_input_float(label="Rep rate (kHz)", default_value=100.0, tag="rep_rate_khz", width=150)
                dpg.add_input_int(label="Periods", default_value=40, tag="n_periods", width=150)
                dpg.add_input_float(
                    label="Steady tol",
                    default_value=0.02,
                    tag="steady_tol",
                    width=150,
                    min_value=0.001,
                    max_value=0.2,
                )
                dpg.add_checkbox(
                    label="Lumped steady-state warmup",
                    tag="steady_warmup_check",
                    default_value=True,
                )

                dpg.add_separator()
                dpg.add_text("Time resolution")
                dpg.add_combo(
                    label="Resolution",
                    items=["low", "standard", "fine"],
                    default_value="standard",
                    tag="time_resolution",
                    width=150,
                )

                dpg.add_separator()
                dpg.add_button(label="Run simulation", callback=_on_run, width=200)
                dpg.add_button(
                    label="Apply flat-packet weights to next run",
                    callback=_on_apply_flat_weights,
                    width=280,
                )
                dpg.add_button(label="Clear log", callback=_on_clear_log, width=200)
                dpg.add_text("", tag="status_text")

            # ---- Right: plots + log ----
            with dpg.child_window(width=-1, height=-1, border=True):
                dpg.add_text("", tag="metrics_text")
                with dpg.group(horizontal=True):
                    with dpg.plot(label="Pump absorption vs z", height=220, width=450):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="z (m)", tag="pump_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="α (dB/m)", tag="pump_y"):
                            dpg.add_line_series([], [], label="α pump", tag=_PUMP_SERIES)

                    with dpg.plot(label="Signal vs time (dual axis)", height=220, width=450):
                        dpg.add_plot_legend()
                        dpg.add_plot_axis(dpg.mvXAxis, label="t − t_center (ns)", tag="time_x")
                        with dpg.plot_axis(dpg.mvYAxis, label="Signal in (W)", tag="time_y"):
                            dpg.add_line_series([], [], label="In", tag=_SIG_IN_SERIES)
                        with dpg.plot_axis(dpg.mvYAxis, label="Signal out (W)", tag="time_y2"):
                            dpg.add_line_series([], [], label="Out", tag=_SIG_OUT_SERIES)

                with dpg.plot(label="Spectrum (time-integrated)", height=240, width=-1):
                    dpg.add_plot_legend()
                    dpg.add_plot_axis(dpg.mvXAxis, label="λ (nm)", tag="lam_x")
                    with dpg.plot_axis(dpg.mvYAxis, label="J/nm", tag="lam_y"):
                        dpg.add_line_series([], [], label="In", tag=_SPEC_IN_SERIES)
                        dpg.add_line_series([], [], label="Out", tag=_SPEC_OUT_SERIES)

                dpg.add_separator()
                dpg.add_text("Log / errors")
                dpg.add_input_text(
                    tag="log_text",
                    multiline=True,
                    readonly=True,
                    width=-1,
                    height=200,
                    default_value="Ready. Set parameters and click Run simulation.\n",
                )


def main() -> None:
    dpg.create_context()
    dpg.configure_app(docking=True, docking_space=True)
    _build_ui()
    dpg.create_viewport(title="Laser Sim — CPA Fiber (Dear PyGui)", width=1280, height=900)
    dpg.setup_dearpygui()
    dpg.show_viewport()
    dpg.set_primary_window("primary", True)
    _update_dopant_display()
    dpg.start_dearpygui()
    dpg.destroy_context()


if __name__ == "__main__":
    main()
