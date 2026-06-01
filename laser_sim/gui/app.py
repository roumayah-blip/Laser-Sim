"""
Laser Sim — CPA fiber amplifier GUI (Streamlit).

Run: streamlit run laser_sim/gui/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laser_sim.calculators.dopant import (
    concentration_from_total_absorption_db,
    estimate_dopant_concentration,
)
from laser_sim.calculators.runtime import (
    estimate_runtime,
    recommend_time_grid,
    recommend_wavelength_grid,
)
from laser_sim.gui.backend_status import probe_compute_backend
from laser_sim.gui.fiber_presets import load_presets, preset_to_session_updates, save_preset
from laser_sim.gui.runner import (
    SimInputs,
    format_cw_reference_summary,
    format_energy_summary,
    run_simulation,
)
from laser_sim.materials import load_material
from laser_sim.pulses.chirp import ChirpedBurstSpec, time_resolution_preset
from laser_sim.viz.plot_limits import (
    apply_plotly_spectrum_layout,
    apply_plotly_temporal_dual_axis,
    apply_plotly_time_heatmap_layout,
    integrated_power_vs_time,
    integrate_signal_spectrum,
    packet_center_time_s,
    pump_plot_limits,
    resolve_plot_spec_from_signal,
    sample_temporal_traces,
    spectrum_plot_limits,
)

def _render_flat_packet_weights_ui(outcome, *, button_key: str) -> None:
    """Show suggested per-pulse weights and apply button (needs burst_count ≥ 2)."""
    if outcome.suggested_flat_weights is not None:
        w = outcome.suggested_flat_weights
        w_apply = ", ".join(f"{x:.4f}" for x in w)
        st.markdown(
            f"**Suggested flat-packet weights** (mean=1, copy or apply): `[{w_apply}]`  "
            f"(peak/mean {max(w) / max(min(w), 1e-30):.2f}×)"
        )
        if st.button("⟳ Apply flat-packet weights to next run", key=button_key):
            st.session_state["pulse_weights"] = w_apply
            st.rerun()
        return
    r = outcome.result
    if r is None or not outcome.ok:
        return
    n_burst = len(r.pulse_energies_in_j) if r.pulse_energies_in_j is not None else 0
    if n_burst <= 1:
        st.caption(
            "Flat-packet weights: set **Pulses in packet** ≥ 2 to enable equalization."
        )
    elif r.pulse_energies_in_j is None or r.pulse_energies_out_j is None:
        st.caption(
            "Flat-packet weights unavailable (per-pulse energies missing from this backend)."
        )


st.set_page_config(page_title="Laser Sim — CPA Fiber", layout="wide")
st.title("Ytterbium fiber CPA amplifier")
st.caption(
    "Chirped ns+ signal bursts (≤50), ns packet spacing, pulsed or CW pump, ASE/spontaneous."
)

_DEFAULTS = {
    "material_key": "yb_glass",
    "core_um": 10.0,
    "core_na": 0.06,
    "clad_um": 400.0,
    "fiber_length_m": 2.0,
    "cladding_pumped": True,
    "ignore_gamma_for_n": False,
    "abs_mode": "dB/m",
    "pump_abs_db_per_m": 6.0,
    "total_abs_db": 12.0,
    "pump_wl_nm": 976.0,
    "sim_pump_wl_nm": 976.0,
    "sim_backend": "cuda",
}
for _k, _v in _DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

# Apply saved-fiber preset before any keyed widgets are drawn (Streamlit restriction).
_pending_fiber = st.session_state.pop("fiber_preset_apply", None)
if _pending_fiber:
    for _sk, _sv in _pending_fiber.items():
        st.session_state[_sk] = _sv


@st.cache_resource
def _cached_compute_status():
    return probe_compute_backend(init_taichi=False)


with st.sidebar.expander("Compute backend (CUDA)", expanded=True):
    _status = _cached_compute_status()
    if _status.gpu_name:
        st.caption(f"GPU: {_status.gpu_name}")
    if _status.taichi_arch:
        st.metric("Taichi arch", _status.taichi_arch)
    if _status.cuda_active:
        st.success("CUDA acceleration is active for the signal pass.")
    elif _status.taichi_installed:
        st.info("CUDA starts in a fresh worker process on each simulation run.")
    else:
        st.info("Install Taichi for GPU: `pip install taichi`")
    st.caption(_status.notes)

mat_key = st.sidebar.selectbox(
    "Material",
    ["yb_glass", "yb_yag", "yb_ylf"],
    format_func=lambda k: k.replace("_", " ").title(),
    key="material_key",
)
material = load_material(mat_key)

st.sidebar.header("Fiber geometry")
core_um = st.sidebar.number_input("Core diameter (µm)", 5.0, 50.0, key="core_um", step=0.5)
core_na = st.sidebar.number_input(
    "Signal core NA",
    0.01,
    0.30,
    key="core_na",
    step=0.005,
    help="Sets signal mode area and guided spontaneous fraction η.",
)
clad_um = st.sidebar.number_input("Cladding diameter (µm)", 50.0, 1000.0, key="clad_um", step=10.0)
fiber_length_m = st.sidebar.number_input(
    "Fiber length (m)",
    min_value=1e-5,
    max_value=20.0,
    step=0.001,
    format="%.6f",
    key="fiber_length_m",
    help="Active Yb-doped length. Values down to ~10 µm are allowed (use enough z steps).",
)
cladding_pumped = st.sidebar.checkbox("Cladding pumped", key="cladding_pumped")
ignore_gamma_for_n = st.sidebar.checkbox(
    "Ignore Γ_p when solving N (legacy N = κ/σ only)",
    key="ignore_gamma_for_n",
    help="Default: N = κ/(Γ_p·σ) with Γ_p = (r_core/r_clad)². Core size then affects N.",
)

st.sidebar.header("Saved fiber parameters")
_presets = load_presets()
_preset_names = ["—"] + sorted(_presets.keys())
_preset_pick = st.sidebar.selectbox("Quick load", _preset_names, key="fiber_preset_pick")
if st.sidebar.button("Apply saved fiber", use_container_width=True):
    if _preset_pick != "—" and _preset_pick in _presets:
        st.session_state["fiber_preset_apply"] = preset_to_session_updates(_presets[_preset_pick])
        st.rerun()
    elif _preset_pick == "—":
        st.sidebar.warning("Choose a saved fiber from the list first.")

_preset_save_name = st.sidebar.text_input("Save as name", key="fiber_preset_save_name", placeholder="e.g. Liekki 30/250 14.6 dB/m")
if st.sidebar.button("Save fiber parameters", use_container_width=True):
    try:
        _saved = save_preset(
            _preset_save_name,
            {
                "material_key": mat_key,
                "core_diameter_um": core_um,
                "core_na": core_na,
                "cladding_diameter_um": clad_um,
                "fiber_length_m": fiber_length_m,
                "cladding_pumped": cladding_pumped,
                "ignore_gamma_for_n": ignore_gamma_for_n,
                "abs_mode_db_per_m": st.session_state["abs_mode"] == "dB/m",
                "pump_absorption_db_per_m": st.session_state["pump_abs_db_per_m"],
                "total_absorption_db": st.session_state["total_abs_db"],
                "pump_wavelength_nm": st.session_state["pump_wl_nm"],
            },
        )
        st.sidebar.success(f"Saved “{_saved}”.")
        st.rerun()
    except ValueError as exc:
        st.sidebar.error(str(exc))

st.sidebar.header("Dopant from pump absorption")
abs_mode = st.sidebar.radio("Absorption spec", ["dB/m", "Total dB over length"], key="abs_mode")
pump_abs_db_per_m = st.sidebar.number_input(
    "Pump absorption (dB/m)", 0.1, 50.0, key="pump_abs_db_per_m", step=0.1
)
total_abs_db = st.sidebar.number_input(
    "Total pump absorption (dB)", 0.1, 100.0, key="total_abs_db", step=0.5
)
pump_wl_nm = st.sidebar.number_input(
    "Pump λ for N calc (nm)",
    850.0,
    1100.0,
    key="pump_wl_nm",
    step=1.0,
    help="Wavelength assumed in the fiber datasheet pump absorption (dB/m). "
    "Used only to back-calculate Yb concentration, not necessarily the simulation pump.",
)

if abs_mode == "dB/m":
    dopant_est = estimate_dopant_concentration(
        pump_absorption_db_per_m=pump_abs_db_per_m,
        core_diameter_um=core_um,
        cladding_diameter_um=clad_um,
        pump_wavelength_nm=pump_wl_nm,
        material=material,
        cladding_pumped=cladding_pumped,
        ignore_overlap_for_concentration=ignore_gamma_for_n,
    )
else:
    dopant_est = concentration_from_total_absorption_db(
        total_absorption_db=total_abs_db,
        fiber_length_m=fiber_length_m,
        core_diameter_um=core_um,
        cladding_diameter_um=clad_um,
        pump_wavelength_nm=pump_wl_nm,
        material=material,
        cladding_pumped=cladding_pumped,
        ignore_overlap_for_concentration=ignore_gamma_for_n,
    )

st.sidebar.metric("Γ_p (overlap)", f"{dopant_est.gamma_pump:.5f}")
st.sidebar.metric("Yb concentration N (m⁻³)", f"{dopant_est.concentration_m3:.3e}")
st.sidebar.metric("N for rates (m⁻³)", f"{dopant_est.concentration_for_rates_m3:.3e}")
st.sidebar.metric("Approx. ppm (wt scale)", f"{dopant_est.concentration_ppm_wt:.0f}")
st.sidebar.caption(dopant_est.notes)

use_manual_n = st.sidebar.checkbox("Override concentration manually", False)
if use_manual_n:
    yb_n = st.sidebar.number_input(
        "N (m⁻³)",
        min_value=1e20,
        max_value=1e28,
        value=float(dopant_est.concentration_for_rates_m3),
        format="%.4e",
    )
else:
    yb_n = None

tab_calc, tab_run = st.tabs(["Calculators & grid planner", "Run simulation"])

with tab_calc:
    st.subheader("Runtime estimator")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**Pump**")
        calc_pump_mode = st.selectbox(
            "Pump mode",
            ["Pulsed", "CW (steady)"],
            key="calc_pump_mode",
        )
        calc_pump_cw = calc_pump_mode.startswith("CW")
        if calc_pump_cw:
            pump_dur_s_calc = st.number_input(
                "Sim window (ms)", 0.1, 5000.0, 5.0, 0.1, key="calc_pump_win_ms"
            ) * 1e-3
        else:
            pump_dur_s_calc = st.number_input(
                "Pump duration (µs)", 1.0, 5000.0, 1000.0, 10.0, key="calc_pump_us"
            ) * 1e-6

    with c2:
        st.markdown("**Chirped signal (CPA)**")
        chirp_ns = st.number_input("Chirped pulse duration (ns)", 0.5, 50.0, 0.8, 0.1)
        burst_count = st.number_input("Pulses in packet", 1, 50, 5, 1)
        burst_spacing_ns = st.number_input(
            "Pulse spacing in packet (ns)",
            0.5,
            1000.0,
            2.5,
            0.1,
            help="Min 0.5 ns. Overlapping chirped pulses add in intensity.",
        )
        sig_bw_nm = st.number_input("Signal bandwidth (nm)", 1.0, 40.0, 8.0, 0.5)
        sig_center_nm = st.number_input("Signal center (nm)", 980.0, 1100.0, 1030.0, 1.0)

    with c3:
        st.markdown("**Discretization**")
        n_z = st.number_input("z steps", 20, 2000, 150, 10)
        points_per_nm = st.number_input("λ points per nm", 0.5, 10.0, 2.0, 0.5)
        backend = st.selectbox(
            "Backend",
            ["cuda", "cpu"],
            index=0 if st.session_state.get("sim_backend", "cuda") == "cuda" else 1,
            key="calc_backend",
            help="cuda → Taichi GPU signal pass; pump QSS stays on CPU.",
        )
        calib = st.slider("Calibration factor", 0.2, 5.0, 1.0, 0.1)

    chirp_s = chirp_ns * 1e-9
    burst_spacing_s = burst_spacing_ns * 1e-9
    burst_span_s = (burst_count - 1) * burst_spacing_s

    t0, t1, n_t_rec = recommend_time_grid(
        pump_duration_s=pump_dur_s_calc,
        burst_span_s=burst_span_s,
        chirped_pulse_duration_s=chirp_s,
        burst_count=int(burst_count),
        burst_spacing_s=burst_spacing_s,
    )
    lam_min, lam_max, n_lam_rec = recommend_wavelength_grid(
        center_nm=sig_center_nm,
        bandwidth_nm=sig_bw_nm,
        points_per_nm=points_per_nm,
    )

    rt = estimate_runtime(
        fiber_length_m=fiber_length_m,
        n_z=int(n_z),
        n_t=n_t_rec,
        n_lambda=n_lam_rec,
        n_burst_pulses=int(burst_count),
        backend=backend,
        calibration_factor=calib,
        pump_duration_s=pump_dur_s_calc,
        signal_chirp_duration_s=chirp_s,
    )

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Recommended n_t", n_t_rec)
    col_b.metric("Recommended n_λ", n_lam_rec)
    col_c.metric("Est. runtime", f"{rt.estimated_seconds:.2f} s ({rt.estimated_minutes:.2f} min)")

    st.json(
        {
            "grid": f"{rt.n_z} × {rt.n_t} × {rt.n_lambda}",
            "burst_pulses": rt.n_burst_pulses,
            "cell_updates": rt.total_cell_updates,
            "backend": rt.backend,
            "breakdown": rt.breakdown,
            "time_window_s": [t0, t1],
            "wavelength_nm": [lam_min, lam_max],
            "notes": rt.notes,
        }
    )

with tab_run:
    st.subheader("Simulation parameters")
    st.info("Use the **Calculators** tab for grid size and runtime; values below drive the physics run.")

    r1, r2, r3 = st.columns(3)
    with r1:
        pump_peak_w = st.number_input("Pump peak power (W)", 0.1, 5000.0, 80.0, 1.0)
        sim_pump_match = st.checkbox(
            "Simulation pump λ = datasheet λ",
            value=True,
            key="sim_pump_match_wl",
            help="Uncheck to pump at a different wavelength in the physics run "
            "(must lie within the glass σ table).",
        )
        if sim_pump_match:
            sim_pump_wl_nm = pump_wl_nm
            st.caption(f"Simulation pump: **{sim_pump_wl_nm:.1f} nm** (same as N calc)")
        else:
            wl_lo, wl_hi = material.wavelength_range_nm
            sim_pump_wl_nm = st.number_input(
                "Simulation pump λ (nm)",
                float(wl_lo),
                float(wl_hi),
                float(st.session_state.get("sim_pump_wl_nm", pump_wl_nm)),
                1.0,
                key="sim_pump_wl_nm",
                help=f"Must be within {material.name} table: {wl_lo:.0f}–{wl_hi:.0f} nm.",
            )
        pump_mode_run = st.selectbox("Pump mode", ["Pulsed", "CW (steady)"], key="pump_mode_run")
        pump_cw_run = pump_mode_run.startswith("CW")
        # Both inputs always exist (avoids NameError when switching pump mode).
        pump_dur_us_run = st.number_input(
            "Pump duration (µs)",
            1.0,
            5000.0,
            1000.0,
            10.0,
            key="pump_dur_run",
            disabled=pump_cw_run,
        )
        pump_dur_ms_run = st.number_input(
            "Sim window (ms)",
            0.1,
            5000.0,
            5.0,
            0.1,
            key="pump_win_ms",
            disabled=not pump_cw_run,
        )
        pump_shape_run = st.selectbox(
            "Pump shape",
            ["flat_top", "gaussian", "trapezoid"],
            key="pump_shape_run",
            disabled=pump_cw_run,
        )
    with r2:
        sig_center_run = st.number_input("Signal center (nm)", 980.0, 1100.0, 1030.0, 1.0, key="sig_c_run")
        sig_bw_run = st.number_input("Signal bandwidth (nm)", 1.0, 40.0, 8.0, 0.5, key="sig_bw_run")
        packet_energy_uj = st.number_input("Packet energy (µJ)", 0.01, 10000.0, 10.0, 0.1)
        chirp_ns_run = st.number_input("Chirped duration (ns)", 0.5, 50.0, 0.8, 0.1, key="chirp_run")

    st.markdown("**Rep-rate steady state** (CW pump: repeat packet until output stabilizes)")
    rep_col1, rep_col2, rep_col3 = st.columns(3)
    with rep_col1:
        rep_rate_mode = st.checkbox(
            "Enable rep-rate mode",
            value=False,
            disabled=not pump_cw_run,
            help="Requires CW pump. Repeats the pulse packet at the rep rate.",
        )
    with rep_col2:
        rep_rate_khz = st.number_input("Rep rate (kHz)", 0.1, 10_000.0, 100.0, 1.0)
    with rep_col3:
        n_periods = st.number_input(
            "Periods to simulate",
            5,
            200,
            40,
            1,
            help="Only used when steady-state warmup is OFF — then we resolve every "
            "packet period explicitly. With warmup ON we lump per-period extraction "
            "until N₂(z) converges, then run ONE time-resolved period.",
        )
    rr_a, rr_b = st.columns(2)
    with rr_a:
        steady_tol = st.slider("Steady-state tolerance", 0.001, 0.1, 0.02, 0.001, format="%.3f")
    with rr_b:
        steady_state_warmup = st.checkbox(
            "Lumped steady-state warmup (RP-Fiber-Power-style)",
            value=True,
            disabled=not (rep_rate_mode and pump_cw_run),
            help="Iterate per-period Frantz–Nodvik extraction + pump rebuild on a "
            "single representative packet until N₂(z) just-before-packet converges, "
            "then run only ONE time-resolved period. Drops n_t by a factor of ~n_periods.",
        )

    with r3:
        burst_n = st.number_input("Pulses in packet", 1, 50, 5, 1, key="burst_n")
        burst_sp_ns = st.number_input(
            "Pulse spacing in packet (ns)",
            0.5,
            1000.0,
            2.5,
            0.1,
            key="burst_sp",
            help="Min 0.5 ns. Overlaps sum in intensity.",
        )
        rep_burst_auto = rep_rate_mode and pump_cw_run
        if rep_burst_auto:
            t_rep_ms = 1000.0 / rep_rate_khz
            st.caption(
                f"Burst delay: auto (1 rep period = {t_rep_ms:.4f} ms) — "
                "CW pump warm-started to steady state"
            )
            # µs: T_rep = 1/(rep_rate_khz·1e3 Hz); do not use 1/rep_rate_khz (that treats kHz as Hz).
            burst_start_us = 1000.0 / rep_rate_khz
        else:
            burst_start_us = st.number_input(
                "Packet delay after t=0 (µs)",
                0.0,
                5000.0,
                200.0,
                10.0,
                key="burst_t0",
                help="Align chirped packet with pump build-up (CPA).",
            )
            if rep_rate_mode and not pump_cw_run:
                try:
                    _mat_rr = load_material(mat_key)
                    _tau_ms = 4.0 * _mat_rr.lifetime_s * 1000.0
                    st.warning(
                        f"For accurate steady state, set burst delay ≥ 4×τ₂₁ = {_tau_ms:.2f} ms"
                    )
                except Exception:
                    pass
        pulse_weights_text = st.text_input(
            "Packet power weights (comma-separated)",
            "",
            key="pulse_weights",
            help="Per-pulse relative energies; length must equal pulse count. "
            "Empty = flat packet. Example: 1.0, 1.0, 1.2, 1.4, 1.6",
        )
        n_z_run = st.number_input("z steps", 30, 800, 120, 10, key="nz_run")
        sim_backend = st.selectbox(
            "Simulation backend",
            ["cuda", "cpu"],
            key="sim_backend",
            help="Use cuda for Taichi GPU acceleration on the signal/ASE pass.",
        )
        _run_status = _cached_compute_status()
        st.caption(
            f"Active Taichi: {_run_status.taichi_arch or 'not initialized'} "
            f"({'CUDA OK' if _run_status.cuda_active else 'CPU fallback'})"
        )
        include_ase = st.checkbox("Include ASE + spontaneous", True)
        export_diagnostics = st.checkbox(
            "Diagnostics (write .txt report)",
            False,
            help="Equations, all coefficients in SI, per-z steady-state populations, "
            "small-signal gain G0, and pulse gain G_pulse.",
        )
        cw_signal_average = st.checkbox(
            "CW signal at packet-average power",
            False,
            help="Also run with constant signal power = packet energy / packet span "
            "(sanity check for expected gain).",
        )

    st.markdown("**Time resolution** (controls n_t for the signal-pass)")
    tr_a, tr_b = st.columns([1, 2])
    with tr_a:
        time_resolution = st.selectbox(
            "Resolution preset",
            ["low", "standard", "fine"],
            index=1,
            help=(
                "low: 20 pts/chirped pulse, 4 pts/pulse spacing (fastest, ~4–6× faster). "
                "standard: 80/16 (legacy default). "
                "fine: 160/32 (most accurate, slowest)."
            ),
        )
    with tr_b:
        try:
            _tr_preview = time_resolution_preset(time_resolution)
            _burst_start_prev = (
                1.0 / (rep_rate_khz * 1e3)
                if (rep_rate_mode and pump_cw_run)
                else burst_start_us * 1e-6
            )
            _pump_dur_prev = pump_dur_ms_run * 1e-3 if pump_cw_run else pump_dur_us_run * 1e-6
            _, _, _n_t_one = recommend_time_grid(
                pump_duration_s=_pump_dur_prev,
                burst_span_s=(int(burst_n) - 1) * float(burst_sp_ns) * 1e-9,
                chirped_pulse_duration_s=float(chirp_ns_run) * 1e-9,
                burst_count=int(burst_n),
                burst_spacing_s=float(burst_sp_ns) * 1e-9,
                burst_start_time_s=_burst_start_prev,
                points_per_chirped_pulse=_tr_preview["points_per_chirped_pulse"],
                points_per_burst_spacing=_tr_preview["points_per_burst_spacing"],
            )
            uses_warmup = bool(rep_rate_mode and pump_cw_run and steady_state_warmup)
            multiplier = 1 if uses_warmup or not rep_rate_mode else max(1, int(n_periods))
            n_t_est = _n_t_one * multiplier
            st.caption(
                f"Estimated n_t ≈ {n_t_est:,}  "
                f"(preset → {_tr_preview['points_per_chirped_pulse']} pts/chirp, "
                f"{_tr_preview['points_per_burst_spacing']} pts/spacing"
                + (
                    f"; ×{multiplier} periods — disable warmup to resolve every period"
                    if multiplier > 1
                    else ("; warmup ON → 1 period time-resolved" if uses_warmup else "")
                )
                + ")"
            )
        except Exception:
            pass

    st.markdown("**Passive delivery fiber (B-integral)**")
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        passive_before_m = st.number_input(
            "Passive fiber before amp (m)",
            0.0,
            50.0,
            0.0,
            0.1,
            key="passive_before",
        )
    with b_col2:
        passive_after_m = st.number_input(
            "Passive fiber after amp (m)",
            0.0,
            50.0,
            0.0,
            0.1,
            key="passive_after",
        )

    run_btn = st.button("Run CPA simulation", type="primary")

    if run_btn:
        pump_dur_s = pump_dur_ms_run * 1e-3 if pump_cw_run else pump_dur_us_run * 1e-6

        pulse_weights: list[float] | None = None
        if pulse_weights_text.strip():
            try:
                pulse_weights = [
                    float(x.strip()) for x in pulse_weights_text.split(",") if x.strip()
                ]
            except ValueError:
                st.error("Packet power weights must be comma-separated numbers.")
                st.stop()
            if len(pulse_weights) != int(burst_n):
                st.warning(
                    f"Packet power weights: got {len(pulse_weights)} values but "
                    f"burst has {int(burst_n)} pulses — ignoring weights (flat packet)."
                )
                pulse_weights = None

        inp = SimInputs(
            material_key=mat_key,
            core_diameter_um=core_um,
            core_na=core_na,
            cladding_diameter_um=clad_um,
            fiber_length_m=fiber_length_m,
            cladding_pumped=cladding_pumped,
            ignore_gamma_for_n=ignore_gamma_for_n,
            abs_mode_db_per_m=(abs_mode == "dB/m"),
            pump_absorption_db_per_m=pump_abs_db_per_m,
            total_absorption_db=total_abs_db,
            pump_wavelength_nm=pump_wl_nm,
            simulation_pump_wavelength_nm=(
                None if sim_pump_match else float(sim_pump_wl_nm)
            ),
            yb_concentration_override_m3=yb_n,
            pump_peak_power_w=pump_peak_w,
            pump_cw=pump_cw_run,
            pump_shape=pump_shape_run,
            pump_duration_s=pump_dur_s,
            signal_center_nm=sig_center_run,
            signal_bandwidth_nm=sig_bw_run,
            chirp_duration_s=chirp_ns_run * 1e-9,
            packet_energy_j=packet_energy_uj * 1e-6,
            burst_count=int(burst_n),
            burst_spacing_s=burst_sp_ns * 1e-9,
            burst_start_s=burst_start_us * 1e-6,
            n_z=int(n_z_run),
            include_ase=include_ase,
            rep_rate_mode=rep_rate_mode and pump_cw_run,
            rep_rate_hz=rep_rate_khz * 1e3,
            n_periods=int(n_periods),
            steady_state_tol=steady_tol,
            export_diagnostics=export_diagnostics,
            cw_signal_average_power=cw_signal_average,
            backend=sim_backend,
            passive_fiber_before_m=float(passive_before_m),
            passive_fiber_after_m=float(passive_after_m),
            pulse_relative_powers=pulse_weights,
            time_resolution=time_resolution,
            steady_state_warmup=bool(steady_state_warmup),
        )

        plot_spec = ChirpedBurstSpec(
            center_wavelength_nm=sig_center_run,
            bandwidth_nm=sig_bw_run,
            chirp_duration_s=chirp_ns_run * 1e-9,
            burst_count=int(burst_n),
            burst_spacing_s=burst_sp_ns * 1e-9,
            burst_start_time_s=burst_start_us * 1e-6,
            rep_rate_mode=rep_rate_mode and pump_cw_run,
            rep_rate_hz=rep_rate_khz * 1e3,
            n_periods=int(n_periods),
        )

        progress_bar = st.progress(0.0, text="0% — Starting…")
        progress_caption = st.empty()

        def _on_progress(frac: float, message: str) -> None:
            pct = int(round(frac * 100))
            progress_bar.progress(frac, text=f"{pct}% — {message}")
            progress_caption.markdown(f"**{pct}%** — {message}")

        outcome = run_simulation(inp, progress_callback=_on_progress)
        progress_bar.progress(1.0, text="100% — Done")
        progress_caption.markdown("**100%** — Simulation complete")
        st.session_state["last_sim_outcome"] = outcome
        st.session_state["last_sim_inputs"] = inp

        if not outcome.ok:
            st.error(f"Simulation failed: **{outcome.error_type}** — {outcome.error_message}")
            with st.expander("Traceback", expanded=True):
                st.code(outcome.traceback_text)
            st.stop()

        result = outcome.result
        st.success("Simulation complete")
        if outcome.backend_used:
            arch_note = f" (Taichi {outcome.taichi_arch})" if outcome.taichi_arch else ""
            st.caption(
                f"Backend: **{outcome.backend_requested}** → `{outcome.backend_used}`{arch_note}"
                + (f" · wall time **{outcome.wall_time_s:.2f} s**" if outcome.wall_time_s else "")
            )
        st.text(format_energy_summary(outcome))
        _render_flat_packet_weights_ui(outcome, button_key="apply_flat_weights_run")
        cw_line = format_cw_reference_summary(outcome)
        if cw_line:
            st.info(cw_line)
        if outcome.diagnostics_report_path:
            st.success(f"Diagnostics written to `{outcome.diagnostics_report_path}`")
            if outcome.diagnostics_report_text:
                st.download_button(
                    "Download diagnostics .txt",
                    outcome.diagnostics_report_text,
                    file_name=Path(outcome.diagnostics_report_path).name,
                    mime="text/plain",
                )
            with st.expander("Diagnostics preview (first 120 lines)", expanded=False):
                preview = outcome.diagnostics_report_text or ""
                st.code("\n".join(preview.splitlines()[:120]))

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pump energy in", f"{result.energy_pump_in_j * 1e3:.3f} mJ")
        m2.metric("Pump energy out (fwd)", f"{result.energy_pump_out_j * 1e3:.3f} mJ")
        m3.metric(
            "Packet energy in",
            f"{result.energy_packet_in_j * 1e6:.3f} µJ",
            help=f"All pulses, packet time window. Expected: "
            f"{result.energy_packet_expected_j * 1e6:.3f} µJ",
        )
        m4.metric("Packet energy out", f"{result.energy_packet_out_j * 1e6:.3f} µJ")
        p1, p2, p3 = st.columns(3)
        p1.metric(
            "Single-pulse energy in (1/e²)",
            f"{result.energy_pulse_in_j * 1e6:.4f} µJ",
            help="First pulse only, integrated to 1/e² intensity radius",
        )
        p2.metric("Single-pulse energy out (1/e²)", f"{result.energy_pulse_out_j * 1e6:.4f} µJ")
        p3.metric("ASE energy out (approx.)", f"{result.energy_ase_out_j * 1e6:.4f} µJ")
        st.caption(
            "Amplification uses instantaneous P(t,λ) on the full time grid — not these "
            "integrated metrics. Gain ratios above are diagnostic only."
        )
        st.caption(result.notes)

        from laser_sim.viz.cross_sections import build_cross_section_figure

        st.subheader("Cross sections (material model)")
        st.plotly_chart(build_cross_section_figure(result), use_container_width=True)
        _pump_cap = (
            f"Pump (simulation): σ_abs = {result.sigma_abs_pump_m2:.3e} m² @ "
            f"{result.pump_wavelength_nm:.1f} nm. "
        )
        if abs(result.pump_wavelength_nm - pump_wl_nm) > 0.5:
            _pump_cap += f"N calc used {pump_wl_nm:.1f} nm. "
        _pump_cap += (
            "N = κ/(Γ_p·σ); z depletion uses α_p = Γ_p·σ·N₀ (core size affects Γ_p and N)."
        )
        st.caption(_pump_cap)

        z, t, wl = result.z_m, result.t_s, result.wavelength_nm
        sig_in = result.signal_fwd_w_nm[0]
        sig_out = result.signal_fwd_w_nm[-1]

        plot_spec = resolve_plot_spec_from_signal(
            t,
            sig_in,
            wl,
            outcome.signal_spec if outcome.signal_spec is not None else plot_spec,
        )

        if outcome.warmup_result is not None and outcome.warmup_result.e_out_per_iter_j.size:
            w = outcome.warmup_result
            iters = np.arange(1, w.e_out_per_iter_j.size + 1)
            st.subheader("Steady-state warmup (lumped per-period energy)")
            wc1, wc2 = st.columns(2)
            with wc1:
                fig_we = go.Figure()
                fig_we.add_trace(
                    go.Scatter(
                        x=iters,
                        y=w.e_out_per_iter_j * 1e6,
                        mode="lines+markers",
                        name="Packet E_out (µJ)",
                    )
                )
                fig_we.add_hline(
                    y=w.e_packet_in_j * 1e6,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="E_in",
                )
                fig_we.update_layout(
                    title="Warmup packet energy per iteration",
                    xaxis_title="Iteration",
                    yaxis_title="Energy (µJ)",
                    height=340,
                )
                st.plotly_chart(fig_we, use_container_width=True)
            with wc2:
                fig_wg = go.Figure()
                fig_wg.add_trace(
                    go.Scatter(
                        x=iters,
                        y=w.gain_per_iter,
                        mode="lines+markers",
                        name="Gain E_out/E_in",
                    )
                )
                fig_wg.add_trace(
                    go.Scatter(
                        x=iters,
                        y=w.residual_per_iter,
                        mode="lines+markers",
                        name="N₂ residual",
                        yaxis="y2",
                    )
                )
                fig_wg.update_layout(
                    title="Warmup gain & inversion residual",
                    xaxis_title="Iteration",
                    yaxis=dict(title="Packet gain"),
                    yaxis2=dict(
                        title="‖ΔN₂‖/‖N₂‖",
                        overlaying="y",
                        side="right",
                        type="log"
                        if float(np.max(w.residual_per_iter)) > 1e-12
                        else "linear",
                    ),
                    height=340,
                )
                st.plotly_chart(fig_wg, use_container_width=True)
            st.caption(
                f"Lumped Frantz–Nodvik extraction + pump rebuild @ "
                f"{result.rep_rate_hz / 1e3:.2f} kHz. "
                f"Final: {w.e_packet_out_j * 1e6:.4f} µJ out, "
                f"gain {w.e_packet_out_j / max(w.e_packet_in_j, 1e-30):.3f}, "
                f"G₀ ≈ {w.small_signal_gain_db:.2f} dB."
            )

        fig_pump = go.Figure()
        fig_pump.add_trace(
            go.Scatter(x=z, y=result.pump_absorption_db_per_m, name="Local α_pump (dB/m)")
        )
        fig_pump.add_trace(
            go.Scatter(
                x=z,
                y=np.mean(result.pump_fwd_w, axis=1),
                name="Mean pump power (W)",
                yaxis="y2",
            )
        )
        plim = pump_plot_limits(z, result.pump_absorption_db_per_m, np.mean(result.pump_fwd_w, axis=1))
        fig_pump.update_layout(
            title="Pump absorption & power vs z",
            xaxis=dict(title="z (m)", range=list(plim["x"])),
            yaxis=dict(title="α (dB/m)", range=list(plim["y_alpha"])),
            yaxis2=dict(
                title="Pump power (W)",
                overlaying="y",
                side="right",
                range=list(plim["y_pump"]),
            ),
            height=380,
        )
        st.plotly_chart(fig_pump, use_container_width=True)

        p_in_t = integrated_power_vs_time(sig_in, wl)
        p_out_t = integrated_power_vs_time(sig_out, wl)
        t_rel, p_in_w, p_out_w, xlim_ns = sample_temporal_traces(t, p_in_t, p_out_t, plot_spec)

        fig_t = go.Figure()
        fig_t.add_trace(
            go.Scatter(
                x=t_rel,
                y=p_in_w,
                name="Signal in",
                line=dict(color="#636EFA"),
            )
        )
        fig_t.add_trace(
            go.Scatter(
                x=t_rel,
                y=p_out_w,
                name="Signal out",
                yaxis="y2",
                line=dict(color="#EF553B"),
            )
        )
        apply_plotly_temporal_dual_axis(fig_t, t_rel, p_in_w, p_out_w, xlim_ns)
        fig_t.update_layout(
            title="Temporal signal (t=0 at packet center, dual axis)",
            height=380,
        )
        st.plotly_chart(fig_t, use_container_width=True)

        spec_in = integrate_signal_spectrum(sig_in, t, wl, plot_spec)
        spec_out = integrate_signal_spectrum(sig_out, t, wl, plot_spec)
        lam_lo, lam_hi = spectrum_plot_limits(plot_spec)
        m_l = (wl >= lam_lo) & (wl <= lam_hi)

        fig_l = go.Figure()
        fig_l.add_trace(go.Scatter(x=wl[m_l], y=spec_in[m_l], name="Spectrum in"))
        fig_l.add_trace(go.Scatter(x=wl[m_l], y=spec_out[m_l], name="Spectrum out"))
        apply_plotly_spectrum_layout(fig_l, plot_spec)
        fig_l.update_layout(title="Signal spectrum (time-integrated)", height=380)
        st.plotly_chart(fig_l, use_container_width=True)

        t_center = packet_center_time_s(plot_spec)
        m_t = (t >= t_center + xlim_ns[0] * 1e-9) & (t <= t_center + xlim_ns[1] * 1e-9)

        fig_hm = go.Figure(
            data=go.Heatmap(
                x=wl[m_l],
                y=(t[m_t] - t_center) * 1e9,
                z=sig_out[np.ix_(m_t, m_l)],
                colorscale="Viridis",
                colorbar_title="W/nm",
            )
        )
        apply_plotly_time_heatmap_layout(fig_hm, t, wl, plot_spec)
        fig_hm.update_layout(title="Output signal P(t, λ)", height=420)
        st.plotly_chart(fig_hm, use_container_width=True)

        st.subheader("Population snapshots (quasi-2-level, N₁ ≈ 0)")
        from laser_sim.viz.populations import (
            build_population_snapshot_figure,
            build_population_vs_z_figure,
        )

        st.plotly_chart(build_population_snapshot_figure(result, plot_spec), use_container_width=True)
        st.plotly_chart(build_population_vs_z_figure(result, plot_spec), use_container_width=True)

        from laser_sim.viz.gain_plot import build_small_signal_gain_vs_z_figure

        st.subheader("Small-signal gain vs z")
        st.caption(
            f"g₀ from Γ_s(σ_e N₂ − σ_a N₀) at t before packet. "
            f"Γ_s={result.gamma_signal:.4f}, η_guided={result.eta_guided_spontaneous:.4f}, "
            f"V={result.v_number:.2f}, τ₂₁={result.four_level_lifetimes.tau_21_s*1e3:.2f} ms"
            if result.four_level_lifetimes
            else ""
        )
        st.plotly_chart(build_small_signal_gain_vs_z_figure(result), use_container_width=True)

    # Persist equalization UI from last successful run (survives Streamlit reruns).
    _last_out = st.session_state.get("last_sim_outcome")
    if _last_out is not None and getattr(_last_out, "ok", False):
        with st.expander("Packet equalization (last run)", expanded=False):
            _render_flat_packet_weights_ui(_last_out, button_key="apply_flat_weights_last")
