"""Streamlit GUI for hybrid fiber + solid-state CPA."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import streamlit as st

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from hybrid_sim.gui.runner import HybridSimInputs, run_hybrid_safe

st.set_page_config(page_title="Hybrid CPA", layout="wide")
st.title("Hybrid Fiber + Solid-State CPA")

with st.sidebar:
    st.header("Fiber pre-amp")
    fiber_material = st.selectbox("Fiber material", ["yb_glass"], index=0)
    fiber_length_m = st.number_input("Fiber length (m)", 0.5, 5.0, 2.0, 0.1)
    fiber_core_um = st.number_input("Core diameter (µm)", 5.0, 30.0, 10.0, 0.5)
    fiber_clad_um = st.number_input("Cladding diameter (µm)", 80.0, 600.0, 125.0, 5.0)
    fiber_pump_db = st.number_input("Pump absorption (dB/m)", 1.0, 30.0, 6.0, 0.5)
    fiber_pump_w = st.number_input("Fiber pump power (W)", 0.1, 500.0, 2.0, 0.1)

    st.header("Solid-state")
    ss_crystal = st.selectbox("Crystal", ["yb_yag", "yb_ylf", "nd_yag"], index=0)
    ss_length_mm = st.number_input("Crystal length (mm)", 1.0, 50.0, 10.0, 0.5)
    ss_at_pct = st.number_input("Doping (at.%)", 0.1, 5.0, 1.0, 0.1)
    use_cavity = st.checkbox("Cavity pump", value=True)
    include_thermal = st.checkbox("Thermal lensing", value=True)
    include_kerr = st.checkbox("Kerr / B-integral", value=True)

    st.header("Signal")
    signal_nj = st.number_input("Packet energy (nJ)", 0.1, 1000.0, 10.0, 0.1)

run_btn = st.button("Run hybrid simulation", type="primary")

tab_diag, tab_energy, tab_beam = st.tabs(["Diagnostics", "Energy audit", "Beam / crystal"])

if run_btn:
    inp = HybridSimInputs(
        fiber_material=fiber_material,
        fiber_length_m=fiber_length_m,
        fiber_core_um=fiber_core_um,
        fiber_clad_um=fiber_clad_um,
        fiber_pump_abs_db_per_m=fiber_pump_db,
        fiber_pump_power_w=fiber_pump_w,
        ss_crystal=ss_crystal,
        ss_crystal_length_mm=ss_length_mm,
        ss_doping_at_pct=ss_at_pct,
        use_cavity_pump=use_cavity,
        include_thermal=include_thermal,
        include_kerr=include_kerr,
        signal_energy_nj=signal_nj,
    )
    with st.spinner("Running…"):
        out = run_hybrid_safe(inp)
    if not out.ok:
        st.error(out.error_message)
        st.code(out.traceback_text)
    else:
        r = out.result
        fr = r.fiber.result
        sol = r.solid
        with tab_energy:
            st.table(
                {
                    "Stage": ["Fiber", "Solid-state", "Overall η"],
                    "E_out": [
                        f"{fr.energy_signal_out_j*1e3:.4f} mJ" if fr else "—",
                        f"{sol.energy_signal_out_j*1e9:.2f} nJ",
                        f"{r.eta_overall*100:.2f} %",
                    ],
                }
            )
        with tab_beam:
            import matplotlib.pyplot as plt

            I = np.abs(sol.U_out) ** 2
            fig, ax = plt.subplots()
            ax.imshow(I, cmap="hot")
            ax.set_title(f"Output |M²≈{sol.M2_x:.2f}| B_max={sol.B_integral_max:.2f} rad")
            st.pyplot(fig)
            plt.close(fig)
            st.caption(sol.b_integral_warning)
        with tab_diag:
            st.write(f"g₀ ≈ {sol.g0_db_per_mm:.3f} dB/mm, pump absorbed {sol.pump_absorbed_frac*100:.1f}%")
            st.write(f"f_thermal = {sol.thermal_lens_f_m:.3g} m, T_max = {sol.T_rise_max_k:.2e} K")
            if sol.cavity_result:
                st.write(
                    f"Cavity: E_pulse = {sol.cavity_result.pulse_energy_j*1e3:.3f} mJ, "
                    f"threshold RT = {sol.cavity_result.threshold_roundtrip}"
                )

else:
    st.info("Configure parameters in the sidebar and click **Run hybrid simulation**.")
