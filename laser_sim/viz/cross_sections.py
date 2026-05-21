"""Plot absorption and emission cross sections used in the simulation."""

from __future__ import annotations

import plotly.graph_objects as go

from laser_sim.physics.fiber_cpa import FiberCPAResult


def build_cross_section_figure(result: FiberCPAResult) -> go.Figure:
    wl = result.wavelength_nm
    sig_a = result.sigma_abs_signal_m2
    sig_e = result.sigma_em_signal_m2
    pump_nm = result.pump_wavelength_nm
    sigma_p = result.sigma_abs_pump_m2

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=sig_a,
            name="σ_abs (signal grid)",
            line=dict(color="#636EFA"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=wl,
            y=sig_e,
            name="σ_em (signal grid)",
            line=dict(color="#EF553B"),
        )
    )
    fig.add_vline(
        x=pump_nm,
        line_dash="dash",
        line_color="#00CC96",
        annotation_text=f"pump λ = {pump_nm:.1f} nm",
    )
    fig.add_trace(
        go.Scatter(
            x=[pump_nm],
            y=[sigma_p],
            mode="markers+text",
            name=f"σ_abs @ pump ({sigma_p:.2e} m²)",
            text=[f"σ_p = {sigma_p:.2e}"],
            textposition="top center",
            marker=dict(size=12, color="#00CC96", symbol="diamond"),
        )
    )
    fig.update_layout(
        title=(
            f"Yb cross sections (κ_datasheet = σ_p·N = {result.kappa_datasheet_np_m:.3f} m⁻¹)"
        ),
        xaxis_title="Wavelength (nm)",
        yaxis_title="Cross section (m²)",
        yaxis_type="log",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig
