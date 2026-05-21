"""Small-signal gain vs fiber position."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from laser_sim.physics.fiber_cpa import FiberCPAResult


def build_small_signal_gain_vs_z_figure(result: FiberCPAResult):
    import plotly.graph_objects as go

    z = result.z_m
    if result.g0_small_signal_db_m is not None:
        g_db = result.g0_small_signal_db_m
        ylab = "Small-signal g₀ (dB/m)"
    elif result.g0_small_signal_np_m is not None:
        import numpy as np

        g_db = result.g0_small_signal_np_m * (10.0 / np.log(10))
        ylab = "Small-signal g₀ (dB/m)"
    else:
        g_db = result.n2_fraction.mean(axis=1)
        ylab = "N₂ fraction (fallback)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=z,
            y=g_db,
            mode="lines+markers",
            name="g₀ (spectral mean)",
            line=dict(color="#00CC96"),
        )
    )
    fig.update_layout(
        title="Small-signal gain coefficient vs z (pump-only steady state, before packet)",
        xaxis_title="z (m)",
        yaxis_title=ylab,
        height=360,
    )
    return fig
