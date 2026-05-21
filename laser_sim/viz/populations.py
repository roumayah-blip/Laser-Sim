"""Population snapshot plots at packet arrival / departure times."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from laser_sim.physics.fiber_cpa import FiberCPAResult
    from laser_sim.pulses.chirp import ChirpedBurstSpec


def population_snapshot_indices(
    t_s: np.ndarray,
    spec: ChirpedBurstSpec,
) -> tuple[int, int, float, float]:
    """Return (it_first, it_last, t_first_s, t_last_s) for packet endpoints."""
    from laser_sim.pulses.chirp import (
        first_pulse_center_time_s,
        last_pulse_center_time_s,
        nearest_time_index,
    )

    t_first = first_pulse_center_time_s(spec)
    t_last = last_pulse_center_time_s(spec)
    return (
        nearest_time_index(t_s, t_first),
        nearest_time_index(t_s, t_last),
        t_first,
        t_last,
    )


def population_fractions_at_time(
    result: FiberCPAResult,
    z_index: int,
    time_index: int,
) -> dict[str, float]:
    """N₀…N₃ fractions at (z, t)."""
    p = result.populations
    return {
        "N₀": float(p.n0[z_index, time_index]),
        "N₁": float(p.n1[z_index, time_index]),
        "N₂": float(p.n2[z_index, time_index]),
        "N₃": float(p.n3[z_index, time_index]),
    }


def population_vs_z_at_time(
    result: FiberCPAResult,
    time_index: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Fractions vs z at one time index."""
    z = result.z_m
    p = result.populations
    return z, {
        "N₀": p.n0[:, time_index],
        "N₁": p.n1[:, time_index],
        "N₂": p.n2[:, time_index],
        "N₃": p.n3[:, time_index],
    }


def build_population_snapshot_figure(result, spec, *, plotly=True):
    """
    Bar charts of N₀…N₃ at fiber input/output when first and last pulses pass.

    Returns a plotly Figure if plotly=True, else (labels, data) for DPG.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    it0, it1, t0, t1 = population_snapshot_indices(result.t_s, spec)
    z_in, z_out = 0, result.z_m.size - 1

    panels = [
        (f"First pulse, z=0 ({t0 * 1e9:.2f} ns)", z_in, it0),
        (f"First pulse, z=L ({t0 * 1e9:.2f} ns)", z_out, it0),
        (f"Last pulse, z=0 ({t1 * 1e9:.2f} ns)", z_in, it1),
        (f"Last pulse, z=L ({t1 * 1e9:.2f} ns)", z_out, it1),
    ]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[p[0] for p in panels],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    levels = ["N₀", "N₁", "N₂", "N₃"]
    colors = ["#636EFA", "#EF553B", "#00CC96", "#AB63FA"]

    for k, (_title, iz, it) in enumerate(panels, start=1):
        row = (k - 1) // 2 + 1
        col = (k - 1) % 2 + 1
        fr = population_fractions_at_time(result, iz, it)
        vals = [fr[lv] for lv in levels]
        fig.add_trace(
            go.Bar(x=levels, y=vals, marker_color=colors, showlegend=False),
            row=row,
            col=col,
        )

    fig.update_yaxes(title_text="Population fraction", range=[0, 1.05])
    fig.update_layout(
        title_text="Level populations at first / last pulse (quasi-2L, N₁ ≈ 0)",
        height=520,
    )
    return fig


def build_population_vs_z_figure(result, spec):
    """N₂ and N₀ fractions along z at steady-state (pre-packet) and first pulse."""
    import plotly.graph_objects as go
    from laser_sim.physics.diagnostics import steady_state_time_index

    it_ss = steady_state_time_index(result.t_s, spec)
    it0, _, _, t0 = population_snapshot_indices(result.t_s, spec)
    z, at_ss = population_vs_z_at_time(result, it_ss)
    _, at_pulse = population_vs_z_at_time(result, it0)
    t_ss_s = float(result.t_s[it_ss])

    fig = go.Figure()
    for label, frac, t_ns in [
        ("N₂ steady state (pre-packet)", at_ss["N₂"], t_ss_s),
        ("N₂ first pulse", at_pulse["N₂"], t0),
        ("N₀ steady state (pre-packet)", at_ss["N₀"], t_ss_s),
        ("N₀ first pulse", at_pulse["N₀"], t0),
    ]:
        fig.add_trace(
            go.Scatter(
                x=z,
                y=frac,
                name=f"{label} ({t_ns * 1e9:.2f} ns)",
                mode="lines",
            )
        )
    fig.update_layout(
        title="Population fractions vs z (steady state before packet vs first pulse)",
        xaxis_title="z (m)",
        yaxis_title="Fraction",
        yaxis=dict(range=[0, 1.05]),
        height=400,
    )
    return fig
