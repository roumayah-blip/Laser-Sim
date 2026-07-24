"""JAX Stage 2: vmapped log-space propagation given a known gain field.

Replaces the failure mode in ``fiber_cpa.py``'s ``_signal_pass``/
``_propagate_ase_z_slab``, where ``P *= exp(g*dz)`` repeated across many
z-slabs can silently overflow to ``inf`` once the cumulative gain (in Nepers)
exceeds float64's ~709-nat range. Confirmed empirically: even a 10x finer
z-grid on that scheme still produces ``inf`` ASE energy (see
tests/test_jax_background.py) — the fix is not resolution, it's staying in
log space.

Here the state is ``y = log(P)``. The per-bin ODE::

    dP/dz = g(z) P + s(z)   (gain/loss + spontaneous source)

becomes::

    dy/dz = g(z) + s(z) * exp(-y)

``y`` can grow to any (finite, ordinary-magnitude) value without overflow —
only ``exp(y)`` would overflow, and that conversion back to linear power is
deferred to the very end (and clipped, so a physically enormous result comes
back as a large-but-finite float rather than ``inf``).

Every wavelength/channel bin is independent given a known gain field, so all
bins are solved in one batched call via ``jax.vmap`` — a single XLA-fused
kernel across the whole array, GPU-resident when a GPU is available.

This module only implements Stage 2 (propagation through a *known* gain
field). Stage 1 (solving for that gain field / the population dynamics that
produce it) is a separate, not-yet-implemented piece — see the project plan.
"""

from __future__ import annotations

import numpy as np

import diffrax
import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

_Y_CLIP = 700.0  # comfortably under float64's ~709-nat exp() overflow point


def _rhs(t, y, args):
    g_interp, s_interp = args
    g = g_interp.evaluate(t)
    s = s_interp.evaluate(t)
    y_safe = jnp.clip(y, -_Y_CLIP, _Y_CLIP)
    return g + s * jnp.exp(-y_safe)


def _solve_one_bin(p0_i, g_col, s_col, z0: float, z1: float, z_save, rtol, atol, max_steps):
    y0 = jnp.log(jnp.clip(p0_i, 1e-300, None))
    g_interp = diffrax.LinearInterpolation(ts=z_save, ys=g_col)
    s_interp = diffrax.LinearInterpolation(ts=z_save, ys=s_col)
    term = diffrax.ODETerm(_rhs)
    solver = diffrax.Kvaerno3()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term,
        solver,
        t0=z0,
        t1=z1,
        dt0=None,
        y0=y0,
        args=(g_interp, s_interp),
        saveat=diffrax.SaveAt(ts=z_save),
        stepsize_controller=controller,
        max_steps=max_steps,
    )
    return sol.ys


def propagate_log_space(
    z_m: np.ndarray,
    p_in_w_nm: np.ndarray,
    gain_np_m: np.ndarray,
    source_w_nm: np.ndarray | None = None,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 10_000,
) -> np.ndarray:
    """
    Propagate many independent wavelength/channel bins along z given a known gain field.

    Parameters
    ----------
    z_m: (nz,) z grid (m).
    p_in_w_nm: (n_bins,) input power density per bin at z=0 (W/nm or W).
    gain_np_m: (nz, n_bins) local gain coefficient g(z, bin) [Np/m], precomputed
        (e.g. from populations solved by an existing pump/signal pass).
    source_w_nm: (nz, n_bins) optional spontaneous-emission source density per
        bin (same units as p_in_w_nm, per z).

    Returns
    -------
    (nz, n_bins) power density P(z, bin) — always finite; a physically
    enormous result is clipped to a large-but-finite value rather than
    silently overflowing to ``inf``.
    """
    z_np = np.asarray(z_m, dtype=np.float64)
    if z_np.size < 2:
        raise ValueError("propagate_log_space needs at least 2 z points")
    z0, z1 = float(z_np[0]), float(z_np[-1])
    z_save = jnp.asarray(z_np)

    p0 = jnp.asarray(p_in_w_nm, dtype=jnp.float64)
    gain = jnp.asarray(gain_np_m, dtype=jnp.float64)
    if source_w_nm is None:
        source = jnp.zeros_like(gain)
    else:
        source = jnp.asarray(source_w_nm, dtype=jnp.float64)

    if gain.shape[0] != z_np.size:
        raise ValueError(f"gain_np_m has {gain.shape[0]} z-rows, expected {z_np.size}")
    if p0.shape[0] != gain.shape[1]:
        raise ValueError(
            f"p_in_w_nm has {p0.shape[0]} bins, gain_np_m has {gain.shape[1]}"
        )

    solve_batched = jax.jit(
        jax.vmap(
            lambda p0_i, g_col, s_col: _solve_one_bin(
                p0_i, g_col, s_col, z0, z1, z_save, rtol, atol, max_steps
            ),
            in_axes=(0, 1, 1),
            out_axes=1,
        )
    )
    logp = solve_batched(p0, gain, source)
    logp = jnp.clip(logp, -_Y_CLIP, _Y_CLIP)
    return np.asarray(jnp.exp(logp))


# ---------------------------------------------------------------------------
# Stage 1: self-consistent scalar N2(z) + forward pump/channel powers.
#
# Forward-propagating fields only (pump forward, signal/ASE channels
# forward): given every field's local intensity, N2(z) is a pure algebraic
# function (no Newton solve needed) — the same closed-form
# ``steady_state_n2_fraction_pump``-style rate balance, generalized to sum
# pump + every channel's stimulated absorption/emission. That algebraic N2(z)
# is embedded directly in the RHS of one combined ODE over
# [log P_pump, log P_channel_1, ..., log P_channel_M], integrated forward in
# z with diffrax — a well-posed initial value problem, no Newton/BVP/
# continuation required.
#
# Backward-propagating fields (backward pump, bidirectional ASE) turn this
# into a genuine two-point boundary value problem (the z=L boundary condition
# isn't known until the whole z-profile is solved) and are NOT implemented
# here yet — that needs the Newton-collocation + continuation/homotopy layer
# described in the project plan. This forward-only solve already fixes the
# reported bug for forward-pumped, forward-ASE-dominated amplifiers.
# ---------------------------------------------------------------------------


class Stage1ForwardResult:
    """z-resolved forward solve: pump, every channel, and the shared N2(z)."""

    def __init__(self, z_m, pump_power_w, channel_power_w, n2_fraction):
        self.z_m = z_m
        self.pump_power_w = pump_power_w  # (nz,)
        self.channel_power_w = channel_power_w  # (nz, n_ch)
        self.n2_fraction = n2_fraction  # (nz,)


def _stage1_forward_rhs(z, y, args):
    (
        n_tot,
        gamma_p,
        sigma_p,
        sigma_ep,
        hnu_p,
        a_pump,
        gamma_s,
        sigma_a_ch,
        sigma_e_ch,
        hnu_ch,
        a_signal,
        tau_21_s,
        is_ase,
        eta_guided,
        dlam_ch,
        ase_norm,
    ) = args
    y_safe = jnp.clip(y, -_Y_CLIP, _Y_CLIP)
    y_pump = y_safe[0]
    y_ch = y_safe[1:]

    i_pump = jnp.exp(y_pump) / a_pump
    i_ch = jnp.exp(y_ch) / a_signal

    w_p_abs = sigma_p * i_pump / hnu_p
    w_p_esa = sigma_ep * i_pump / hnu_p

    w_ch_abs = gamma_s * sigma_a_ch * i_ch / hnu_ch
    w_ch_se = gamma_s * sigma_e_ch * i_ch / hnu_ch

    w_abs_total = w_p_abs + jnp.sum(w_ch_abs)
    w_deplete_total = w_p_esa + jnp.sum(w_ch_se) + 1.0 / tau_21_s

    n2 = n_tot * w_abs_total / jnp.maximum(w_abs_total + w_deplete_total, 1e-60)
    n0 = n_tot - n2

    dy_pump = -gamma_p * (sigma_p * n0 - sigma_ep * n2)
    dy_ch = gamma_s * (sigma_e_ch * n2 - sigma_a_ch * n0)

    # Spontaneous source (ASE bins only), same normalization convention as
    # four_level.spontaneous_power_w_per_nm, as an affine term s*exp(-y) so
    # ASE bins can build up from a zero/near-zero seed without a singularity.
    spont_density = (
        eta_guided * gamma_s * jnp.maximum(n2, 0.0) * hnu_ch * sigma_e_ch
        / jnp.maximum(tau_21_s * ase_norm, 1e-60)
    )
    src = jnp.where(is_ase, spont_density * a_signal * dlam_ch, 0.0)
    dy_ch = dy_ch + src * jnp.exp(-jnp.clip(y_ch, -_Y_CLIP, _Y_CLIP))

    return jnp.concatenate([dy_pump[None], dy_ch])


def solve_stage1_forward(
    z_m: np.ndarray,
    *,
    n_tot: float,
    gamma_p: float,
    sigma_p: float,
    sigma_ep: float,
    hnu_p: float,
    a_pump: float,
    p_pump_in_w: float,
    gamma_s: float,
    channel_wavelengths_nm: np.ndarray,
    sigma_a_ch: np.ndarray,
    sigma_e_ch: np.ndarray,
    a_signal: float,
    p_ch_in_w: np.ndarray,
    tau_21_s: float,
    is_ase: np.ndarray,
    eta_guided: float,
    dlam_ch_nm: np.ndarray,
    material_hnu_ch: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 20_000,
) -> Stage1ForwardResult:
    """
    Self-consistent forward solve: shared N2(z) + pump + every channel's power.

    All channels (real CW signal channels, pulsed signals already converted
    to their CW-average-power equivalent, and ASE pseudo-channels flagged via
    ``is_ase``) compete for the same local N2(z) — N2 at each z is computed
    algebraically from every field's *combined* local intensity, not solved
    one channel at a time. Because only forward fields are modeled, this is a
    single well-posed IVP integrated once in z (no Newton/BVP/continuation).
    """
    z_np = np.asarray(z_m, dtype=np.float64)
    z0, z1 = float(z_np[0]), float(z_np[-1])
    z_save = jnp.asarray(z_np)

    is_ase_arr = np.asarray(is_ase, dtype=bool)
    ase_wl = np.asarray(channel_wavelengths_nm)[is_ase_arr]
    ase_sigma_e = np.asarray(sigma_e_ch)[is_ase_arr]
    if ase_wl.size >= 2:
        ase_norm = float(np.trapezoid(ase_sigma_e, ase_wl))
    else:
        ase_norm = float(np.sum(ase_sigma_e)) if ase_sigma_e.size else 1.0
    if ase_norm <= 0:
        ase_norm = 1.0

    args = (
        float(n_tot),
        float(gamma_p),
        float(sigma_p),
        float(sigma_ep),
        float(hnu_p),
        float(a_pump),
        float(gamma_s),
        jnp.asarray(sigma_a_ch, dtype=jnp.float64),
        jnp.asarray(sigma_e_ch, dtype=jnp.float64),
        jnp.asarray(material_hnu_ch, dtype=jnp.float64),
        float(a_signal),
        float(tau_21_s),
        jnp.asarray(is_ase_arr),
        float(eta_guided),
        jnp.asarray(dlam_ch_nm, dtype=jnp.float64),
        ase_norm,
    )

    y0_pump = jnp.log(jnp.clip(jnp.asarray(float(p_pump_in_w)), 1e-300, None))
    y0_ch = jnp.log(jnp.clip(jnp.asarray(p_ch_in_w, dtype=jnp.float64), 1e-300, None))
    y0 = jnp.concatenate([y0_pump[None], y0_ch])

    term = diffrax.ODETerm(_stage1_forward_rhs)
    solver = diffrax.Kvaerno3()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    solve = jax.jit(
        lambda y0_: diffrax.diffeqsolve(
            term,
            solver,
            t0=z0,
            t1=z1,
            dt0=None,
            y0=y0_,
            args=args,
            saveat=diffrax.SaveAt(ts=z_save),
            stepsize_controller=controller,
            max_steps=max_steps,
        ).ys
    )
    y_z = solve(y0)
    y_z = jnp.clip(y_z, -_Y_CLIP, _Y_CLIP)
    pump_power = np.asarray(jnp.exp(y_z[:, 0]))
    channel_power = np.asarray(jnp.exp(y_z[:, 1:]))

    # Recompute N2(z) from the converged fields for reporting (same algebra as the RHS).
    i_pump = pump_power / a_pump
    i_ch = channel_power / a_signal
    w_p_abs = sigma_p * i_pump / hnu_p
    w_p_esa = sigma_ep * i_pump / hnu_p
    w_ch_abs = gamma_s * np.asarray(sigma_a_ch)[None, :] * i_ch / np.asarray(material_hnu_ch)[None, :]
    w_ch_se = gamma_s * np.asarray(sigma_e_ch)[None, :] * i_ch / np.asarray(material_hnu_ch)[None, :]
    w_abs_total = w_p_abs + np.sum(w_ch_abs, axis=1)
    w_deplete_total = w_p_esa + np.sum(w_ch_se, axis=1) + 1.0 / tau_21_s
    n2_fraction = w_abs_total / np.maximum(w_abs_total + w_deplete_total, 1e-60)

    return Stage1ForwardResult(z_np, pump_power, channel_power, n2_fraction)
