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

from functools import partial

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


# ---------------------------------------------------------------------------
# Stage 1, bidirectional: backward pump + bidirectional ASE via a Newton
# shooting method.
#
# Backward-propagating fields have their boundary condition at z=L, not z=0,
# so the combined system (forward + backward fields, all sharing one N2(z))
# is a genuine two-point BVP. This solves it by shooting: guess every
# backward field's value AT z=0, integrate the WHOLE system forward to z=L,
# and Newton-correct the z=0 guess so the computed z=L values match the known
# z=L boundary conditions. ``jax.jacobian`` differentiates straight through
# the diffrax integration for an exact residual Jacobian (no finite
# differencing). Continuation ramps the pump/seed power up from a small
# fraction of the target in adaptive steps (halving on Newton failure) so
# Newton always starts from an already-converged, nearby solution rather
# than a cold, badly-conditioned guess.
#
# Sign convention (derived from flipping the propagation direction into an
# equivalent z-increasing "forward" ODE): forward fields gain +g(z) per unit
# z; backward fields, evaluated at increasing z, evolve as -g(z) (and pump
# loss flips sign the same way) — this is a bookkeeping choice checked
# against the existing CPU backward-pump test in
# tests/test_jax_background.py (p_bwd must be largest at z=L, decaying
# toward z=0).
# ---------------------------------------------------------------------------


class Stage1BidirectionalResult:
    def __init__(
        self,
        z_m,
        pump_fwd_w,
        pump_bwd_w,
        channel_fwd_w,
        channel_bwd_w,
        n2_fraction,
        newton_iters,
        continuation_steps,
        converged,
    ):
        self.z_m = z_m
        self.pump_fwd_w = pump_fwd_w
        self.pump_bwd_w = pump_bwd_w
        self.channel_fwd_w = channel_fwd_w  # (nz, n_ch)
        self.channel_bwd_w = channel_bwd_w  # (nz, n_ch), meaningful only where is_ase
        self.n2_fraction = n2_fraction
        self.newton_iters = newton_iters
        self.continuation_steps = continuation_steps
        self.converged = converged


def _make_bidir_rhs(n_ch: int):
    """n_ch must be a static Python int (used for slice bounds, not traced)."""

    def _bidir_rhs(z, y, args):
        (
            n_tot, gamma_p, sigma_p, sigma_ep, hnu_p, a_pump,
            gamma_s, sigma_a_ch, sigma_e_ch, hnu_ch, a_signal, tau_21_s,
            is_ase, eta_guided, dlam_ch, ase_norm,
        ) = args
        y_safe = jnp.clip(y, -_Y_CLIP, _Y_CLIP)
        y_pf = y_safe[0]
        y_cf = y_safe[1 : 1 + n_ch]
        y_pb = y_safe[1 + n_ch]
        y_cb = y_safe[2 + n_ch : 2 + 2 * n_ch]
        return _bidir_rhs_body(
            y_pf, y_cf, y_pb, y_cb, n_tot, gamma_p, sigma_p, sigma_ep, hnu_p, a_pump,
            gamma_s, sigma_a_ch, sigma_e_ch, hnu_ch, a_signal, tau_21_s,
            is_ase, eta_guided, dlam_ch, ase_norm,
        )

    return _bidir_rhs


def _bidir_rhs_body(
    y_pf, y_cf, y_pb, y_cb, n_tot, gamma_p, sigma_p, sigma_ep, hnu_p, a_pump,
    gamma_s, sigma_a_ch, sigma_e_ch, hnu_ch, a_signal, tau_21_s,
    is_ase, eta_guided, dlam_ch, ase_norm,
):

    i_pf = jnp.exp(y_pf) / a_pump
    i_pb = jnp.exp(y_pb) / a_pump
    i_cf = jnp.exp(y_cf) / a_signal
    i_cb = jnp.where(is_ase, jnp.exp(y_cb), 0.0) / a_signal

    w_p_abs = sigma_p * (i_pf + i_pb) / hnu_p
    w_p_esa = sigma_ep * (i_pf + i_pb) / hnu_p
    w_ch_abs = gamma_s * sigma_a_ch * (i_cf + i_cb) / hnu_ch
    w_ch_se = gamma_s * sigma_e_ch * (i_cf + i_cb) / hnu_ch

    w_abs_total = w_p_abs + jnp.sum(w_ch_abs)
    w_deplete_total = w_p_esa + jnp.sum(w_ch_se) + 1.0 / tau_21_s
    n2 = n_tot * w_abs_total / jnp.maximum(w_abs_total + w_deplete_total, 1e-60)
    n0 = n_tot - n2

    spont_density = (
        eta_guided * gamma_s * jnp.maximum(n2, 0.0) * hnu_ch * sigma_e_ch
        / jnp.maximum(tau_21_s * ase_norm, 1e-60)
    )
    src = jnp.where(is_ase, spont_density * a_signal * dlam_ch, 0.0)

    g_ch = gamma_s * (sigma_e_ch * n2 - sigma_a_ch * n0)
    alpha_p = gamma_p * (sigma_p * n0 - sigma_ep * n2)

    dy_pf = -alpha_p
    dy_cf = g_ch + src * jnp.exp(-jnp.clip(y_cf, -_Y_CLIP, _Y_CLIP))
    dy_pb = alpha_p
    dy_cb_raw = -g_ch - src * jnp.exp(-jnp.clip(y_cb, -_Y_CLIP, _Y_CLIP))
    dy_cb = jnp.where(is_ase, dy_cb_raw, 0.0)

    return jnp.concatenate([dy_pf[None], dy_cf, dy_pb[None], dy_cb])


@partial(jax.jit, static_argnames=("n_ch", "max_steps"))
def _integrate_bidir(y0, z0, z1, z_save, args, n_ch, rtol, atol, max_steps):
    term = diffrax.ODETerm(_make_bidir_rhs(n_ch))
    solver = diffrax.Kvaerno3()
    controller = diffrax.PIDController(rtol=rtol, atol=atol)
    sol = diffrax.diffeqsolve(
        term, solver, t0=z0, t1=z1, dt0=None, y0=y0, args=args,
        saveat=diffrax.SaveAt(ts=z_save), stepsize_controller=controller, max_steps=max_steps,
        throw=False,
    )
    # throw=False turns a failed solve (e.g. max_steps reached from a bad
    # continuation guess) into a result code instead of an exception, so a
    # failed step reads as NaN residuals -- the Newton loop then reports
    # non-convergence for this step instead of crashing, and continuation
    # retries with a smaller power step.
    ok = sol.result == diffrax.RESULTS.successful
    return jnp.where(ok, sol.ys, jnp.nan)


@partial(jax.jit, static_argnames=("n_ch", "max_steps"))
def _shoot_residual(b0, y_fwd0, z0, z1, z_save, args, n_ch, b_target, rtol, atol, max_steps):
    y0 = jnp.concatenate([y_fwd0, b0])
    y_z = _integrate_bidir(y0, z0, z1, z_save, args, n_ch, rtol, atol, max_steps)
    y_L = jnp.clip(y_z[-1], -_Y_CLIP, _Y_CLIP)
    b_computed_at_L = jnp.concatenate([y_L[1 + n_ch][None], y_L[2 + n_ch : 2 + 2 * n_ch]])
    return b_computed_at_L - b_target


@partial(jax.jit, static_argnames=("n_ch", "max_steps"))
def _shoot_jacobian(b0, y_fwd0, z0, z1, z_save, args, n_ch, b_target, rtol, atol, max_steps):
    return jax.jacobian(_shoot_residual, argnums=0)(
        b0, y_fwd0, z0, z1, z_save, args, n_ch, b_target, rtol, atol, max_steps
    )


def _newton_solve_b0(b0_guess, y_fwd0, z0, z1, z_save, args, n_ch, b_target, rtol, atol, max_steps, *, tol=1e-6, max_newton_iter=20):
    # _shoot_residual/_shoot_jacobian are module-level jitted functions (not
    # recreated per call), so XLA compilation is cached and reused across
    # every Newton iteration AND every continuation step within a run --
    # recreating the jit wrapper per call here was the original bottleneck
    # (13s/call uncompiled vs ~0.004s cached, confirmed by direct timing).
    b0 = b0_guess
    for it in range(max_newton_iter):
        f = _shoot_residual(b0, y_fwd0, z0, z1, z_save, args, n_ch, b_target, rtol, atol, max_steps)
        res_norm = float(jnp.max(jnp.abs(f)))
        if not np.isfinite(res_norm):
            return b0, it, False
        if res_norm < tol:
            return b0, it, True
        j = _shoot_jacobian(b0, y_fwd0, z0, z1, z_save, args, n_ch, b_target, rtol, atol, max_steps)
        try:
            step = jnp.linalg.solve(j, f)
        except Exception:
            return b0, it, False
        if not bool(jnp.all(jnp.isfinite(step))):
            return b0, it, False
        b0 = b0 - step
    f = _shoot_residual(b0, y_fwd0, z0, z1, z_save, args, n_ch, b_target, rtol, atol, max_steps)
    return b0, max_newton_iter, bool(jnp.max(jnp.abs(f)) < tol)


def solve_stage1_bidirectional(
    z_m: np.ndarray,
    *,
    n_tot: float,
    gamma_p: float,
    sigma_p: float,
    sigma_ep: float,
    hnu_p: float,
    a_pump: float,
    p_pump_fwd_in_w: float,
    p_pump_bwd_in_w: float,
    gamma_s: float,
    channel_wavelengths_nm: np.ndarray,
    sigma_a_ch: np.ndarray,
    sigma_e_ch: np.ndarray,
    a_signal: float,
    p_ch_fwd_in_w: np.ndarray,
    tau_21_s: float,
    is_ase: np.ndarray,
    eta_guided: float,
    dlam_ch_nm: np.ndarray,
    material_hnu_ch: np.ndarray,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    max_steps: int = 20_000,
    p_floor_w: float = 1e-12,
    n_continuation_steps: int = 6,
    tol: float = 1e-6,
) -> Stage1BidirectionalResult:
    """
    Self-consistent bidirectional solve: shared N2(z) with forward pump/
    channels AND backward pump + bidirectional ASE, via Newton shooting.

    Continuation (adaptive): starts at a small fraction of the target
    pump/seed powers (where Newton is well-conditioned, near the trivial
    near-zero-field solution) and ramps toward the requested powers in
    ``n_continuation_steps`` steps, using each converged step's backward
    z=0 values as the next step's initial guess. On Newton non-convergence,
    the step is halved and retried.
    """
    z_np = np.asarray(z_m, dtype=np.float64)
    z0, z1 = float(z_np[0]), float(z_np[-1])
    z_save = jnp.asarray(z_np)
    n_ch = int(np.asarray(channel_wavelengths_nm).size)
    is_ase_arr = np.asarray(is_ase, dtype=bool)

    ase_wl = np.asarray(channel_wavelengths_nm)[is_ase_arr]
    ase_sigma_e = np.asarray(sigma_e_ch)[is_ase_arr]
    if ase_wl.size >= 2:
        ase_norm = float(np.trapezoid(ase_sigma_e, ase_wl))
    else:
        ase_norm = float(np.sum(ase_sigma_e)) if ase_sigma_e.size else 1.0
    if ase_norm <= 0:
        ase_norm = 1.0

    def make_args(p_pump_fwd_w, p_pump_bwd_w, p_ch_fwd_w):
        return (
            float(n_tot), float(gamma_p), float(sigma_p), float(sigma_ep), float(hnu_p), float(a_pump),
            float(gamma_s), jnp.asarray(sigma_a_ch, jnp.float64), jnp.asarray(sigma_e_ch, jnp.float64),
            jnp.asarray(material_hnu_ch, jnp.float64), float(a_signal), float(tau_21_s),
            jnp.asarray(is_ase_arr), float(eta_guided), jnp.asarray(dlam_ch_nm, jnp.float64), ase_norm,
        ), p_pump_fwd_w, p_pump_bwd_w, p_ch_fwd_w

    b_target = jnp.concatenate(
        [
            jnp.log(jnp.clip(jnp.asarray(float(p_pump_bwd_in_w)), p_floor_w, None))[None],
            jnp.full((n_ch,), np.log(p_floor_w)),
        ]
    )
    b0 = b_target  # initial guess: assume backward fields near their (small) target everywhere

    fractions = np.linspace(1.0 / n_continuation_steps, 1.0, n_continuation_steps)
    total_newton_iters = 0
    steps_taken = 0
    converged_all = True
    idx = 0
    while idx < fractions.size:
        frac = fractions[idx]
        p_pf = max(p_pump_fwd_in_w * frac, p_floor_w)
        p_pb = max(p_pump_bwd_in_w * frac, p_floor_w) if p_pump_bwd_in_w > 0 else p_floor_w
        p_cf = np.clip(np.asarray(p_ch_fwd_in_w, dtype=np.float64) * frac, p_floor_w, None)

        args, _, _, _ = make_args(p_pf, p_pb, p_cf)
        y_fwd0 = jnp.concatenate(
            [jnp.log(jnp.asarray(p_pf))[None], jnp.log(jnp.asarray(p_cf))]
        )
        b_target_step = b_target.at[0].set(float(np.log(max(p_pb, p_floor_w))))

        b0_new, n_iter, ok = _newton_solve_b0(
            b0, y_fwd0, z0, z1, z_save, args, n_ch, b_target_step, rtol, atol, max_steps, tol=tol
        )
        total_newton_iters += n_iter
        steps_taken += 1
        if ok:
            b0 = b0_new
            idx += 1
        else:
            # Continuation failure: halve the remaining step by inserting a
            # finer fraction between the last converged point and this target.
            prev_frac = fractions[idx - 1] if idx > 0 else 0.0
            mid = 0.5 * (prev_frac + frac)
            if abs(mid - frac) < 1e-4:
                converged_all = False
                break
            fractions = np.insert(fractions, idx, mid)

    y_fwd0 = jnp.concatenate(
        [
            jnp.log(jnp.asarray(float(p_pump_fwd_in_w)))[None],
            jnp.log(jnp.clip(jnp.asarray(p_ch_fwd_in_w, jnp.float64), p_floor_w, None)),
        ]
    )
    args, _, _, _ = make_args(p_pump_fwd_in_w, p_pump_bwd_in_w, p_ch_fwd_in_w)
    y0_final = jnp.concatenate([y_fwd0, b0])
    y_z = _integrate_bidir(y0_final, z0, z1, z_save, args, n_ch, rtol, atol, max_steps)
    y_z = jnp.clip(y_z, -_Y_CLIP, _Y_CLIP)

    pump_fwd = np.asarray(jnp.exp(y_z[:, 0]))
    ch_fwd = np.asarray(jnp.exp(y_z[:, 1 : 1 + n_ch]))
    pump_bwd = np.asarray(jnp.exp(y_z[:, 1 + n_ch]))
    ch_bwd = np.asarray(jnp.exp(y_z[:, 2 + n_ch : 2 + 2 * n_ch]))

    i_pf = pump_fwd / a_pump
    i_pb = pump_bwd / a_pump
    i_cf = ch_fwd / a_signal
    i_cb = np.where(is_ase_arr[None, :], ch_bwd, 0.0) / a_signal
    w_p_abs = sigma_p * (i_pf + i_pb) / hnu_p
    w_p_esa = sigma_ep * (i_pf + i_pb) / hnu_p
    w_ch_abs = gamma_s * np.asarray(sigma_a_ch)[None, :] * (i_cf + i_cb) / np.asarray(material_hnu_ch)[None, :]
    w_ch_se = gamma_s * np.asarray(sigma_e_ch)[None, :] * (i_cf + i_cb) / np.asarray(material_hnu_ch)[None, :]
    w_abs_total = w_p_abs + np.sum(w_ch_abs, axis=1)
    w_deplete_total = w_p_esa + np.sum(w_ch_se, axis=1) + 1.0 / tau_21_s
    n2_fraction = w_abs_total / np.maximum(w_abs_total + w_deplete_total, 1e-60)

    return Stage1BidirectionalResult(
        z_np, pump_fwd, pump_bwd, ch_fwd, ch_bwd, n2_fraction,
        total_newton_iters, steps_taken, converged_all,
    )
