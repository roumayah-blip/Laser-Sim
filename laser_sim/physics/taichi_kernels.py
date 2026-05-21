"""
Taichi GPU kernels for the Yb fiber CPA two-pass solver.

Populations in GPU fields are stored as fractions in [0, 1]; multiply by N_tot
when forming rates and gain coefficients.

Kernels are registered after ``init_taichi()`` — not at import time.
"""

import numpy as np

try:
    import taichi as ti
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Taichi is required for the GPU backend. Install with: pip install taichi"
    ) from exc

_ti_initialized = False
_kernels_registered = False
_last_arch: str | None = None

# Populated by _register_kernels() after ti.init()
kernel_pump_advance_z = None
kernel_pump_bwd_advance_z = None
kernel_qss_march_z = None
kernel_gain_slab = None
kernel_compute_gain = None
kernel_advance_signal_slab = None
kernel_spontaneous_source = None
kernel_ase_fwd_slab = None
kernel_ase_bwd_slab = None
kernel_rk4_populations = None
kernel_causal_signal_slab = None


def _runtime_materialized() -> bool:
    try:
        return bool(ti.lang.impl.get_runtime().materialized)
    except Exception:
        return False


def abandon_taichi_runtime() -> None:
    """Clear Python-side Taichi state without calling ``ti.reset()`` (safe if CUDA context is dead)."""
    global _ti_initialized, _kernels_registered, _last_arch
    global kernel_pump_advance_z, kernel_pump_bwd_advance_z, kernel_qss_march_z
    global kernel_gain_slab, kernel_compute_gain, kernel_advance_signal_slab
    global kernel_spontaneous_source
    global kernel_ase_fwd_slab, kernel_ase_bwd_slab, kernel_rk4_populations
    global kernel_causal_signal_slab

    _ti_initialized = False
    _kernels_registered = False
    _last_arch = None
    _cache.clear()

    kernel_pump_advance_z = None
    kernel_pump_bwd_advance_z = None
    kernel_qss_march_z = None
    kernel_gain_slab = None
    kernel_compute_gain = None
    kernel_advance_signal_slab = None
    kernel_spontaneous_source = None
    kernel_ase_fwd_slab = None
    kernel_ase_bwd_slab = None
    kernel_rk4_populations = None
    kernel_causal_signal_slab = None


def reset_taichi_runtime() -> None:
    """Best-effort Taichi shutdown, then always drop local field/kernel handles."""
    if _ti_initialized or _runtime_materialized():
        try:
            ti.sync()
            ti.reset()
        except Exception:
            pass
    abandon_taichi_runtime()


def init_taichi(arch: str = "cuda", fp: str = "f32", *, force_reinit: bool = False) -> None:
    """Initialize Taichi runtime (call explicitly; not at import)."""
    global _ti_initialized, _last_arch

    if force_reinit:
        reset_taichi_runtime()
    elif _ti_initialized:
        try:
            ti.sync()
            _register_kernels()
            return
        except Exception:
            abandon_taichi_runtime()

    if _runtime_materialized() and not _ti_initialized:
        try:
            ti.sync()
            ti.reset()
        except Exception:
            abandon_taichi_runtime()

    if not _ti_initialized:
        dtype = ti.f32 if fp == "f32" else ti.f64
        ti.init(arch=getattr(ti, arch), default_fp=dtype, device_memory_fraction=0.80)
        _ti_initialized = True
        _last_arch = arch
    _register_kernels()


def active_arch_name() -> str | None:
    """Return Taichi arch string after ``init_taichi``, else None."""
    if not _ti_initialized:
        return None
    return str(ti.cfg.arch)


class FieldCache:
    """Holds ti.field objects; reallocated when grid shape changes."""

    def __init__(self) -> None:
        self.shape: tuple[int, int, int] | None = None
        self.p_pump_fwd = None
        self.p_pump_bwd = None
        self.p_sig = None
        self.p_ase_f = None
        self.p_ase_b = None
        self.n0 = None
        self.n2 = None
        self.n3 = None
        self.gain = None
        self.spont_src = None
        self.dt_arr = None
        self.sigma_e = None
        self.sigma_a = None
        self.hnu = None
        self.dlam = None

    def clear(self) -> None:
        self.shape = None
        for name in (
            "p_pump_fwd",
            "p_pump_bwd",
            "p_sig",
            "p_ase_f",
            "p_ase_b",
            "n0",
            "n2",
            "n3",
            "gain",
            "spont_src",
            "dt_arr",
            "sigma_e",
            "sigma_a",
            "hnu",
            "dlam",
        ):
            setattr(self, name, None)

    def ensure(self, n_z: int, n_t: int, n_lam: int) -> None:
        if not _ti_initialized:
            raise RuntimeError("Call init_taichi() before FieldCache.ensure()")
        shape = (n_z, n_t, n_lam)
        if self.shape == shape and self.sigma_e is not None:
            return
        self.shape = shape
        self.p_pump_fwd = ti.field(dtype=ti.f32, shape=(n_z, n_t))
        self.p_pump_bwd = ti.field(dtype=ti.f32, shape=(n_z, n_t))
        self.p_sig = ti.field(dtype=ti.f32, shape=(n_z, n_t, n_lam))
        self.p_ase_f = ti.field(dtype=ti.f32, shape=(n_z, n_t, n_lam))
        self.p_ase_b = ti.field(dtype=ti.f32, shape=(n_z, n_t, n_lam))
        self.n0 = ti.field(dtype=ti.f32, shape=(n_z, n_t))
        self.n2 = ti.field(dtype=ti.f32, shape=(n_z, n_t))
        self.n3 = ti.field(dtype=ti.f32, shape=(n_z, n_t))
        self.gain = ti.field(dtype=ti.f32, shape=(n_z, n_t, n_lam))
        self.spont_src = ti.field(dtype=ti.f32, shape=(n_z, n_t, n_lam))
        self.dt_arr = ti.field(dtype=ti.f32, shape=(n_t,))
        self.sigma_e = ti.field(dtype=ti.f32, shape=(n_lam,))
        self.sigma_a = ti.field(dtype=ti.f32, shape=(n_lam,))
        self.hnu = ti.field(dtype=ti.f32, shape=(n_lam,))
        self.dlam = ti.field(dtype=ti.f32, shape=(n_lam,))


_cache = FieldCache()


def _register_kernels() -> None:
    global _kernels_registered
    global kernel_pump_advance_z, kernel_pump_bwd_advance_z, kernel_qss_march_z
    global kernel_gain_slab, kernel_compute_gain, kernel_advance_signal_slab
    global kernel_spontaneous_source
    global kernel_ase_fwd_slab, kernel_ase_bwd_slab, kernel_rk4_populations
    global kernel_causal_signal_slab
    if _kernels_registered:
        return

    @ti.kernel
    def _kernel_pump_advance_z(iz: int, sigma_p: float, gamma_p: float, dz: float, n_tot: float):
        for it in range(_cache.p_pump_fwd.shape[1]):
            n0_d = _cache.n0[iz, it] * n_tot
            alpha = gamma_p * sigma_p * ti.max(n0_d, 0.0)
            _cache.p_pump_fwd[iz + 1, it] = _cache.p_pump_fwd[iz, it] * ti.exp(-alpha * dz)

    @ti.kernel
    def _kernel_pump_bwd_advance_z(iz: int, sigma_p: float, gamma_p: float, dz: float, n_tot: float):
        for it in range(_cache.p_pump_bwd.shape[1]):
            n0_d = _cache.n0[iz, it] * n_tot
            alpha = gamma_p * sigma_p * ti.max(n0_d, 0.0)
            _cache.p_pump_bwd[iz, it] = _cache.p_pump_bwd[iz + 1, it] * ti.exp(-alpha * dz)

    @ti.kernel
    def _kernel_qss_march_z(
        iz: int,
        n_tot: float,
        a_pump: float,
        gamma_p: float,
        sigma_p: float,
        sigma_ep: float,
        hnu_p: float,
        tau_32: float,
        tau_21: float,
    ):
        n0_i = 1.0
        n2_i = 0.0
        n3_i = 0.0
        nt = _cache.p_pump_fwd.shape[1]
        _cache.n0[iz, 0] = n0_i
        _cache.n2[iz, 0] = n2_i
        _cache.n3[iz, 0] = n3_i
        for it in range(1, nt):
            dt_i = _cache.dt_arr[it]
            ip = (
                (
                    _cache.p_pump_fwd[iz, it - 1]
                    + _cache.p_pump_bwd[iz, it - 1]
                    + _cache.p_pump_fwd[iz, it]
                    + _cache.p_pump_bwd[iz, it]
                )
                * 0.5
                / a_pump
            )
            w_p = gamma_p * ip / hnu_p
            w_abs = w_p * sigma_p
            w_esa = w_p * sigma_ep
            n0_i = ti.max(1.0 - n2_i - n3_i, 0.0)
            n3_ss = w_abs * n0_i * tau_32 / (1.0 + w_esa * tau_32)
            _wt = w_abs * tau_21
            n2_ss = _wt / (1.0 + _wt)
            n2i = n2_ss - (n2_ss - n2_i) * ti.exp(-dt_i / tau_21)
            n2_i = ti.min(n2i, 1.0 - n3_ss)
            n3_i = n3_ss
            n0_i = ti.max(1.0 - n2_i - n3_i, 0.0)
            _cache.n0[iz, it] = n0_i
            _cache.n2[iz, it] = n2_i
            _cache.n3[iz, it] = n3_i

    @ti.kernel
    def _kernel_compute_gain(iz: int, gamma_s: float, n_tot: float):
        """Fill gain[iz] from populations; does not touch signal or ASE power."""
        n_t = _cache.n0.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        for it, ilam in ti.ndrange(n_t, n_lam):
            n0_d = ti.min(ti.max(_cache.n0[iz, it] * n_tot, 0.0), n_tot)
            n2_d = ti.min(ti.max(_cache.n2[iz, it] * n_tot, 0.0), n_tot)
            _cache.gain[iz, it, ilam] = gamma_s * (
                _cache.sigma_e[ilam] * n2_d - _cache.sigma_a[ilam] * n0_d
            )

    @ti.kernel
    def _kernel_advance_signal_slab(iz: int, dz: float):
        """Advance signal only using precomputed gain[iz]."""
        n_t = _cache.n0.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        for it, ilam in ti.ndrange(n_t, n_lam):
            gdz = ti.min(ti.max(_cache.gain[iz, it, ilam] * dz, -50.0), 50.0)
            p_in = _cache.p_sig[iz, it, ilam]
            _cache.p_sig[iz + 1, it, ilam] = ti.max(p_in * ti.exp(gdz), 0.0)

    @ti.kernel
    def _kernel_gain_slab(iz: int, gamma_s: float, dz: float, n_tot: float):
        """Signal pass only: compute g[iz] and advance P_sig (never use inside ASE loops)."""
        n_t = _cache.n0.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        for it, ilam in ti.ndrange(n_t, n_lam):
            n0_d = ti.min(ti.max(_cache.n0[iz, it] * n_tot, 0.0), n_tot)
            n2_d = ti.min(ti.max(_cache.n2[iz, it] * n_tot, 0.0), n_tot)
            g = gamma_s * (
                _cache.sigma_e[ilam] * n2_d - _cache.sigma_a[ilam] * n0_d
            )
            _cache.gain[iz, it, ilam] = g
            gdz = ti.min(ti.max(g * dz, -50.0), 50.0)
            p_in = _cache.p_sig[iz, it, ilam]
            _cache.p_sig[iz + 1, it, ilam] = ti.max(p_in * ti.exp(gdz), 0.0)

    @ti.kernel
    def _kernel_spontaneous_source(
        iz: int,
        eta_guided: float,
        gamma_s: float,
        tau_21: float,
        sigma_e_norm: float,
        n_tot: float,
    ):
        """Time-averaged spontaneous source at z=iz (matches CPU spont_row/nt average)."""
        n_t = _cache.n2.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        n2_sum = 0.0
        for it in range(n_t):
            n2_sum += ti.min(ti.max(_cache.n2[iz, it], 0.0), 1.0)
        n2_mean_d = (n2_sum / ti.max(float(n_t), 1.0)) * n_tot
        for it, ilam in ti.ndrange(n_t, n_lam):
            _cache.spont_src[iz, it, ilam] = (
                eta_guided
                * gamma_s
                * n2_mean_d
                * _cache.hnu[ilam]
                * _cache.sigma_e[ilam]
                / (tau_21 * sigma_e_norm)
            )

    @ti.kernel
    def _kernel_ase_fwd_slab(iz: int, dz: float):
        n_t = _cache.n2.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        for it, ilam in ti.ndrange(n_t, n_lam):
            gdz = ti.min(ti.max(_cache.gain[iz, it, ilam] * dz, -50.0), 50.0)
            src = _cache.spont_src[iz, it, ilam] * dz
            p_in = _cache.p_ase_f[iz, it, ilam]
            _cache.p_ase_f[iz + 1, it, ilam] = ti.max(p_in * ti.exp(gdz) + src, 0.0)

    @ti.kernel
    def _kernel_ase_bwd_slab(iz: int, dz: float):
        n_t = _cache.n2.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        for it, ilam in ti.ndrange(n_t, n_lam):
            gdz = ti.min(ti.max(_cache.gain[iz, it, ilam] * dz, -50.0), 50.0)
            src = _cache.spont_src[iz, it, ilam] * dz
            p_out = _cache.p_ase_b[iz + 1, it, ilam]
            _cache.p_ase_b[iz, it, ilam] = ti.max(p_out * ti.exp(gdz) + src, 0.0)

    @ti.kernel
    def _kernel_rk4_populations(
        iz: int,
        n_tot: float,
        dt_travel: float,
        gamma_s: float,
        a_signal: float,
        tau_21: float,
        sig_threshold: float,
        include_ase: int,
    ):
        n_t = _cache.n0.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        for it in range(n_t):
            w_se = 0.0
            w_abs = 0.0
            sig_w = 0.0
            for ilam in range(n_lam):
                p_avg = 0.5 * (
                    _cache.p_sig[iz, it, ilam] + _cache.p_sig[iz + 1, it, ilam]
                )
                sig_w += p_avg * _cache.dlam[ilam]

            include_signal = sig_w >= sig_threshold
            if include_signal:
                for ilam in range(n_lam):
                    p_avg = 0.5 * (
                        _cache.p_sig[iz, it, ilam] + _cache.p_sig[iz + 1, it, ilam]
                    )
                    i_s = p_avg / a_signal
                    fac = gamma_s * _cache.dlam[ilam] / _cache.hnu[ilam]
                    w_se += _cache.sigma_e[ilam] * i_s * fac
                    w_abs += _cache.sigma_a[ilam] * i_s * fac

            if include_ase != 0:
                for ilam in range(n_lam):
                    p_ase = _cache.p_ase_f[iz, it, ilam] + _cache.p_ase_b[iz, it, ilam]
                    i_ase = p_ase / a_signal
                    fac = gamma_s * _cache.dlam[ilam] / _cache.hnu[ilam]
                    w_se += _cache.sigma_e[ilam] * i_ase * fac

            f0 = _cache.n0[iz, it]
            f2 = _cache.n2[iz, it]
            dt = dt_travel

            n0 = f0 * n_tot
            n2 = f2 * n_tot

            dn2_k1 = -w_se * n2 + w_abs * n0 - n2 / tau_21
            dn0_k1 = -dn2_k1

            n0b = n0 + 0.5 * dt * dn0_k1
            n2b = n2 + 0.5 * dt * dn2_k1
            dn2_k2 = -w_se * n2b + w_abs * n0b - n2b / tau_21
            dn0_k2 = -dn2_k2

            n0c = n0 + 0.5 * dt * dn0_k2
            n2c = n2 + 0.5 * dt * dn2_k2
            dn2_k3 = -w_se * n2c + w_abs * n0c - n2c / tau_21
            dn0_k3 = -dn2_k3

            n0d = n0 + dt * dn0_k3
            n2d = n2 + dt * dn2_k3
            dn2_k4 = -w_se * n2d + w_abs * n0d - n2d / tau_21
            dn0_k4 = -dn2_k4

            n0n = n0 + (dt / 6.0) * (dn0_k1 + 2.0 * dn0_k2 + 2.0 * dn0_k3 + dn0_k4)
            n2n = n2 + (dt / 6.0) * (dn2_k1 + 2.0 * dn2_k2 + 2.0 * dn2_k3 + dn2_k4)

            n0n = ti.max(n0n, 0.0)
            n2n = ti.max(n2n, 0.0)
            n2n = ti.min(n2n, 0.99 * n_tot)
            n0n = ti.max(n_tot - n2n, 0.0)

            _cache.n0[iz, it] = n0n / n_tot
            _cache.n2[iz, it] = n2n / n_tot
            _cache.n3[iz, it] = 0.0

    @ti.kernel
    def _kernel_causal_signal_slab(
        iz: int,
        n_tot: float,
        dz: float,
        gamma_s: float,
        a_signal: float,
        tau_21: float,
        sig_threshold: float,
    ):
        """
        Causal signal pass at one z-slab: gain and P_sig from running populations,
        RK4 depletion over dt_arr[it+1] (actual time-grid step, not dt_travel).
        """
        nt = _cache.n0.shape[1]
        n_lam = _cache.sigma_e.shape[0]
        n0_run = _cache.n0[iz, 0]
        n2_run = _cache.n2[iz, 0]

        for it in range(nt):
            _cache.n0[iz, it] = n0_run
            _cache.n2[iz, it] = n2_run
            _cache.n3[iz, it] = 0.0

            n0_d = ti.min(ti.max(n0_run * n_tot, 0.0), n_tot)
            n2_d = ti.min(ti.max(n2_run * n_tot, 0.0), n_tot)

            for ilam in range(n_lam):
                g = gamma_s * (
                    _cache.sigma_e[ilam] * n2_d - _cache.sigma_a[ilam] * n0_d
                )
                _cache.gain[iz, it, ilam] = g
                gdz = ti.min(ti.max(g * dz, -50.0), 50.0)
                p_in = _cache.p_sig[iz, it, ilam]
                _cache.p_sig[iz + 1, it, ilam] = ti.max(p_in * ti.exp(gdz), 0.0)

            w_se = 0.0
            w_abs = 0.0
            sig_w = 0.0
            for ilam in range(n_lam):
                p_avg = 0.5 * (
                    _cache.p_sig[iz, it, ilam] + _cache.p_sig[iz + 1, it, ilam]
                )
                sig_w += p_avg * _cache.dlam[ilam]

            if sig_w >= sig_threshold:
                for ilam in range(n_lam):
                    p_avg = 0.5 * (
                        _cache.p_sig[iz, it, ilam] + _cache.p_sig[iz + 1, it, ilam]
                    )
                    i_s = p_avg / a_signal
                    fac = gamma_s * _cache.dlam[ilam] / _cache.hnu[ilam]
                    w_se += _cache.sigma_e[ilam] * i_s * fac
                    w_abs += _cache.sigma_a[ilam] * i_s * fac

            dt = 0.0
            if it + 1 < nt:
                dt = _cache.dt_arr[it + 1]
            elif nt > 1:
                dt = _cache.dt_arr[nt - 1]

            if dt > 0.0:
                n0 = n0_d
                n2 = n2_d
                dn2_k1 = -w_se * n2 + w_abs * n0 - n2 / tau_21
                dn0_k1 = -dn2_k1

                n0b = n0 + 0.5 * dt * dn0_k1
                n2b = n2 + 0.5 * dt * dn2_k1
                dn2_k2 = -w_se * n2b + w_abs * n0b - n2b / tau_21
                dn0_k2 = -dn2_k2

                n0c = n0 + 0.5 * dt * dn0_k2
                n2c = n2 + 0.5 * dt * dn2_k2
                dn2_k3 = -w_se * n2c + w_abs * n0c - n2c / tau_21
                dn0_k3 = -dn2_k3

                n0d = n0 + dt * dn0_k3
                n2d = n2 + dt * dn2_k3
                dn2_k4 = -w_se * n2d + w_abs * n0d - n2d / tau_21
                dn0_k4 = -dn2_k4

                n0n = n0 + (dt / 6.0) * (dn0_k1 + 2.0 * dn0_k2 + 2.0 * dn0_k3 + dn0_k4)
                n2n = n2 + (dt / 6.0) * (dn2_k1 + 2.0 * dn2_k2 + 2.0 * dn2_k3 + dn2_k4)

                n0n = ti.max(n0n, 0.0)
                n2n = ti.max(n2n, 0.0)
                n2n = ti.min(n2n, 0.99 * n_tot)
                n0n = ti.max(n_tot - n2n, 0.0)

                n0_run = n0n / n_tot
                n2_run = n2n / n_tot

    kernel_pump_advance_z = _kernel_pump_advance_z
    kernel_pump_bwd_advance_z = _kernel_pump_bwd_advance_z
    kernel_qss_march_z = _kernel_qss_march_z
    kernel_compute_gain = _kernel_compute_gain
    kernel_advance_signal_slab = _kernel_advance_signal_slab
    kernel_gain_slab = _kernel_gain_slab
    kernel_spontaneous_source = _kernel_spontaneous_source
    kernel_ase_fwd_slab = _kernel_ase_fwd_slab
    kernel_ase_bwd_slab = _kernel_ase_bwd_slab
    kernel_rk4_populations = _kernel_rk4_populations
    kernel_causal_signal_slab = _kernel_causal_signal_slab
    _kernels_registered = True


def warmup() -> None:
    """Ensure Taichi kernels are registered (JIT compile on first real launch)."""
    _register_kernels()
