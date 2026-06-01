"""Split-step angular-spectrum beam propagation."""

from __future__ import annotations

import numpy as np


def angular_spectrum_propagator(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    dz: float,
    wavelength_m: float,
    n_background: float,
) -> np.ndarray:
    """Precompute H(kx, ky) = exp(i kz dz), shape (nx, ny)."""
    k0 = 2.0 * np.pi * n_background / wavelength_m
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    kz2 = k0**2 - KX**2 - KY**2
    kz = np.sqrt(np.maximum(kz2, 0.0))
    return np.exp(1j * kz * dz)


def bpm_step(
    U_in: np.ndarray,
    H: np.ndarray,
    gain_xy: np.ndarray,
    phase_xy: np.ndarray,
) -> np.ndarray:
    """Split-step: FFT propagate, then gain + phase."""
    U_k = np.fft.fft2(U_in)
    U_prop = np.fft.ifft2(U_k * H)
    return U_prop * np.exp(gain_xy) * np.exp(1j * phase_xy)


def beam_quality_m2(
    U: np.ndarray,
    dx: float,
    dy: float,
    wavelength_m: float,
) -> tuple[float, float]:
    """M² from second-moment widths (x and y)."""
    I = np.abs(U) ** 2
    s = float(np.sum(I))
    if s <= 0:
        return 1.0, 1.0
    nx, ny = U.shape
    x = (np.arange(nx) - (nx - 1) / 2.0) * dx
    y = (np.arange(ny) - (ny - 1) / 2.0) * dy
    xx, yy = np.meshgrid(x, y, indexing="ij")
    mx = float(np.sum(xx * I) / s)
    my = float(np.sum(yy * I) / s)
    sx2 = float(np.sum((xx - mx) ** 2 * I) / s)
    sy2 = float(np.sum((yy - my) ** 2 * I) / s)
    w_x = 2.0 * np.sqrt(max(sx2, 0.0))
    w_y = 2.0 * np.sqrt(max(sy2, 0.0))
    # Far-field divergence estimate via FFT width
    U_k = np.fft.fftshift(np.fft.fft2(U))
    Ik = np.abs(U_k) ** 2
    sk = float(np.sum(Ik))
    if sk <= 0:
        return 1.0, 1.0
    kx = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(nx, d=dx))
    ky = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(ny, d=dy))
    kxx, kyy = np.meshgrid(kx, ky, indexing="ij")
    mkx = float(np.sum(kxx * Ik) / sk)
    mky = float(np.sum(kyy * Ik) / sk)
    skx2 = float(np.sum((kxx - mkx) ** 2 * Ik) / sk)
    sky2 = float(np.sum((kyy - mky) ** 2 * Ik) / sk)
    dkx = float(np.max(kxx) - np.min(kxx))
    dky = float(np.max(kyy) - np.min(kyy))
    theta_x = 2.0 * np.sqrt(max(skx2, 0.0)) / max(dkx, 1e-30) * nx
    theta_y = 2.0 * np.sqrt(max(sky2, 0.0)) / max(dky, 1e-30) * ny
    m2_x = np.pi * w_x * theta_x / wavelength_m
    m2_y = np.pi * w_y * theta_y / wavelength_m
    return float(np.clip(m2_x, 0.5, 50.0)), float(np.clip(m2_y, 0.5, 50.0))


def gaussian_beam_waist(z: float, w0: float, wavelength_m: float, n: float = 1.0) -> float:
    z_r = np.pi * w0**2 * n / wavelength_m
    return w0 * np.sqrt(1.0 + (z / z_r) ** 2)
