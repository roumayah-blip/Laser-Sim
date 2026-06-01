import numpy as np

from hybrid_sim.physics.bpm import angular_spectrum_propagator, bpm_step, gaussian_beam_waist


def test_gaussian_waist_propagation():
    nx = ny = 128
    dx = dy = 20e-6
    w0 = 200e-6
    lam = 1030e-9
    z_r = np.pi * w0**2 / lam
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="ij")
    U0 = np.exp(-(xx**2 + yy**2) / w0**2).astype(np.complex128)
    H = angular_spectrum_propagator(nx, ny, dx, dy, z_r, lam, 1.0)
    U1 = bpm_step(U0, H, np.zeros((nx, ny)), np.zeros((nx, ny)))
    w_theory = gaussian_beam_waist(z_r, w0, lam)
    I = np.abs(U1) ** 2
    mx = np.sum(xx * I) / np.sum(I)
    sx2 = np.sum((xx - mx) ** 2 * I) / np.sum(I)
    w_sim = 2 * np.sqrt(sx2)
    assert abs(w_sim - w_theory) / w_theory < 0.15
