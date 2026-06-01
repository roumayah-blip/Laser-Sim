import numpy as np

from hybrid_sim.physics.kerr import b_integral_2d, check_b_integral


def test_flat_top_b_integral():
    nz, nx, ny = 10, 8, 8
    I0 = 1e12
    intensity = np.full((nz, nx, ny), I0)
    k0 = 2 * np.pi / 1030e-9
    n2 = 6e-20
    dz = 0.001
    B = b_integral_2d(intensity, n2, k0, dz)
    B_expected = k0 * n2 * I0 * dz * nz
    assert np.allclose(B, B_expected, rtol=0.01)
    assert "OK" in check_b_integral(float(B.max()))
