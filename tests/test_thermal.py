import numpy as np

from hybrid_sim.physics.thermal import solve_heat_2d, thermal_lens_focal_length


def test_parabolic_thermal_lens():
    nx, ny = 64, 64
    dx = dy = 50e-6
    T0 = 5.0
    x = (np.arange(nx) - (nx - 1) / 2) * dx
    xx, yy = np.meshgrid(x, x, indexing="ij")
    w_p = 0.5e-3
    T_profile = T0 * (1.0 - 2.0 * (xx**2 + yy**2) / w_p**2)
    f = thermal_lens_focal_length(
        7.3e-6, T_profile, dx, dy, 1030e-9, crystal_length_m=0.01
    )
    assert f > 0.01, "thermal lens should be finite and paraxial-valid"
