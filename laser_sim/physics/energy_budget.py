"""Energy accounting for CPA fiber amplifier runs (pump vs radiation)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from laser_sim.physics.four_level import FourLevelPopulations, guided_fraction_into_mode


@dataclass(frozen=True)
class AmplifierEnergyBudget:
    """Pump power vs radiation leaving the fiber (signal + ASE + unguided fluorescence)."""

    pump_absorbed_j: float
    signal_net_j: float
    ase_out_j: float
    spontaneous_total_j: float
    spontaneous_guided_j: float
    spontaneous_unguided_j: float
    total_emission_j: float
    balance_residual_j: float
    balance_ok: bool
    ase_fraction_of_emission: float


def compute_amplifier_energy_budget(
    *,
    t_s: np.ndarray,
    z_m: np.ndarray,
    populations: FourLevelPopulations,
    wavelength_nm: np.ndarray,
    n_tot: float,
    tau_21_s: float,
    a_signal_m2: float,
    eta_guided: float,
    gamma_s: float,
    energy_pump_in_j: float,
    energy_pump_out_j: float,
    energy_packet_in_j: float,
    energy_packet_out_j: float,
    energy_ase_out_j: float,
    tol_fraction: float = 0.05,
) -> AmplifierEnergyBudget:
    """
    Compare pump energy absorbed to radiation extracted via populations.

    Spontaneous emission removes N₂ at rate n₂/τ₂₁; fraction f_g = η·Γ_s enters the
    guided ASE channel; (1−f_g) is unguided fluorescence loss.
    """
    t = np.asarray(t_s, dtype=np.float64)
    z = np.asarray(z_m, dtype=np.float64)
    n2 = populations.n2 * n_tot
    tau = float(tau_21_s)
    e_pump_abs = max(energy_pump_in_j - energy_pump_out_j, 0.0)
    e_sig = energy_packet_out_j - energy_packet_in_j
    e_ase = energy_ase_out_j

    if tau <= 0.0 or t.size < 2 or z.size < 2:
        e_tot = e_sig + e_ase
        return AmplifierEnergyBudget(
            pump_absorbed_j=e_pump_abs,
            signal_net_j=e_sig,
            ase_out_j=e_ase,
            spontaneous_total_j=0.0,
            spontaneous_guided_j=0.0,
            spontaneous_unguided_j=0.0,
            total_emission_j=e_tot,
            balance_residual_j=e_tot - e_pump_abs,
            balance_ok=e_tot <= e_pump_abs * (1.0 + tol_fraction),
            ase_fraction_of_emission=e_ase / max(e_tot, 1e-30),
        )

    dt = np.zeros(t.size, dtype=np.float64)
    dt[1:] = np.diff(t)
    dt[0] = dt[1]
    dz = float(z[1] - z[0])

    wl = np.asarray(wavelength_nm, dtype=np.float64)
    hnu = 6.62607015e-34 * 2.99792458e8 / (wl * 1e-9)
    hnu_mean = float(np.mean(hnu))

    e_spont = 0.0
    nz, nt = n2.shape
    for iz in range(nz):
        for it in range(nt):
            if dt[it] <= 0.0:
                continue
            e_spont += float(n2[iz, it]) / tau * hnu_mean * a_signal_m2 * dz * dt[it]

    f_g = guided_fraction_into_mode(eta_guided, gamma_s)
    e_guided = f_g * e_spont
    e_unguided = e_spont - e_guided

    e_emit = e_sig + e_ase + e_unguided
    residual = e_emit - e_pump_abs
    balance_ok = e_emit <= e_pump_abs * (1.0 + tol_fraction)

    return AmplifierEnergyBudget(
        pump_absorbed_j=e_pump_abs,
        signal_net_j=e_sig,
        ase_out_j=e_ase,
        spontaneous_total_j=e_spont,
        spontaneous_guided_j=e_guided,
        spontaneous_unguided_j=e_unguided,
        total_emission_j=e_emit,
        balance_residual_j=residual,
        balance_ok=balance_ok,
        ase_fraction_of_emission=e_ase / max(e_emit, 1e-30),
    )
