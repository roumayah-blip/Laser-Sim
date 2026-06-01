"""Packet equalization: compute input weights to flatten amplifier output."""

from __future__ import annotations

import numpy as np


def estimate_flat_packet_weights(
    pulse_energies_in_j: np.ndarray,
    pulse_energies_out_j: np.ndarray,
    clip_ratio: float = 10.0,
) -> np.ndarray:
    """
    Return per-pulse relative power weights that equalise output energy.

    Given input and output energies for each pulse in the last run:
        G_b = E_out[b] / E_in[b]          (measured gain per pulse)
        w_raw[b] = 1 / G_b                 (inverse gain weighting)
        w[b] = w_raw[b] / mean(w_raw)      (normalised, mean=1)

    Parameters
    ----------
    pulse_energies_in_j : array of per-pulse input energies
    pulse_energies_out_j : array of per-pulse output energies
    clip_ratio : max allowed weight ratio w_max/w_min (prevents extreme distortion)

    Returns
    -------
    weights : shape (N,), mean=1, ready for ChirpedBurstSpec.pulse_relative_powers
    """
    e_in = np.asarray(pulse_energies_in_j, dtype=np.float64)
    e_out = np.asarray(pulse_energies_out_j, dtype=np.float64)
    if e_in.shape != e_out.shape or e_in.size == 0:
        raise ValueError("pulse_energies_in_j and _out_j must be same non-empty shape")
    gains = e_out / np.maximum(e_in, 1e-30)
    gains = np.maximum(gains, 1e-10)
    w_raw = 1.0 / gains
    w_min = np.max(w_raw) / clip_ratio
    w_raw = np.maximum(w_raw, w_min)
    mean = float(np.mean(w_raw))
    return w_raw / max(mean, 1e-30)
