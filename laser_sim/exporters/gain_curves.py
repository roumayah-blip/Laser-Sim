"""
gain_curves.py -- extract temporally & spectrally resolved gain curves from a
FiberCPAResult, for feeding rigorous gain narrowing into the external
"Temporally and Spectrally Resolved SPM" NLSE tool.

Physics
-------
The signal field is stored as P_s(z, t, lambda) in W/nm (`signal_fwd_w_nm`,
shape (nz, nt, nlam)).  The net gain a pulse experiences is the ratio of the
z = L output to the z = 0 input.  Because the signal is a *chirped burst*, each
wavelength recurs once per pulse, so wavelength<->time are "married" only WITHIN
a single pulse.  We therefore extract per-pulse:

  * spectral gain   G_lambda(lambda) = integral_t Pout / integral_t Pin   (over one pulse)
  * temporal gain   G_time(t)        = integral_lambda Pout / integral_lambda Pin (over one pulse)
  * chirp map       lambda_c(t)      = intensity-weighted mean wavelength vs t

For a separable, non-double-counting gain model we also derive:

  * S(lambda) = spectral-narrowing shape (from G_lambda, mean removed)
  * T(t)      = pure temporal depletion = G_time(t) / S(lambda_c(t)),
                i.e. the temporal gain with the spectral shape (mapped through
                the chirp) divided out.  With no saturation T is flat; with
                front-to-back inversion depletion T droops across the pulse.
  * G0        = pulse energy gain (Eout/Ein for the selected pulse)

so that the applied gain factorizes as  G0 * S(lambda) * T(t)  with S, T carrying
only shape (the SPM tool renormalizes energy to G0 exactly).
"""
from __future__ import annotations

import os
import numpy as np

# numpy >= 2.0 renamed trapz -> trapezoid; support both
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))


# ----------------------------------------------------------------------
# pulse segmentation of the burst
# ----------------------------------------------------------------------
def _pulse_windows(t_s, p_in_time, n_expected, frac=0.02):
    """Segment the burst into per-pulse [i_lo, i_hi] index windows.

    Regions where the input temporal power exceeds `frac` of its peak are taken
    as pulses; returns up to the `n_expected` most energetic, time-ordered."""
    p = np.asarray(p_in_time, float)
    if p.max() <= 0:
        return []
    above = p >= frac * p.max()
    # contiguous True runs
    edges = np.diff(above.astype(int))
    starts = list(np.where(edges == 1)[0] + 1)
    stops = list(np.where(edges == -1)[0] + 1)
    if above[0]:
        starts = [0] + starts
    if above[-1]:
        stops = stops + [len(p)]
    runs = list(zip(starts, stops))
    if not runs:
        return []
    # rank by integrated energy, keep the n_expected strongest, then re-sort by time
    runs.sort(key=lambda r: -_trapz(p[r[0]:r[1]], t_s[r[0]:r[1]]))
    runs = runs[:max(1, n_expected)]
    runs.sort(key=lambda r: r[0])
    return runs


# ----------------------------------------------------------------------
# main extraction
# ----------------------------------------------------------------------
def extract_gain_curves(result, pulse_index=0, meta=None, floor_frac=1e-4):
    """Extract per-pulse gain curves from a FiberCPAResult.

    pulse_index : which pulse in the burst to characterize (0 = first, least
                  depleted; -1 = last, worst-case depletion).
    Returns a dict of arrays + scalars + metadata (see module docstring)."""
    t = np.asarray(result.t_s, float)
    lam = np.asarray(result.wavelength_nm, float)
    Psz = np.asarray(result.signal_fwd_w_nm, float)      # (nz, nt, nlam)
    if Psz.ndim != 3:
        raise ValueError("signal_fwd_w_nm must be (nz, nt, nlam)")
    P_in = Psz[0]                                        # (nt, nlam) at z=0
    P_out = Psz[-1]                                      # (nt, nlam) at z=L
    if not (np.isfinite(P_in).all() and np.isfinite(P_out).all()):
        raise ValueError("signal field contains non-finite values (unstable run)")

    # temporal power (integrate over lambda) for pulse segmentation
    p_in_t = _trapz(P_in, lam, axis=1)
    n_pulses = int(getattr(meta, "burst_count", None) or (meta or {}).get("burst_count", 1) or 1) \
        if not isinstance(meta, dict) else int(meta.get("burst_count", 1) or 1)
    windows = _pulse_windows(t, p_in_t, n_pulses)
    if not windows:
        raise ValueError("no signal pulses found in the input field")
    idx = pulse_index if pulse_index >= 0 else len(windows) + pulse_index
    idx = int(np.clip(idx, 0, len(windows) - 1))
    i_lo, i_hi = windows[idx]
    sl = slice(i_lo, i_hi)
    tw = t[sl]

    Pin_w = P_in[sl]        # (ntw, nlam)
    Pout_w = P_out[sl]

    # --- spectral gain over the pulse:  integral_t Pout / integral_t Pin ---
    ein_lam = _trapz(Pin_w, tw, axis=0)               # J/nm-ish (per lambda)
    eout_lam = _trapz(Pout_w, tw, axis=0)
    flr_l = floor_frac * ein_lam.max()
    G_lambda = np.where(ein_lam > flr_l, eout_lam / np.maximum(ein_lam, flr_l), 1.0)

    # --- temporal gain over the pulse:  integral_lambda Pout / integral_lambda Pin ---
    p_in_tw = _trapz(Pin_w, lam, axis=1)              # W vs t
    p_out_tw = _trapz(Pout_w, lam, axis=1)
    flr_t = floor_frac * p_in_tw.max()
    G_time = np.where(p_in_tw > flr_t, p_out_tw / np.maximum(p_in_tw, flr_t), 1.0)

    # --- chirp map lambda_c(t): input-intensity-weighted mean wavelength ---
    denom = np.maximum(p_in_tw, flr_t)
    lam_c = np.where(p_in_tw > flr_t,
                     _trapz(Pin_w * lam[None, :], lam, axis=1) / denom,
                     np.nan)

    # --- separable, non-double-counting decomposition ---
    # spectral-narrowing shape S(lambda): input-energy-weighted mean removed
    w_lam = ein_lam / max(ein_lam.sum(), 1e-300)
    Sbar = float(np.sum(w_lam * G_lambda)) or 1.0
    S_lambda = G_lambda / Sbar
    # map S onto time via the chirp, divide out of temporal gain -> pure depletion.
    # only trust the region where the pulse actually has power; elsewhere T=1
    # (no information), and clamp S away from zero to avoid edge blow-up.
    core = p_in_tw > 0.05 * p_in_tw.max()
    S_of_t = np.interp(lam_c, lam, S_lambda, left=1.0, right=1.0)
    S_of_t = np.clip(np.where(np.isfinite(S_of_t), S_of_t, 1.0), 0.1, 10.0)
    w_t = np.where(core, p_in_tw, 0.0)
    w_t = w_t / max(w_t.sum(), 1e-300)
    g_t = G_time / (np.sum(w_t * G_time) or 1.0)
    T_time_raw = np.where(core, g_t / S_of_t, 1.0)
    Tbar = float(np.sum(w_t * T_time_raw)) or 1.0
    T_time = np.where(core, T_time_raw / Tbar, 1.0)

    # --- pulse energy gain G0 ---
    ein = float(_trapz(p_in_tw, tw))
    eout = float(_trapz(p_out_tw, tw))
    G0 = eout / ein if ein > 0 else float("nan")

    md = dict(meta) if isinstance(meta, dict) else {}
    md.setdefault("pulse_index", idx)
    md.setdefault("n_pulses_found", len(windows))
    md["pulse_energy_gain_G0"] = G0
    md["pulse_energy_gain_dB"] = 10 * np.log10(G0) if G0 and G0 > 0 else float("nan")

    return dict(
        wavelength_nm=lam,
        G_lambda=G_lambda,
        S_lambda=S_lambda,
        t_rel_s=tw - float(tw[np.argmax(p_in_tw)]),      # relative to pulse peak
        G_time=G_time,
        T_time=T_time,
        lambda_of_t_nm=lam_c,
        input_spectral_wnm=ein_lam,
        output_spectral_wnm=eout_lam,
        input_power_w=p_in_tw,
        output_power_w=p_out_tw,
        G0=G0,
        meta=md,
    )


# ----------------------------------------------------------------------
# labeling & saving
# ----------------------------------------------------------------------
def _sanitize(s):
    return "".join(c if (c.isalnum() or c in ".-") else "" for c in str(s))


def make_label(meta):
    """Intuitive auto-label from fiber type, MFD, pump power, direction, length."""
    m = meta or {}
    fiber = _sanitize(m.get("material", m.get("fiber_type", "fiber")))
    mfd = m.get("mfd_um")
    pw = m.get("pump_power_w")
    direction = m.get("pump_direction", "fwd")
    L = m.get("fiber_length_m")
    parts = [f"gain_{fiber}"]
    if mfd is not None:
        parts.append(f"MFD{float(mfd):.0f}um")
    if pw is not None:
        cw = "CW" if m.get("pump_cw") else ""
        parts.append(f"Pp{float(pw):.0f}W{cw}")
    parts.append(_sanitize(direction))
    if L is not None:
        parts.append(f"L{float(L):.2f}m")
    if m.get("n_pulses_found", 1) > 1:
        parts.append(f"p{m.get('pulse_index', 0)}")
    return "_".join(parts)


def save_gain_curves(curves, out_dir, label=None):
    """Write curves to <label>.npz (full) and <label>.csv (portable). Returns paths."""
    os.makedirs(out_dir, exist_ok=True)
    label = label or make_label(curves.get("meta"))
    npz_path = os.path.join(out_dir, label + ".npz")
    csv_path = os.path.join(out_dir, label + ".csv")

    # npz: arrays + flattened scalar metadata
    arrays = {k: v for k, v in curves.items()
              if isinstance(v, np.ndarray)}
    meta = curves.get("meta", {})
    np.savez(npz_path, label=label,
             meta_keys=np.array(list(meta.keys())),
             meta_vals=np.array([str(v) for v in meta.values()]),
             G0=float(curves.get("G0", np.nan)),
             **arrays)

    # csv: the two curves the SPM tool ingests, on their native axes, with a
    # metadata header.  Spectral block then temporal block.
    lam = curves["wavelength_nm"]; Gl = curves["G_lambda"]; Sl = curves["S_lambda"]
    tr = curves["t_rel_s"]; Gt = curves["G_time"]; Tt = curves["T_time"]
    lam_t = curves["lambda_of_t_nm"]
    with open(csv_path, "w") as f:
        f.write(f"# gain curves: {label}\n")
        for k, v in meta.items():
            f.write(f"# {k}: {v}\n")
        f.write(f"# G0_energy_gain: {curves.get('G0')}\n")
        f.write("# --- SPECTRAL: wavelength_nm, G_lambda, S_lambda(shape) ---\n")
        f.write("section,x,y,shape\n")
        for a, b, c in zip(lam, Gl, Sl):
            f.write(f"spectral,{a:.6g},{b:.6g},{c:.6g}\n")
        f.write("# --- TEMPORAL: t_rel_s, G_time, T_time(depletion), lambda_of_t_nm ---\n")
        for a, b, c, d in zip(tr, Gt, Tt, lam_t):
            f.write(f"temporal,{a:.6g},{b:.6g},{c:.6g},{d:.6g}\n")
    return npz_path, csv_path


def export_all_pulses(result, out_dir, sim_inputs=None, meta=None, want_csv=True):
    """Extract & save gain curves for EVERY pulse in the burst. Returns a list
    of (pulse_index, npz_path, csv_path_or_None, G0_dB)."""
    base_meta = meta if meta is not None else meta_from_result(result, sim_inputs)
    # discover how many pulses exist by extracting pulse 0 (fills n_pulses_found)
    probe = extract_gain_curves(result, pulse_index=0, meta=dict(base_meta))
    n = int(probe["meta"].get("n_pulses_found", 1))
    out = []
    for p in range(n):
        curves = extract_gain_curves(result, pulse_index=p, meta=dict(base_meta))
        label = make_label(curves["meta"])
        npz, csv = save_gain_curves(curves, out_dir, label=label)
        if not want_csv:
            try:
                os.remove(csv)
            except OSError:
                pass
            csv = None
        out.append((p, npz, csv, curves["meta"].get("pulse_energy_gain_dB")))
    return out


def meta_from_result(result, sim_inputs=None):
    """Build a labeling metadata dict from a result (+ optional SimInputs)."""
    m = {}
    si = sim_inputs
    if si is not None:
        m["material"] = getattr(si, "material_key", "fiber")
        m["pump_power_w"] = getattr(si, "pump_peak_power_w", None)
        m["pump_cw"] = bool(getattr(si, "pump_cw", False))
        m["fiber_length_m"] = getattr(si, "fiber_length_m", None)
        m["burst_count"] = getattr(si, "burst_count", 1)
        m["signal_center_nm"] = getattr(si, "signal_center_nm", None)
        # pump direction: single-mode is forward; multichannel may set a
        # backward fraction per pump channel
        direction = "fwd"
        chans = getattr(si, "pump_channels_inputs", None)
        if chans:
            fracs = [float(c.get("backward_fraction", 0.0) or 0.0)
                     for c in chans if isinstance(c, dict)]
            fmax = max(fracs) if fracs else 0.0
            direction = "bwd" if fmax >= 0.5 else ("bidir" if fmax > 0.0 else "fwd")
        m["pump_direction"] = direction
    # MFD from the simulated signal mode area
    area = getattr(result, "signal_mode_area_m2", 0.0) or getattr(result, "a_signal_m2", 0.0)
    if area and area > 0:
        m["mfd_um"] = 2.0 * np.sqrt(area / np.pi) * 1e6
    return m
