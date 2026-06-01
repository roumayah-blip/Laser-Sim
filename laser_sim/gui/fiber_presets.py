"""Save and load named fiber geometry / dopant presets for the GUI."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

PRESETS_PATH = Path.home() / ".laser_sim" / "fiber_presets.json"
BUNDLED_PRESETS_DIR = Path(__file__).resolve().parent / "presets"

FIBER_PRESET_FIELDS = (
    "material_key",
    "core_diameter_um",
    "core_na",
    "cladding_diameter_um",
    "fiber_length_m",
    "cladding_pumped",
    "ignore_gamma_for_n",
    "abs_mode_db_per_m",
    "pump_absorption_db_per_m",
    "total_absorption_db",
    "pump_wavelength_nm",
    "simulation_pump_wavelength_nm",
)


def _sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\s\-]+", "", name.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)
    if not cleaned:
        raise ValueError("Preset name cannot be empty.")
    return cleaned


def _load_bundled_presets() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if not BUNDLED_PRESETS_DIR.is_dir():
        return out
    for path in sorted(BUNDLED_PRESETS_DIR.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict):
            out[path.stem] = payload
    return out


def load_presets() -> dict[str, dict[str, Any]]:
    out = _load_bundled_presets()
    if not PRESETS_PATH.is_file():
        return out
    try:
        data = json.loads(PRESETS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return out
    if not isinstance(data, dict):
        return out
    for name, payload in data.items():
        if isinstance(name, str) and isinstance(payload, dict):
            out[name] = payload
    return out


def save_preset(name: str, fields: dict[str, Any]) -> str:
    key = _sanitize_name(name)
    presets = load_presets()
    payload = {k: fields[k] for k in FIBER_PRESET_FIELDS if k in fields}
    presets[key] = payload
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2, sort_keys=True), encoding="utf-8")
    return key


def delete_preset(name: str) -> bool:
    key = _sanitize_name(name)
    presets = load_presets()
    if key not in presets:
        return False
    del presets[key]
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2, sort_keys=True), encoding="utf-8")
    return True


def preset_to_session_updates(payload: dict[str, Any]) -> dict[str, Any]:
    """Map stored preset keys to Streamlit session_state widget keys."""
    updates: dict[str, Any] = {}
    mapping = {
        "material_key": "material_key",
        "core_diameter_um": "core_um",
        "core_na": "core_na",
        "cladding_diameter_um": "clad_um",
        "fiber_length_m": "fiber_length_m",
        "cladding_pumped": "cladding_pumped",
        "ignore_gamma_for_n": "ignore_gamma_for_n",
        "pump_absorption_db_per_m": "pump_abs_db_per_m",
        "total_absorption_db": "total_abs_db",
        "pump_wavelength_nm": "pump_wl_nm",
        "simulation_pump_wavelength_nm": "sim_pump_wl_nm",
    }
    for src, dst in mapping.items():
        if src in payload:
            updates[dst] = payload[src]
    if "abs_mode_db_per_m" in payload:
        updates["abs_mode"] = "dB/m" if payload["abs_mode_db_per_m"] else "Total dB over length"
    return updates
