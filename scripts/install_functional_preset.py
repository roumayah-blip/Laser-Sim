#!/usr/bin/env python3
"""Install repo bundled preset 'functional_5_20_2026' into ~/.laser_sim/fiber_presets.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from laser_sim.gui.fiber_presets import PRESETS_PATH, load_presets

NAME = "functional 5_20_2026"
BUNDLED = ROOT / "laser_sim" / "gui" / "presets" / "functional_5_20_2026.json"


def main() -> int:
    if not BUNDLED.is_file():
        print(f"Missing {BUNDLED}", file=sys.stderr)
        return 1
    payload = json.loads(BUNDLED.read_text(encoding="utf-8"))
    presets = load_presets()
    presets[NAME] = payload
    PRESETS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PRESETS_PATH.write_text(json.dumps(presets, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Installed preset '{NAME}' -> {PRESETS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
