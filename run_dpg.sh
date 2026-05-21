#!/usr/bin/env bash
cd "$(dirname "$0")"
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
exec python -m laser_sim.gui.dpg_app
