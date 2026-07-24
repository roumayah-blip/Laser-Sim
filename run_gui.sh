#!/usr/bin/env bash
# Launch CPA fiber simulator GUI
cd "$(dirname "$0")"
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
exec streamlit run laser_sim/gui/app.py --server.headless true