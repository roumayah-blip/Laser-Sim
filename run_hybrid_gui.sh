#!/usr/bin/env bash
# Launch hybrid fiber + solid-state CPA GUI
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi
exec streamlit run hybrid_sim/gui/app.py "$@"
