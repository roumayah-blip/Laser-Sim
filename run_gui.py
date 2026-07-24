#!/usr/bin/env python3
"""Launch the CPA fiber simulator GUI. Click Run/Play on this file instead of using a terminal."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

venv_python = ROOT / ".venv" / "bin" / "python"
python = str(venv_python) if venv_python.exists() else sys.executable

subprocess.run(
    [python, "-m", "streamlit", "run", "laser_sim/gui/app.py", "--server.headless", "true"],
    cwd=ROOT,
)
