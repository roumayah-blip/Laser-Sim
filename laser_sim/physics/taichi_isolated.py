"""
Run the Taichi CPA solver in a child process (fresh CUDA context).

Used automatically from Streamlit, where in-process CUDA contexts become invalid
across script reruns. Uses ``subprocess`` (not ``multiprocessing.Process``) because
Streamlit executes user code on a worker thread.
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING

from laser_sim.physics.progress import (
    ProgressCallback,
    emit_progress,
    read_progress_file,
)

if TYPE_CHECKING:
    from laser_sim.physics.fiber_cpa import FiberCPAConfig, FiberCPAResult

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _worker_main(
    cfg_path: str,
    out_path: str,
    err_path: str,
    progress_path: str = "",
) -> None:
    try:
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
        from laser_sim.physics.progress import file_progress_writer
        from laser_sim.physics.taichi_solver import run_fiber_cpa_taichi

        progress_cb = file_progress_writer(progress_path) if progress_path else None
        result = run_fiber_cpa_taichi(cfg, progress_callback=progress_cb)
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
    except Exception:
        with open(err_path, "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())


def _child_env() -> dict[str, str]:
    env = os.environ.copy()
    root = str(_PROJECT_ROOT)
    prev = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = root if not prev else f"{root}{os.pathsep}{prev}"
    return env


def run_fiber_cpa_taichi_isolated(
    cfg: FiberCPAConfig,
    progress_callback: ProgressCallback | None = None,
) -> FiberCPAResult:
    emit_progress(progress_callback, 0.05, "Starting GPU worker process…")

    tmp = tempfile.mkdtemp(prefix="laser_sim_taichi_")
    cfg_path = os.path.join(tmp, "cfg.pkl")
    out_path = os.path.join(tmp, "out.pkl")
    err_path = os.path.join(tmp, "err.txt")
    progress_path = os.path.join(tmp, "progress.json")
    stderr_path = os.path.join(tmp, "stderr.txt")

    try:
        with open(cfg_path, "wb") as f:
            pickle.dump(cfg, f, protocol=pickle.HIGHEST_PROTOCOL)

        cmd = [
            sys.executable,
            "-m",
            "laser_sim.physics.taichi_worker",
            cfg_path,
            out_path,
            err_path,
            progress_path,
        ]
        with open(stderr_path, "wb") as stderr_file:
            proc = subprocess.Popen(
                cmd,
                env=_child_env(),
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
            )

        t_start = time.monotonic()
        while proc.poll() is None:
            elapsed = time.monotonic() - t_start
            snap = read_progress_file(progress_path)
            if snap is not None:
                worker_frac, worker_msg = snap
                # Map worker 0–1 into parent band 8%–93% (leave headroom for pickle load).
                parent_frac = 0.08 + 0.85 * worker_frac
                emit_progress(progress_callback, parent_frac, worker_msg)
            else:
                emit_progress(
                    progress_callback,
                    min(0.08 + 0.02 * (elapsed / 30.0), 0.12),
                    f"GPU worker starting… ({elapsed:.0f} s)",
                )
            time.sleep(0.25)

        exit_code = proc.returncode if proc.returncode is not None else proc.wait()

        if os.path.isfile(err_path) and os.path.getsize(err_path) > 0:
            raise RuntimeError(
                "Taichi GPU worker failed:\n" + Path(err_path).read_text(encoding="utf-8")
            )

        if not os.path.isfile(out_path):
            parts = [f"Taichi GPU worker exited with code {exit_code} and wrote no result."]
            if os.path.isfile(stderr_path) and os.path.getsize(stderr_path) > 0:
                parts.append(
                    "Worker stderr:\n"
                    + Path(stderr_path).read_text(encoding="utf-8", errors="replace")
                )
            if exit_code == -9:
                parts.append(
                    "The process was likely killed by the OS (out of memory). "
                    "Try fewer z steps, a shorter time window, or disable ASE."
                )
            raise RuntimeError("\n".join(parts))

        emit_progress(progress_callback, 0.95, "GPU worker finished")
        with open(out_path, "rb") as f:
            result: FiberCPAResult = pickle.load(f)

        result.notes = (result.notes or "") + " Taichi CUDA via isolated worker (Streamlit)."
        return result
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
