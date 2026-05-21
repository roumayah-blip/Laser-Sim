"""Probe Taichi / NVIDIA availability for the GUI."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class ComputeBackendStatus:
    taichi_installed: bool
    taichi_arch: str | None
    cuda_requested: bool
    cuda_active: bool
    gpu_name: str | None
    nvidia_smi_ok: bool
    notes: str


def _nvidia_gpu_name() -> str | None:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    line = proc.stdout.strip().splitlines()
    return line[0].strip() if line else None


def probe_compute_backend(*, init_taichi: bool = False) -> ComputeBackendStatus:
    """
    Report whether Taichi can run on CUDA.

    When ``init_taichi`` is True, attempts ``ti.init(arch=cuda)`` (use from
    ``scripts/check_cuda.py`` only). The Streamlit GUI leaves this False so CUDA
    is initialized in the same execution context as the simulation run.
    """
    gpu = _nvidia_gpu_name()
    nvidia_ok = gpu is not None

    try:
        import taichi as ti
    except ImportError:
        return ComputeBackendStatus(
            taichi_installed=False,
            taichi_arch=None,
            cuda_requested=False,
            cuda_active=False,
            gpu_name=gpu,
            nvidia_smi_ok=nvidia_ok,
            notes="Taichi not installed — simulations use CPU only.",
        )

    arch_name: str | None = None
    cuda_active = False
    notes_parts: list[str] = []

    from laser_sim.physics import taichi_kernels as tk

    arch_name = tk.active_arch_name()
    if arch_name:
        cuda_active = "cuda" in arch_name.lower()
    elif init_taichi:
        try:
            tk.init_taichi(arch="cuda", fp="f32", force_reinit=False)
            arch_name = str(ti.cfg.arch)
            cuda_active = "cuda" in arch_name.lower()
        except Exception as exc:
            try:
                tk.abandon_taichi_runtime()
                tk.init_taichi(arch="cpu", fp="f32", force_reinit=False)
                arch_name = str(ti.cfg.arch)
            except Exception:
                arch_name = None
            notes_parts.append(f"CUDA init failed ({exc}); Taichi fell back to CPU.")
    else:
        notes_parts.append(
            "CUDA initializes when you run a simulation with backend “cuda”."
        )

    if cuda_active:
        notes_parts.append(
            "Taichi CUDA is active. Pump QSS still runs on CPU; signal/ASE slabs use the GPU."
        )
    elif nvidia_ok and not cuda_active:
        notes_parts.append(
            "NVIDIA GPU detected but Taichi is not on CUDA — pick backend “cuda” and check drivers."
        )
    elif not nvidia_ok:
        notes_parts.append("nvidia-smi unavailable — GPU backend will fall back to CPU.")

    return ComputeBackendStatus(
        taichi_installed=True,
        taichi_arch=arch_name,
        cuda_requested=init_taichi,
        cuda_active=cuda_active,
        gpu_name=gpu,
        nvidia_smi_ok=nvidia_ok,
        notes=" ".join(notes_parts) if notes_parts else "Ready.",
    )


def normalize_sim_backend(name: str) -> str:
    """Map GUI labels to ``run_fiber_cpa`` backend argument."""
    key = name.lower().replace("-", "_")
    if key in ("cuda", "taichi", "taichi_cuda", "gpu"):
        return "taichi"
    return "cpu"
