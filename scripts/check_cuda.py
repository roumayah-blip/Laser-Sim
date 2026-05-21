#!/usr/bin/env python3
"""Verify Taichi can initialize on CUDA (same path as the CPA GPU backend)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from laser_sim.gui.backend_status import probe_compute_backend


def main() -> int:
    status = probe_compute_backend(init_taichi=True)
    print(f"Taichi installed: {status.taichi_installed}")
    print(f"nvidia-smi OK: {status.nvidia_smi_ok}")
    if status.gpu_name:
        print(f"GPU: {status.gpu_name}")
    print(f"Taichi arch: {status.taichi_arch}")
    print(f"CUDA active: {status.cuda_active}")
    print(status.notes)
    if status.cuda_active:
        from laser_sim.physics import taichi_kernels as tk

        tk.warmup()
        print("CUDA kernel warmup: OK")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
