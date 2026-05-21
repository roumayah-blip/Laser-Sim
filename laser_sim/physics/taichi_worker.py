"""
CLI entry point for isolated Taichi CPA runs.

Launched as ``python -m laser_sim.physics.taichi_worker <cfg.pkl> <out.pkl> <err.txt> [progress.json]``
so Streamlit (non-main thread) never has to spawn multiprocessing children.
"""

from __future__ import annotations

import sys

from laser_sim.physics.taichi_isolated import _worker_main


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) not in (3, 4):
        print(
            "usage: python -m laser_sim.physics.taichi_worker CFG.pkl OUT.pkl ERR.txt [progress.json]",
            file=sys.stderr,
        )
        return 2
    progress_path = args[3] if len(args) == 4 else ""
    _worker_main(args[0], args[1], args[2], progress_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
