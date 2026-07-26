"""Ricci curvature features for cell fate trajectories."""

import os
from importlib.metadata import PackageNotFoundError, version

# Scanpy/CellRank import Numba-decorated functions at module import time. On some
# shared filesystems Numba cannot locate a writable package-adjacent cache, so set
# a stable writable cache before any optional official loader is imported.
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba-cache-ricci-cell-fate")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-ricci-cell-fate")

try:
    __version__ = version("ricci-cell-fate")
except PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"
