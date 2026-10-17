"""NumCompute: a modular scientific computing toolkit built on plain Python and NumPy.

Submodules are importable as ``numcompute.<module>``. The top-level public API
(names re-exported directly from ``numcompute``) is finalised at integration.
"""

from . import (
    benchmarking,
    io,
    metrics,
    optim,
    pipeline,
    preprocessing,
    rank,
    sort_search,
    stats,
    utils,
)

__version__ = "0.1.0"

__all__ = [
    "benchmarking",
    "io",
    "metrics",
    "optim",
    "pipeline",
    "preprocessing",
    "rank",
    "sort_search",
    "stats",
    "utils",
]
