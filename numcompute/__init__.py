"""
NumCompute: a modular scientific computing toolkit built on plain Python and NumPy.

The public API is organised by submodule: use ``numcompute.<module>``, e.g.
``from numcompute import stats`` then ``stats.mean(X)``. 
Names are not re-exported at the top level, which avoids clashes such as 
the ``rank`` module and the ``rank`` function.

``benchmarking`` is a development tool and is not imported here.
"""

from . import (
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
