"""Experiment management package for Med-DeepSeek.

This package provides utilities to run, track, and compare epidemiological
forecasting experiments. It is intentionally framework-agnostic so that
statistical, machine learning, deep learning, and agent-based models can share
one consistent interface.
"""

from __future__ import annotations

import os

# matplotlib 백엔드를 가장 먼저 설정
os.environ.setdefault("MPLBACKEND", "Agg")

# matplotlib을 먼저 설정
try:
    import matplotlib

    matplotlib.use("Agg", force=True)
except (ImportError, Exception):
    pass

__all__ = ["ExperimentRunner", "BatchRunner"]


def __getattr__(name: str):  # pragma: no cover - simple lazy loader
    if name == "ExperimentRunner":
        from .run_experiment import ExperimentRunner as _ExperimentRunner

        return _ExperimentRunner
    if name == "BatchRunner":
        from .batch_runner import BatchRunner as _BatchRunner

        return _BatchRunner
    raise AttributeError(f"module 'experiments' has no attribute {name!r}")


def __dir__():  # pragma: no cover - cosmetic helper
    return sorted(list(globals().keys()) + __all__)
