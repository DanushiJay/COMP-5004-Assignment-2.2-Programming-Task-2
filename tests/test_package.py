"""Smoke tests: every module in the required package layout imports cleanly."""

import importlib

import pytest

import numcompute

MODULES = [
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


@pytest.mark.parametrize("name", MODULES)
def test_module_is_importable(name):
    module = importlib.import_module(f"numcompute.{name}")
    assert getattr(numcompute, name) is module
