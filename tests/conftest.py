"""Shared pytest fixtures and markers for DecodeBench."""

import pytest


def _has_cuda() -> bool:
    """Check for CUDA availability without importing torch eagerly."""
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def pytest_collection_modifyitems(config, items):
    """Auto-skip gpu tests when CUDA is unavailable."""
    if _has_cuda():
        return
    skip_gpu = pytest.mark.skip(reason="CUDA not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
