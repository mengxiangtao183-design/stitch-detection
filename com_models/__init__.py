"""Comparison segmentation models and their unified C3Net adapters."""

from .registry import build_model, is_comparison_model, list_models

__all__ = ["build_model", "is_comparison_model", "list_models"]

