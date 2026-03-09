
"""Data loading and schema normalization utilities for churn prediction."""

from .dataset import REQUIRED_TARGET_COLUMN, load_dataset

__all__ = ["REQUIRED_TARGET_COLUMN", "load_dataset"]
