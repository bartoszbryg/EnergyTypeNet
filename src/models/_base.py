"""Shared internal data structures for custom models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Node:
    """Tree node used by custom CART models."""

    feature_index: int | None = None
    threshold: float | None = None
    left: 'Node | None' = None
    right: 'Node | None' = None
    value: float | int | np.ndarray | None = None
    impurity: float = 0.0
    n_samples: int = 0


__all__ = ["Node"]
