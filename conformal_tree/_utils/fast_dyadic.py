"""Corrected, cached implementation of the robust multidimensional dyadic tree.

The implementation follows the efficient heap-based package algorithm but
makes the choices needed by the revised theory explicit:

* dyadic children form a genuine partition (left <= midpoint, right >);
* the fitting score of a designated test point can be hidden while its X value
  remains in all routing and count calculations;
* no candidates are evaluated after the leaf cap has been reached;
* ties are resolved by a fixed lexicographic order;
* the package's efficient-tree criterion is retained exactly: improvement is
  parent range minus the mean of the two child ranges, and the improvement
  multiplier is parent range divided by mean child range, minus one.

For repeated score-hidden fits with the same calibration sample, all
calibration counts and ranges are cached by dyadic region. A fit for a new
test covariate then requires only tree/heap operations and cached lookups.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


Path = tuple[tuple[int, int], ...]


def _child_path(path: Path, direction: int, side: int) -> Path:
    return path + ((int(direction), int(side)),)


def conformal_threshold(scores: np.ndarray, alpha: float) -> float:
    """Exact split-conformal order statistic, including the +infinity case."""
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    order = int(math.ceil((len(scores) + 1) * (1.0 - alpha)))
    if order > len(scores):
        return math.inf
    return float(np.partition(scores, order - 1)[order - 1])


@dataclass(frozen=True)
class SplitSummary:
    left: np.ndarray
    right: np.ndarray
    value: float
    rate: float


class DyadicSummaryCache:
    """Calibration-sample summaries shared by many hidden-score tree fits."""

    def __init__(
        self,
        X: np.ndarray,
        tree_scores: np.ndarray,
        domain: np.ndarray,
        min_samples_leaf: int,
        max_leaves: int,
        max_depth: int = 8,
        threshold: float = 0.05,
        max_directions: Optional[int] = None,
        direction_seed: int = 0,
    ):
        X = np.asarray(X, dtype=float)
        tree_scores = np.asarray(tree_scores, dtype=float)
        domain = np.asarray(domain, dtype=float)
        if X.ndim != 2 or tree_scores.shape != (len(X),):
            raise ValueError("X and tree_scores have incompatible shapes")
        if domain.shape != (X.shape[1], 2):
            raise ValueError("domain must have shape (d, 2)")
        if min_samples_leaf < 2:
            raise ValueError("the revised theory requires min_samples_leaf >= 2")
        if max_leaves < 1 or max_depth < 1:
            raise ValueError("max_leaves and max_depth must be positive")
        self.X = X
        self.tree_scores = tree_scores
        self.domain = domain
        self.m = int(min_samples_leaf)
        self.max_leaves = int(max_leaves)
        self.max_depth = int(max_depth)
        self.threshold = float(threshold)
        self.d = X.shape[1]
        self.max_directions = (
            self.d if max_directions is None else min(int(max_directions), self.d)
        )
        if self.max_directions < 1:
            raise ValueError("max_directions must be positive")
        self.direction_seed = int(direction_seed)
        self._indices: dict[Path, np.ndarray] = {(): np.arange(len(X), dtype=int)}
        self._bounds: dict[Path, np.ndarray] = {(): domain.copy()}
        self._summaries: dict[tuple[Path, int], SplitSummary] = {}

    def indices(self, path: Path) -> np.ndarray:
        return self._indices[path]

    def bounds(self, path: Path) -> np.ndarray:
        return self._bounds[path]

    def directions(self, path: Path) -> np.ndarray:
        if self.max_directions >= self.d:
            return np.arange(self.d, dtype=int)
        seed_words = [self.direction_seed, len(path)]
        seed_words.extend(2 * direction + side for direction, side in path)
        rng = np.random.default_rng(np.random.SeedSequence(seed_words))
        return np.sort(rng.choice(self.d, self.max_directions, replace=False))

    def summary(self, path: Path, direction: int) -> SplitSummary:
        key = (path, int(direction))
        if key in self._summaries:
            return self._summaries[key]
        parent_indices = self._indices[path]
        parent_bounds = self._bounds[path]
        midpoint = float(np.mean(parent_bounds[direction]))
        values = self.X[parent_indices, direction]
        left = parent_indices[values <= midpoint]
        right = parent_indices[values > midpoint]

        left_path = _child_path(path, direction, 0)
        right_path = _child_path(path, direction, 1)
        left_bounds = parent_bounds.copy()
        right_bounds = parent_bounds.copy()
        left_bounds[direction, 1] = midpoint
        right_bounds[direction, 0] = midpoint
        self._indices.setdefault(left_path, left)
        self._indices.setdefault(right_path, right)
        self._bounds.setdefault(left_path, left_bounds)
        self._bounds.setdefault(right_path, right_bounds)

        if len(left) and len(right):
            parent_values = self.tree_scores[parent_indices]
            left_values = self.tree_scores[left]
            right_values = self.tree_scores[right]
            parent_range = float(np.max(parent_values) - np.min(parent_values))
            child_mean_range = 0.5 * (
                float(np.max(left_values) - np.min(left_values))
                + float(np.max(right_values) - np.min(right_values))
            )
            value = parent_range - child_mean_range
            if child_mean_range > 0:
                rate = parent_range / child_mean_range - 1.0
            elif parent_range > 0:
                rate = math.inf
            else:
                rate = -math.inf
        else:
            value = -math.inf
            rate = -math.inf
        answer = SplitSummary(left=left, right=right, value=value, rate=rate)
        self._summaries[key] = answer
        return answer

    def fit(self, hidden_x: Optional[np.ndarray] = None) -> "FittedDyadicTree":
        if hidden_x is not None:
            hidden_x = np.asarray(hidden_x, dtype=float)
            if hidden_x.shape != (self.d,):
                raise ValueError("hidden_x must have shape (d,)")
            # The initial bounds fix the midpoint locations. Outermost leaves
            # extend beyond these bounds under the <= / > routing rule, so a
            # hidden covariate outside training extrema remains well defined.

        splits: dict[Path, int] = {}
        leaves: set[Path] = {()}
        contains_hidden: dict[Path, bool] = {(): hidden_x is not None}
        heap: list[tuple[float, Path, int]] = []

        def add_candidate(path: Path) -> None:
            if len(path) >= self.max_depth:
                return
            has_x = contains_hidden[path]
            bounds = self._bounds[path]
            best_value: Optional[float] = None
            best_direction: Optional[int] = None
            for direction_value in self.directions(path):
                direction = int(direction_value)
                summary = self.summary(path, direction)
                hidden_left = bool(
                    has_x and hidden_x[direction] <= float(np.mean(bounds[direction]))
                )
                left_count = len(summary.left) + int(hidden_left)
                right_count = len(summary.right) + int(has_x and not hidden_left)
                if left_count < self.m or right_count < self.m:
                    continue
                if summary.rate <= self.threshold:
                    continue
                if (
                    best_value is None
                    or summary.value > best_value
                    or (
                        summary.value == best_value
                        and direction < int(best_direction)
                    )
                ):
                    best_value = summary.value
                    best_direction = direction
            if best_direction is not None:
                heapq.heappush(heap, (-float(best_value), path, best_direction))

        add_candidate(())
        while heap and len(leaves) < self.max_leaves:
            _, path, direction = heapq.heappop(heap)
            if path not in leaves:
                continue
            leaves.remove(path)
            splits[path] = int(direction)
            left_path = _child_path(path, direction, 0)
            right_path = _child_path(path, direction, 1)
            self.summary(path, direction)
            leaves.add(left_path)
            leaves.add(right_path)
            has_x = contains_hidden.pop(path)
            if has_x:
                midpoint = float(np.mean(self._bounds[path][direction]))
                hidden_left = bool(hidden_x[direction] <= midpoint)
            else:
                hidden_left = False
            contains_hidden[left_path] = bool(has_x and hidden_left)
            contains_hidden[right_path] = bool(has_x and not hidden_left)
            if len(leaves) >= self.max_leaves:
                break
            add_candidate(left_path)
            add_candidate(right_path)

        return FittedDyadicTree(
            cache=self,
            splits=splits,
            leaves=leaves,
            hidden_leaf=(
                next((path for path, flag in contains_hidden.items() if flag), None)
                if hidden_x is not None
                else None
            ),
        )

    @property
    def cached_regions(self) -> int:
        return len(self._indices)

    @property
    def cached_summaries(self) -> int:
        return len(self._summaries)


@dataclass
class FittedDyadicTree:
    cache: DyadicSummaryCache
    splits: dict[Path, int]
    leaves: set[Path]
    hidden_leaf: Optional[Path] = None

    def route_one(self, x: np.ndarray) -> Path:
        x = np.asarray(x, dtype=float)
        path: Path = ()
        while path in self.splits:
            direction = self.splits[path]
            midpoint = float(np.mean(self.cache.bounds(path)[direction]))
            path = _child_path(path, direction, int(x[direction] > midpoint))
        return path

    def apply(self, X: np.ndarray) -> list[Path]:
        return [self.route_one(row) for row in np.asarray(X, dtype=float)]

    def calibration_indices(self, path: Path) -> np.ndarray:
        return self.cache.indices(path)

    def signature(self) -> tuple[tuple[Path, int], ...]:
        return tuple(sorted(self.splits.items()))

    @property
    def n_leaves(self) -> int:
        return len(self.leaves)

    @property
    def depth(self) -> int:
        return max((len(path) for path in self.leaves), default=0)
