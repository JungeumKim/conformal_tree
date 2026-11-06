"""Dyadic conformal forests that include each test covariate in node counts."""
from __future__ import annotations

import math
from numbers import Integral

import numpy as np

from ._utils.fast_dyadic import DyadicSummaryCache, conformal_threshold


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def stability_bound(n, min_samples_leaf, max_leaves, max_directions):
    """One-term score-hiding bound for the hallucination procedure."""
    n = _positive_integer(n, "n")
    m = _positive_integer(min_samples_leaf, "min_samples_leaf")
    k = _positive_integer(max_leaves, "max_leaves")
    q = _positive_integer(max_directions, "max_directions")
    if m < 2:
        raise ValueError("min_samples_leaf must be at least 2")
    if k == 1:
        return 0.0
    if n+1 <= m*k:
        raise ValueError("The bound requires n+1 > min_samples_leaf * max_leaves")
    return min(1.0, (4*q*k-6*q+2)/(n+1))


def choose_leaf_cap(n, min_samples_leaf, max_directions, upper_cap, budget=0.1):
    """Largest admissible integer cap up to upper_cap with delta <= budget."""
    n = _positive_integer(n, "n")
    m = _positive_integer(min_samples_leaf, "min_samples_leaf")
    q = _positive_integer(max_directions, "max_directions")
    upper_cap = _positive_integer(upper_cap, "upper_cap")
    if m < 2 or not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Require min_samples_leaf >= 2 and budget in [0, 1]")
    for k in range(upper_cap, 1, -1):
        if m*k < n+1 and stability_bound(n, m, k, q) <= budget:
            return k
    return 1


def aggregate_offsets(offsets, threshold):
    """Half-width selected by the strict vote rule: fraction containing y > threshold."""
    offsets = np.asarray(offsets, dtype=float)
    if offsets.ndim != 2 or offsets.shape[0] == 0 or np.isnan(offsets).any() or np.any(offsets < 0):
        raise ValueError("offsets must be a nonnegative (trees, observations) matrix")
    threshold = np.broadcast_to(np.asarray(threshold, dtype=float), (offsets.shape[1],))
    if not np.isfinite(threshold).all() or np.any((threshold < 0) | (threshold >= 1)):
        raise ValueError("threshold must lie in [0, 1)")
    ranks = len(offsets)-np.floor(len(offsets)*threshold).astype(int)-1
    return np.sort(offsets, axis=0)[ranks, np.arange(offsets.shape[1])]


def _ordinary_offsets(tree, X_test, scores, alpha):
    paths = sorted(tree.leaves)
    labels = {path: i for i, path in enumerate(paths)}
    ids = np.empty(len(X_test), dtype=int)
    pending = [((), np.arange(len(X_test)))]
    while pending:
        path, indices = pending.pop()
        if path not in tree.splits:
            ids[indices] = labels[path]
            continue
        direction = tree.splits[path]
        cut = np.mean(tree.cache.bounds(path)[direction])
        left = X_test[indices, direction] <= cut
        pending.append((path+((direction, 0),), indices[left]))
        pending.append((path+((direction, 1),), indices[~left]))
    quantiles = np.array([conformal_threshold(scores[tree.calibration_indices(path)], alpha)
                          for path in paths])
    return quantiles[ids]


def _possible_change_mask(tree, X_test):
    cache = tree.cache
    affected = np.zeros(len(X_test), dtype=bool)
    if cache.max_leaves == 1:
        return affected
    masks = {(): np.ones(len(X_test), dtype=bool)}

    def belongs(path):
        if path not in masks:
            parent = path[:-1]
            direction, side = path[-1]
            left = X_test[:, direction] <= np.mean(cache.bounds(parent)[direction])
            masks[path] = belongs(parent) & (left if side == 0 else ~left)
        return masks[path]

    # Freeze this trace when fitting, before test-specific fits grow the cache.
    for (path, direction), summary in tree.ordinary_summaries_:
        if summary.rate <= cache.threshold:
            continue
        nl, nr = len(summary.left), len(summary.right)
        cut = np.mean(cache.bounds(path)[direction])
        if nl == cache.m-1 and nr >= cache.m:
            affected |= belongs(path) & (X_test[:, direction] <= cut)
        if nr == cache.m-1 and nl >= cache.m:
            affected |= belongs(path) & (X_test[:, direction] > cut)
    return affected


class ConformalForest:
    """Hallucination trees with fixed random direction subsets and no bootstrap."""

    def __init__(self, min_samples_leaf=20, max_leaves=20, max_directions="sqrt",
                 n_trees=100, max_depth=8, threshold=0.05, seed=0):
        self.min_samples_leaf = _positive_integer(min_samples_leaf, "min_samples_leaf")
        self.max_leaves = _positive_integer(max_leaves, "max_leaves")
        self.n_trees = _positive_integer(n_trees, "n_trees")
        self.max_depth = _positive_integer(max_depth, "max_depth")
        if self.min_samples_leaf < 2 or not np.isfinite(threshold) or threshold < 0:
            raise ValueError("Require min_samples_leaf >= 2 and a finite nonnegative threshold")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.max_directions = max_directions
        self.threshold = float(threshold)
        self.seed = int(seed)

    def fit(self, X_cal, scores, domain, tree_scores=None):
        """Store calibration scores and fit the trees used for cached hidden-score fits."""
        X_cal = np.array(X_cal, dtype=float, copy=True)
        scores = np.asarray(scores, dtype=float)
        domain = np.array(domain, dtype=float, copy=True)
        if X_cal.ndim != 2 or min(X_cal.shape) == 0 or scores.shape != (len(X_cal),):
            raise ValueError("Require nonempty X_cal of shape (n, d) and scores of shape (n,)")
        if not np.isfinite(X_cal).all() or not np.isfinite(scores).all() or np.any(scores < 0):
            raise ValueError("X_cal must be finite and scores must be finite and nonnegative")
        d = X_cal.shape[1]
        if domain.shape != (d, 2) or not np.isfinite(domain).all() or np.any(domain[:, 0] > domain[:, 1]):
            raise ValueError("domain must contain finite lower/upper bounds of shape (d, 2)")
        if isinstance(self.max_directions, str):
            rules = {"all": d, "sqrt": math.ceil(math.sqrt(d)),
                     "log2": max(1, math.ceil(math.log2(d)))}
            if self.max_directions not in rules:
                raise ValueError("max_directions must be 'all', 'sqrt', 'log2', or a positive integer")
            q = rules[self.max_directions]
        else:
            q = min(d, _positive_integer(self.max_directions, "max_directions"))
        if tree_scores is None:
            tree_scores = scores + np.random.default_rng(self.seed).uniform(0, 1e-12, len(scores))
        tree_scores = np.array(tree_scores, dtype=float, copy=True)
        if tree_scores.shape != scores.shape or not np.isfinite(tree_scores).all():
            raise ValueError("tree_scores must be a finite vector with the same shape as scores")
        self.n_features_in_ = d
        self.max_directions_ = q
        self.scores_ = scores.copy()
        self.delta_ = stability_bound(len(scores), self.min_samples_leaf, self.max_leaves, q)
        self.trees_ = []
        seeds = np.random.SeedSequence(self.seed).generate_state(self.n_trees)
        count = 1 if q == d else self.n_trees
        for seed in seeds[:count]:
            cache = DyadicSummaryCache(X_cal, tree_scores, domain,
                                       self.min_samples_leaf, self.max_leaves, self.max_depth,
                                       self.threshold, q, int(seed))
            tree = cache.fit()
            tree.ordinary_summaries_ = tuple(cache._summaries.items())
            self.trees_.append(tree)
        return self

    def predict_component_offsets(self, X_test, alpha=0.1):
        """Return shape (n_trees, n_test); each test X is counted separately, never scored."""
        if not hasattr(self, "trees_"):
            raise ValueError("Call fit before predicting")
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim != 2 or X_test.shape[1] != self.n_features_in_ or not np.isfinite(X_test).all():
            raise ValueError("X_test must be a finite matrix with the fitted number of columns")
        if not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("alpha must lie in (0, 1)")
        offsets = []
        for tree in self.trees_:
            q = _ordinary_offsets(tree, X_test, self.scores_, alpha)
            affected = _possible_change_mask(tree, X_test)
            signature = tree.signature()
            for i in np.flatnonzero(affected):
                hidden = tree.cache.fit(hidden_x=X_test[i])
                if hidden.signature() != signature:
                    q[i] = conformal_threshold(self.scores_[hidden.calibration_indices(hidden.hidden_leaf)], alpha)
            offsets.append(q)
        matrix = np.stack(offsets)
        if len(matrix) == 1 and self.n_trees > 1:
            matrix = np.repeat(matrix, self.n_trees, axis=0)
        return matrix

    def predict_offsets(self, X_test, alpha=0.1, aggregation="uniform", rng=None):
        """Aggregate hallucination half-widths, drawing a fresh independent U per test point."""
        if aggregation not in ("uniform", "majority"):
            raise ValueError("aggregation must be 'uniform' or 'majority'")
        matrix = self.predict_component_offsets(X_test, alpha)
        if aggregation == "majority":
            return aggregate_offsets(matrix, 0.5)
        rng = np.random.default_rng() if rng is None else rng
        return aggregate_offsets(matrix, rng.uniform(size=matrix.shape[1]))

    def predict_intervals(self, X_test, y_pred, alpha=0.1, aggregation="uniform", rng=None):
        """Return lower and upper endpoints centered on independent black-box predictions."""
        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape != (len(X_test),) or not np.isfinite(y_pred).all():
            raise ValueError("y_pred must be a finite vector of length len(X_test)")
        offsets = self.predict_offsets(X_test, alpha, aggregation, rng)
        return y_pred-offsets, y_pred+offsets
