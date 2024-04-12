import numpy as np
from typing import Any
from collections import deque

from conformal_tree._utils import tree_utils

class Tree:
    """Parent class for wrapper for various tree implementations for standardized usage"""

    length: int
    node_indices: list
    dim: int
    support: np.ndarray
    _tree: Any

    leaf_boundaries: np.ndarray
    leaf_intensities: np.ndarray

    leaf_acceptances: np.ndarray
    leaf_rejections: np.ndarray

    def __init__(self, X, y, support):
        self.support = support
        self.leaf_boundaries, self.leaf_intensities = self._get_leaf_info()

        self.leaf_acceptances = np.zeros(len(self.leaf_boundaries))
        self.leaf_rejections = np.zeros(len(self.leaf_boundaries))

        for i,hrect in enumerate(self.leaf_boundaries):
            relevant_idxs = get_indices_in_hyperrect(X, hrect)
            self.leaf_acceptances[i] = np.sum(y[relevant_idxs] == 1)
            self.leaf_rejections[i] = np.sum(y[relevant_idxs] == 0)



    def _is_leaf(self, _):
        raise NotImplementedError("Implement in a child class")

    def _left_child(self, _):
        raise NotImplementedError("Implement in a child class")

    def _right_child(self, _):
        raise NotImplementedError("Implement in a child class")

    def _split_var(self, _):
        raise NotImplementedError("Implement in a child class")

    def _value(self, _):
        raise NotImplementedError("Implement in a child class")

    def _intensity(self, _):
        raise NotImplementedError("Implement in a child class")

    def _get_leaf_info(self):
        """Compute final decision rule for each node in tree"""

        aabbs = {k:AABB(self.dim, self.support) for k in self.node_indices}

        queue = deque([0])
        while queue:
            node_idx = queue.pop()

            if not self._is_leaf(node_idx):
                l_child_idx = self._left_child(node_idx)
                r_child_idx = self._right_child(node_idx)

                aabbs[l_child_idx], aabbs[r_child_idx] = aabbs[node_idx].split(self._split_var(node_idx), self._value(node_idx))
                queue.extend([l_child_idx, r_child_idx])

        leaf_info = [
                        (aabbs[node_idx].limits, self._intensity(node_idx))
                        for node_idx in self.node_indices if self._is_leaf(node_idx)
                    ]
        return zip(*leaf_info)

class AABB:
    """Axis-aligned bounding box"""
    def __init__(self, dim, support=None):
        self.support = support
        if support is None:
            self.limits = np.array([[-np.inf, np.inf]] * dim)
        else:
            self.limits = np.array(support, dtype=float)

    def __repr__(self):
        return f"AABB: {self.limits}"

    def split(self, f, v):
        left = AABB(self.limits.shape[0], self.support)
        right = AABB(self.limits.shape[0], self.support)
        left.limits = self.limits.copy()
        right.limits = self.limits.copy()

        left.limits[f, 1] = v
        right.limits[f, 0] = v

        return left, right



def get_indices_in_hyperrect(points, hrect):
    """Returns the indices in points where the point lies inside hyperrectangle hrect"""
    return np.logical_and(hrect[:,0] < points, hrect[:,1] > points).all(axis=1)

class CTree:

    tree_model: Any
    bins: Any
    deltas: Any

    def __init__(self, tree_model, *args, **kwargs):
        self.tree = tree_model
        pass

    def fit_tree(self, X_calib, y_calib):
        """Fits tree to calibration data

        Args:
            X_calib (np.ndarray): (N,D) array of calibration data
            y_calib (np.ndarray): (N,) array of response data
        """
        self.tree_model.fit(X_calib, y_calib)

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray, y_model: np.ndarray, alpha: float, n_its: int = 5):
        """Calibrate to calibration data

        Args:
            X_calib (np.ndarray): covariates in calibration data
            y_calib (np.ndarray): responses in calibration data
            y_model (np.ndarray): estimated mean of model
        """


        tree_model, membership = tree_utils.tree_membership(X_calib, X_calib[:,0]*0)
        bin_idx = np.unique(membership)
        scores = np.abs(y_calib-y_model)

        for it in range(n_its):

            y_lb = np.copy(y_model)
            y_ub = np.copy(y_model)



            for idx in bin_idx:
                scores_subset = scores[membership == idx]
                # print(scores_subset)
                C = np.quantile(scores_subset, 1-alpha)

                y_lb -= C*(membership == idx)
                y_ub += C*(membership == idx)

            c_gap = np.minimum(np.abs(y_lb-y_calib), np.abs(y_ub-y_calib))
            tree_model, membership = tree_utils.tree_membership(
                X_calib, c_gap, max_depth=20, max_leaf_nodes=(it+1)*2, min_samples_leaf=20
            )
            # print(f"MEMB: {membership}")
            bin_idx = np.unique(membership)


        return y_lb, y_ub, c_gap


    def interval(self, X: np.ndarray):
        """Return interval

        Args:
            X (np.ndarray): N x D array of test data
        Returns:
            np.ndarray: N x 2 array of upper and lower bounds for each x
        """
        pass

