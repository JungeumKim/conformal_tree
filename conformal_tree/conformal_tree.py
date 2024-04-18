import numpy as np
from typing import Any
from collections import deque

from conformal_tree._utils import tree_utils
from sklearn.tree import _tree as ctree

"""NOTE: Currently Tree, AABB are not used. Only CTree class is used for now"""

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
        self.leaf_boundaries = np.array(self.leaf_boundaries)

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


class CARTTree(Tree):
    """Wrapper for logistic CART implementation in sklearn.tree.DecisionTreeClassifier"""

    def __init__(self, X, y, tree, support):
        dim = support.shape[0]
        self._tree = tree
        self.length = tree.node_count
        self.node_indices = np.arange(self.length)
        if dim is not None:
            self.dim = np.max(tree.feature) + 1
        else:
            self.dim = dim
        super().__init__(X, y, support)

    def _is_leaf(self, node_idx):
        return self._tree.children_left[node_idx] == ctree.TREE_LEAF

    def _left_child(self, node_idx):
        return self._tree.children_left[node_idx]

    def _right_child(self, node_idx):
        return self._tree.children_right[node_idx]

    def _split_var(self, node_idx):
        return self._tree.feature[node_idx]

    def _value(self, node_idx):
        return self._tree.threshold[node_idx]

    def _intensity(self, node_idx):
        return self._tree.value[node_idx, 0, 0]





def get_indices_in_hyperrect(points, hrect):
    """Returns the indices in points where the point lies inside hyperrectangle hrect"""
    return np.logical_and(hrect[:,0] < points, hrect[:,1] > points).all(axis=1)

class CTree:

    tree_object: Any
    tree_model: Any
    bins: Any
    deltas: Any
    domain: np.ndarray
    offsets: np.ndarray

    def __init__(self, model, domain, *args, **kwargs):
        """Conformal tree constructor

        Args:
            model (sklearn.base): Model endowed with predict function
            domain (np.ndarray): bounding box of domain
        """
        self.model = model
        self.domain = domain
        self.offsets = None
        pass

    def calibrate(self, X_calib: np.ndarray, y_calib: np.ndarray, y_model: np.ndarray, alpha: float, n_its: int = 5):
        """Calibrate to calibration data

        Args:
            X_calib (np.ndarray): covariates in calibration data
            y_calib (np.ndarray): responses in calibration data
            y_model (np.ndarray): estimated mean of model
        """


        self.tree_model, membership = tree_utils.tree_membership(X_calib, X_calib[:,0]*0)

        bin_idx = np.unique(membership)
        scores = np.abs(y_calib-y_model)

        for it in range(n_its):

            y_lb = np.copy(y_model)
            y_ub = np.copy(y_model)

            self.offsets = {}

            # print(f"bidx: {bin_idx}")
            # print(self.tree_model.get_n_leaves())

            for idx in bin_idx:
                scores_subset = scores[membership == idx]
                # print(scores_subset)
                C = np.quantile(scores_subset, 1-alpha)
                self.offsets[idx] = C

                y_lb -= C*(membership == idx)
                y_ub += C*(membership == idx)

            c_gap = np.minimum(np.abs(y_lb-y_calib), np.abs(y_ub-y_calib))
            if it != n_its - 1:
                self.tree_model, membership = tree_utils.tree_membership(
                    X_calib, c_gap, max_depth=20, max_leaf_nodes=(it+1)*2, min_samples_leaf=20
                )
                # print(f"MEMB: {membership}")
                bin_idx = np.unique(membership)

        self.tree_object = CARTTree(X_calib, y_calib, self.tree_model.tree_, self.domain)

        return y_lb, y_ub, c_gap


    def test_interval(self, X_test: np.ndarray):
        """Return a prediction interval for test data

        Args:
            X (np.ndarray): N x D array of test data
        Returns:
            np.ndarray: N x 2 array of upper and lower bounds for each x
        """

        y_test_model = self.model.predict(X_test)

        y_lb = np.copy(y_test_model)
        y_ub = np.copy(y_test_model)

        test_leaf_idxs = self.tree_model.apply(X_test)
        # print(f"lenuniq: {np.unique(test_leaf_idxs)}")
        # print(f"TLI{test_leaf_idxs}")

        lookup = np.vectorize(self.offsets.get)
        test_offsets = lookup(test_leaf_idxs)

        # print(f"TO: {test_offsets}")

        y_lb -= test_offsets
        y_ub += test_offsets

        # for i,x_t in enumerate(X_test):
        #     bin_x_t = find_bounding_box(self.tree_object.leaf_boundaries, x_t)
        #     C = self.offsets[bin_x_t]
        #     y_lb[i] = y_test_model[i] - C
        #     y_ub[i] = y_test_model[i] + C

        return y_lb, y_ub

def find_bounding_box(bounding_boxes, point):
    """
    Determines the index of the bounding box that contains the given point.

    Parameters:
        bounding_boxes (np.ndarray): An array of shape (m, d, 2) where m is the number of boxes,
                                     d is the dimensionality, and the last dimension stores [lower, upper] bounds.
        point (np.ndarray): An array of shape (d,) representing the point.

    Returns:
        int: The index (1-based) of the first bounding box that contains the point, or -1 if no such box exists.
    """
    # Check if the point lies between the lower and upper bounds for each dimension in each bounding box
    lower_bounds = bounding_boxes[:,:,0]  # Extract all lower bounds
    upper_bounds = bounding_boxes[:,:,1]  # Extract all upper bounds
    # Create a mask to check containment across all dimensions
    contained = np.all((lower_bounds <= point) & (upper_bounds >= point), axis=1)

    # Find the first bounding box in which the point is contained
    indices = np.where(contained)[0]
    if indices.size > 0:
        return indices[0]
    else:
        return -1
