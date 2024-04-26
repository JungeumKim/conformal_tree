import numpy as np
from typing import Any
from collections import deque

from conformal_tree._utils import tree_utils
from sklearn.tree import _tree as ctree
from IPython.core.debugger import set_trace


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
                m = np.sum(membership == idx)
                C = np.quantile(scores_subset, np.ceil((1-alpha)*(m+1))/m)
                self.offsets[idx] = C

                y_lb -= C*(membership == idx)
                y_ub += C*(membership == idx)

            c_gap = np.minimum(np.abs(y_lb-y_calib), np.abs(y_ub-y_calib))
            if it != n_its - 1:
                self.tree_model, membership = tree_utils.tree_membership(
                    # X_calib, c_gap, max_depth=20, max_leaf_nodes=4**(it+1), min_samples_leaf=20
                    X_calib, c_gap, max_depth=4, max_leaf_nodes=8, min_samples_leaf=20
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

        return y_lb, y_ub

class ConformalForest:

    model: Any
    forest_model: Any
    domain: np.ndarray
    offsets: Any

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

        # Calibrate by averaging: fit a random forest,
        #
        scores = np.abs(y_calib - y_model)
        self.forest_model, forest_membership = tree_utils.forest_membership(X_calib, scores)

        self.offsets = []
        for i in range(self.forest_model.n_estimators):
            membership = forest_membership[:,i]
            bin_idx = np.unique(membership)
            c_offsets = {}
            for idx in bin_idx:
                scores_subset = scores[membership == idx]
                m = np.sum(membership == idx)
                C = np.quantile(scores_subset, np.ceil((1-alpha)*(m+1))/m)
                c_offsets[idx] = C

            self.offsets.append(c_offsets)

    def tree_interval(self, X_test: np.ndarray, tree_idx=0):
        """Return a prediction interval for test data, from a single tree

        Args:
            X (np.ndarray): N x D array of test data
        Returns:
            np.ndarray: N x 2 array of upper and lower bounds for each x
        """

        offsets = self.offsets[tree_idx]

        y_test_model = self.model.predict(X_test)

        y_lb = np.copy(y_test_model)
        y_ub = np.copy(y_test_model)

        test_forest_leaf_idxs = self.forest_model.apply(X_test)

        test_tree_leaf_idxs = test_forest_leaf_idxs[:,tree_idx]

        lookup = np.vectorize(offsets.get)
        test_offsets = lookup(test_tree_leaf_idxs)

        y_lb -= test_offsets
        y_ub += test_offsets

        return y_lb, y_ub

    def combined_test_interval(self,
                               X_test : np.ndarray,
                               strategy: str = "average"):

        if strategy not in ["average"]:
            raise ValueError("Strategy not recognized")

        y_lbs = np.zeros((self.forest_model.n_estimators, X_test.shape[0]))
        y_ubs = np.zeros((self.forest_model.n_estimators, X_test.shape[0]))

        for i in range(self.forest_model.n_estimators):
            _y_lb, _y_ub = self.tree_interval(X_test, i)
            y_lbs[i] = _y_lb
            y_ubs[i] = _y_ub

        # print(y_lbs.shape)
        # print(y_ubs.shape)

        return y_lbs, y_ubs

        if strategy == "average":
            y_lb = np.mean(y_lbs, axis=0)
            y_ub = np.mean(y_ubs, axis=0)

        elif strategy == "vote":
            raise ValueError("Not implemented yet")

        return y_lb, y_ub


def majority_vote(a: np.ndarray,
                  b: np.ndarray,
                  w: np.ndarray = None,
                  tau: float = 0.5):

    if w is None:
        w = np.ones(a.shape)

    lower = []
    upper = []

    q = np.sort(np.concatenate((a,b)))
    for i in range(1,2*len(a)):
        if np.sum(w*((a <= (q[i-1] + q[i])/2 ) & (b >= (q[i-1] + q[i])/2 ))) > tau:
            lower.append(q[i-1])
            j = i
            while (j < 2*len(a) and
                np.sum(w*((a <= (q[j-1] + q[j])/2 ) & (b >= (q[j-1] + q[j])/2 ))) > tau):

                j += 1
            i = j
            upper.append(q[i-1])
        else:
            i += 1

    return lower, upper


def test_forest():
    X = np.random.uniform(0,1,200)
    y = np.random.normal(0,10,200) + X
    X = X.reshape(200,1)

    from sklearn import ensemble
    rfm = ensemble.RandomForestRegressor()
    rfm.fit(X,y)
    cf = ConformalForest(rfm, np.array([[0,1]]))

    cf.calibrate(X,y, rfm.predict(X), 0.1)

    X_test = np.random.uniform(0,1,200)
    y_test = np.random.normal(0,10,200) + X_test
    X_test = X_test.reshape(200,1)

    return cf.combined_test_interval(X_test)



class ConformalClassifier:

    tree_object: Any
    tree_model: Any
    bins: Any
    deltas: Any
    domain: np.ndarray
    offsets: np.ndarray
    classification_model: Any

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

    def calibrate(self, X_calib: np.ndarray,
                  y_calib: np.ndarray,
                  y_model: np.ndarray,
                  alpha: float):
        """Calibrate to calibration data

        Args:
            X_calib (np.ndarray): covariates in calibration data
            y_calib (np.ndarray): responses in calibration data
            y_model (np.ndarray): estimated mean of model
        """



        scores = np.abs(y_calib-y_model)

        sorted_idxs = np.argsort(scores)

        from sklearn.tree import DecisionTreeRegressor
        _tree = DecisionTreeRegressor(max_leaf_nodes = 10, min_samples_leaf=20)
        pseudo_X = np.arange(X_calib.shape[0]).reshape(-1,1)
        _tree.fit(pseudo_X, scores[sorted_idxs])

        leaf_class = np.zeros(len(sorted_idxs))
        leaf_class[sorted_idxs] = _tree.apply(pseudo_X)

        from sklearn.ensemble import RandomForestClassifier
        self.classification_model = RandomForestClassifier(n_estimators=10)
        self.classification_model.fit(X_calib, leaf_class)

        unique_classes = np.unique(leaf_class)
        self.offsets = {}

        print(f"Total classes: {len(unique_classes)}")
        for i,cls in enumerate(unique_classes):
            scores_subset = scores[leaf_class == cls]
            m = np.sum(leaf_class == cls)
            C = np.quantile(scores_subset, np.ceil((1-alpha)*(m+1))/m)
            self.offsets[cls] = C

        pass

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

        test_classes = self.classification_model.predict(X_test)

        lookup = np.vectorize(self.offsets.get)
        test_offsets = lookup(test_classes)


        y_lb = y_lb.flatten() - test_offsets.flatten()
        y_ub = y_ub.flatten() + test_offsets.flatten()

        return y_lb, y_ub
