import numpy as np
from typing import Any
from collections import deque
from typing import List, Dict

from ._utils import tree_utils
from ._utils import validation_utils

class ConformalTree:
    """Parent class for conformal tree
    """

    def __init__(self, domain, min_samples_leaf=20, max_leaves = 20,
                 max_depth = 8, threshold=0.05, *args, **kwargs) -> None:
        """Conformal tree constructor

        Args:
            domain (np.ndarray): bounding box of domain, shape (d,2)
            min_samples_leaf (int): minimum number of samples per leaf for the robust tree
            max_leaves (int): maximum number of leaf nodes for the robust tree
            max_depth (int): maximum depth for the robust tree
            threshold (float): minimum relative improvement for splitting in the robust tree
        """

        validation_utils.validate_constructor(domain, min_samples_leaf, max_leaves, max_depth, threshold)

        self.domain = domain
        self.offsets = {}
        self.m = min_samples_leaf
        self.K = max_leaves
        self.max_depth = max_depth
        self.threshold = threshold


    def calibrate(self, X_calib: np.ndarray, scores: np.ndarray,  alpha: float):
        """Calibrate to calibration data

        Args:
            X_calib (np.ndarray): covariates in calibration data, shape (n,d)
            scores (np.ndarray): conformity scores, shape (n,)
            alpha (float): desired confidence level
        """

        validation_utils.validate_calibrate(X_calib, scores, alpha)

        self.tree_model, membership = tree_utils.multi_dyadic_tree_membership(X_calib, scores, self.domain,
                                                                              max_depth=self.max_depth,
                                                                              max_leaf_nodes=self.K,
                                                                              min_samples_leaf=self.m,
                                                                              threshold=self.threshold)

        bin_idx = np.unique(membership)

        for idx in bin_idx:
            scores_subset = scores[membership == idx]
            m = np.sum(membership == idx)
            qtile = np.min((1,np.ceil((1-alpha)*(m-2) +1)/m))
            C = np.quantile(scores_subset, qtile)
            self.offsets[idx] = C

    def test_set(self):
        raise NotImplementedError("Implemented in child classes.")

class ConformalTreeRegression(ConformalTree):
    """Child class for conformal tree for regression models"""

    tree_model: Any
    domain: np.ndarray
    offsets: dict

    def test_set(self, X_test: np.ndarray, y_test_pred: np.ndarray):
        """Return a prediction interval for test data. Computes conformal sets for absolute error conformity score.

        Args:
            X_test (np.ndarray): (n,d) array of test data
            y_test_pred (np.ndarray): (n,) array of model predictions on test data
        Returns:
            Tuple[np.ndarray]: tuple of 2 length (n,) np.ndarray of lower and upper bounds
        """
        validation_utils.validate_test_set_regression(X_test, y_test_pred)

        y_lb = np.copy(y_test_pred)
        y_ub = np.copy(y_test_pred)

        test_leaf_idxs = self.tree_model.apply(X_test)

        lookup = np.vectorize(self.offsets.get)
        test_offsets = lookup(test_leaf_idxs)

        y_lb -= test_offsets
        y_ub += test_offsets

        return y_lb, y_ub



class ConformalTreeClassification(ConformalTree):

    tree_model: Any
    deltas: Any
    domain: np.ndarray
    offsets: dict

    def test_set(self, X_test: np.ndarray, y_test_pred: List[Dict]):
        """Return a prediction set for test data. Computes conformal sets for classification conformity score.

        Args:
            X_test (np.ndarray): (n,d) array of test data
            y_test_pred (Union[np.ndarray, List[Dict]]): Array-like of length n of model predictions on test data
                Each dictionary has class names as keys and assigned probabilities as values, representing a simplex
        Returns:
            List[List]: List of length n of prediction sets, each of which is a list of class names
        """

        validation_utils.validate_test_set_classification(X_test, y_test_pred)

        test_sets = []
        for i in range(X_test.shape[0]):
            X_test_pt = X_test[i]

            y_test_probs = y_test_pred[i]

            test_leaf_idx = self.tree_model.apply(X_test_pt.reshape(1,-1))

            lookup = np.vectorize(self.offsets.get)
            test_offset = lookup(test_leaf_idx)

            class_nonconformity_scores = {class_: 1 - prob for class_, prob in y_test_probs.items()}
            filtered_scores = {class_: score for class_, score in class_nonconformity_scores.items() if score <= test_offset + 2*np.finfo(np.float32).eps} #for floating point issues
            test_sets.append(list(filtered_scores.keys()))

        return test_sets


