import numpy as np
from typing import List, Dict, Union

def validate_constructor(domain: np.ndarray,
                         min_samples_leaf: int = 20,
                         max_leaves: int = 20,
                         max_depth: int = 8,
                         threshold: float = 0.05):
    # validate domain
    try:
        domain = np.array(domain)
        if domain.ndim != 2 or domain.shape[1] != 2:
            raise ValueError("Domain must be a 2D array-like with shape (d, 2)")
        if not np.all(domain[:, 0] < domain[:, 1]):
            raise ValueError("Lower bounds must be strictly less than upper bounds in domain")
    except Exception as e:
        raise ValueError(f"Invalid domain: {str(e)}")

    if not isinstance(min_samples_leaf, int) or min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be an integer >= 1")

    if not isinstance(max_leaves, int) or max_leaves < 1:
        raise ValueError("max_leaves must be an integer >= 1")

    if not isinstance(max_depth, int) or max_depth < 1:
        raise ValueError("max_depth must be an integer >= 1")

    if not isinstance(threshold, (int, float)) or threshold < 0:
        raise ValueError("threshold must be a non-negative number")

    return True

def validate_calibrate(X_calib: np.ndarray, scores: np.ndarray, alpha: float):
    try:
        X_calib = np.array(X_calib)
        if X_calib.ndim != 2:
            raise ValueError("X_calib must be a 2D array-like")
    except Exception as e:
        raise ValueError(f"Invalid X_calib: {str(e)}")

    try:
        scores = np.array(scores)
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D array-like")
    except Exception as e:
        raise ValueError(f"Invalid scores: {str(e)}")

    if len(X_calib) != len(scores):
        raise ValueError(f"X_calib and scores must have the same number of samples. "
                         f"Got {len(X_calib)} and {len(scores)} respectively.")

    if not isinstance(alpha, float) or not 0 < alpha < 1:
        raise ValueError("alpha must be a float between 0 and 1 (exclusive)")

    return True

def validate_test_set_regression(X_test: np.ndarray, y_test_pred: np.ndarray):

    try:
        X_test = np.array(X_test)
        if X_test.ndim != 2:
            raise ValueError("X_test must be a 2D array-like")
    except Exception as e:
        raise ValueError(f"Invalid X_test: {str(e)}")

    try:
        y_test_pred = np.array(y_test_pred)
        if y_test_pred.ndim != 1:
            raise ValueError("y_test_pred must be a 1D array-like")
    except Exception as e:
        raise ValueError(f"Invalid y_test_pred: {str(e)}")

    if len(X_test) != len(y_test_pred):
        raise ValueError(f"X_test and y_test_pred must have the same number of samples. "
                         f"Got {len(X_test)} and {len(y_test_pred)} respectively.")

    if len(X_test) == 0:
        raise ValueError("X_test cannot be empty")

    return True

def validate_test_set_classification(X_test: np.ndarray, y_test_pred: Union[np.ndarray, List[Dict]] ):

    try:
        X_test = np.array(X_test)
        if X_test.ndim != 2:
            raise ValueError("X_test must be a 2D array-like")
    except Exception as e:
        raise ValueError(f"Invalid X_test: {str(e)}")

    try:
        y_test_pred = np.array(y_test_pred)
    except:
        if not isinstance(y_test_pred, list):
            raise ValueError("y_test_pred must be array-like (numpy array or list)")

    if len(X_test) != len(y_test_pred):
        raise ValueError(f"X_test and y_test_pred must have the same number of samples. "
                         f"Got {len(X_test)} and {len(y_test_pred)} respectively.")

    if len(X_test) == 0:
        raise ValueError("X_test cannot be empty")

    class_names = None
    for i, pred_dict in enumerate(y_test_pred):
        if not isinstance(pred_dict, dict):
            raise ValueError(f"Each element in y_test_pred must be a dictionary. "
                             f"Got {type(pred_dict)} at index {i}")

        if class_names is None:
            class_names = set(pred_dict.keys())
        elif set(pred_dict.keys()) != class_names:
            raise ValueError(f"All dictionaries in y_test_pred must have the same keys (class names). "
                             f"Mismatch at index {i}")

        prob_sum = sum(pred_dict.values())
        if not np.isclose(prob_sum, 1.0, atol=1e-4):
            raise ValueError(f"Probabilities in y_test_pred must sum to 1. "
                             f"Got sum {prob_sum} at index {i}")

    return True
