#!/usr/bin/env python3
import numpy as np
from sklearn.base import BaseEstimator

class MultiDyadicTree:
    def __init__(self):
        self.nodes = {(0,1): (None, 0.0)}

    def __getitem__(self, key):
        if key not in self.nodes:
            raise ValueError("Tree does not contain this node")
        return self.nodes[key]

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __repr__(self):
        return f"{self.nodes}"

    def split(self, parent_key, split_dir, values):
        h, i = parent_key
        left_child_key = (h+1, 2*i - 1)
        left_child_val, right_child_val = values
        right_child_key = (h+1, 2*i)
        self.nodes[parent_key] = (split_dir, self.nodes[parent_key][1])
        self.nodes[left_child_key] = (None, left_child_val)
        self.nodes[right_child_key] = (None, right_child_val)

    def is_leaf(self, node):
        h, i = node
        return ((h+1,2*i-1) not in self.nodes) and ((h+1,2*i) not in self.nodes)

    def get_n_leaves(self):
        return np.sum(np.array([self.is_leaf(node) for node in self.nodes.keys()]))

    def get_max_depth(self):
        return np.max([node[0] for node in self.nodes])

class MultiDyadicTreeRegressor(BaseEstimator):
    def __init__(self,
                 domain,
                 criterion,
                 max_depth=30,
                 min_samples_split=1,
                 min_samples_leaf=1,
                 max_leaf_nodes=100,
                 threshold = 0.1):
        self.domain = domain
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes
        self.threshold = threshold

        self.d = domain.shape[0]

    def get_depth(self):
        if self._tree is None:
            raise ValueError("DyadicTreeRegressor has not been fitted")
        return self._tree.get_max_depth()

    def get_n_leaves(self):
        if self._tree is None:
            raise ValueError("DyadicTreeRegressor has not been fitted")
        return self._tree.get_n_leaves()

    def _fit(self, X, y, check_input=True):
        if check_input:
            check_X_params = dict(
                dtype=float, accept_sparse="no", force_all_finite=False
            )
            check_y_params = dict(ensure_2d=False, dtype=None)
            X, y = self._validate_data(
                X, y, validate_separately=(check_X_params, check_y_params)
            )


        self._build_tree(X, y)


    def _build_tree(self, X, y):
        current_n_leaves = 1
        self._tree = MultiDyadicTree()

        current_leaves = [(0,1)]

        while current_n_leaves < self.max_leaf_nodes:
            candidate_leaves_criteria = {}
            for leaf in current_leaves:
                for direc in range(self.d):
                    depth_check = leaf[0] < self.max_depth
                    min_samples_split_check = len(self._indices_in_region(X, self._get_region_from_node(leaf))) >= self.min_samples_split
                    min_samples_left_child_check = len(self._indices_in_region(X, self._get_region_from_node(left_child(leaf), direc))) >= self.min_samples_leaf
                    min_samples_right_child_check = len(self._indices_in_region(X, self._get_region_from_node(right_child(leaf), direc))) >= self.min_samples_leaf
                    if depth_check and min_samples_split_check and min_samples_left_child_check and min_samples_right_child_check:
                        _current_criterion_val, mult = self._criterion_wrapper(X, y, leaf, new_dir = direc)
                        if mult > self.threshold:
                            candidate_leaves_criteria[(leaf, direc)] = _current_criterion_val

            # candidate_leaves_criteria = {leaf: self.criterion(X, y, leaf) for leaf in current_leaves if leaf[0] < self.max_depth}
            if len(candidate_leaves_criteria) == 0:
                break

            best_leaf = max(candidate_leaves_criteria, key=candidate_leaves_criteria.get)
            (h, i), direction = best_leaf


            region_left = self._get_region_from_node((h+1,2*i-1), new_dir=direction)
            region_right = self._get_region_from_node((h+1,2*i), new_dir=direction)
            leaf_vals = (np.mean(y[self._indices_in_region(X, region_left)]), np.mean(y[self._indices_in_region(X, region_right)]))
            self._tree.split((h,i), direction, leaf_vals)
            current_leaves = [node for node in current_leaves if node != (h,i)]

            current_leaves.append((h+1,2*i-1))
            current_leaves.append((h+1,2*i))
            current_n_leaves += 1

    def _get_region_from_node(self, node, new_dir=None):
        h, i = node
        path = []

        if h==0 and i==1:
            return self.domain
        h_it = h
        i_it = i


        h_it = h
        i_it = i
        while h_it > 0:
            parent_node = (h_it - 1, int(np.ceil(i_it / 2)))
            s, v = self._tree.nodes.get(parent_node, (None, None))
            if s is None and new_dir is None:
                raise ValueError(f"The split direction is not defined for the parent node. Node: {parent_node}. Tree: {self.nodes}")
            if s is None:
                s = new_dir  # Use new_dir if the split direction is None
            path.append((s, i_it % 2))
            h_it -= 1
            i_it = int(np.ceil(i_it / 2))

        region = np.copy(self.domain)

        for (s, direction) in reversed(path):
            if s is not None:
                cdimsize = region[s, 1] - region[s, 0]
                if direction == 1:  # Left child
                    region[s, 1] -= cdimsize / 2
                else:  # Right child
                    region[s, 0] += cdimsize / 2

        return region


    def _criterion_wrapper(self, X, y, node, new_dir):
        h, i = node
        region_parent = self._get_region_from_node((h,i))
        region_left_child = self._get_region_from_node((h+1,2*i-1), new_dir=new_dir)
        region_right_child = self._get_region_from_node((h+1,2*i), new_dir=new_dir)

        parent_inds = self._indices_in_region(X, region_parent)
        l_child_inds = self._indices_in_region(X, region_left_child)
        r_child_inds = self._indices_in_region(X, region_right_child)

        return self.criterion(X, y, parent_inds, l_child_inds, r_child_inds)




    def _indices_in_region(self, X, region):
        if not (isinstance(X, np.ndarray) and isinstance(region, np.ndarray)):
            raise ValueError("Both X and region must be numpy arrays.")

        if X.shape[1] != region.shape[0]:
            raise ValueError("Dimension mismatch: X should have the same number of columns as the number of rows in region.")

        within_bounds = np.logical_and(X >= region[:, 0], X <= region[:, 1])
        within_all_bounds = np.all(within_bounds, axis=1)
        indices = np.nonzero(within_all_bounds)[0]
        return indices

    def apply(self, X):
        if self._tree is None:
            raise ValueError("DyadicTreeRegressor has not been fitted")
        leaves = [node for node in self._tree.nodes if self._tree.is_leaf(node)]
        regions = np.array([self._get_region_from_node(leaf) for leaf in leaves])

        n = X.shape[0]
        result = np.empty(n, dtype=int)

        for i in range(n):
            point = X[i]

            in_bounds = (regions[:, :, 0] <= point) & (regions[:, :, 1] >= point)

            if not in_bounds.any():
                raise ValueError(f"No region found for point {point}")

            try:
                leaf_index = np.where(in_bounds.all(axis=1))[0][0]
            except IndexError as e:
                leaf_index = 0
                print("IndexError caught")

            result[i] = leaf_index

        return result




    def plot(self,ax, color="skyblue", alpha=0.5, label=None):
        if self.domain.shape[0] > 1:
            raise ValueError("Plotting currently only supports one dimensional X space")
        leaves = []
        leaves = [node for node in self._tree.nodes if self._tree.is_leaf(node)]
        regions = [self._get_region_from_node(leaf) for leaf in leaves]
        for i,reg in enumerate(regions):
            val = self._tree[leaves[i]]
            labv = label if i == 0 else None
            _ = ax.fill_between([reg[0][0], reg[0][1]], [val, val], color=color, alpha=alpha, step='post', label=labv)

def left_child(node):
    h,i = node
    return (h+1, 2*i-1)

def right_child(node):
    h,i = node
    return (h+1, 2*i)


def criterion_variance_reduction(X, y, parent_inds, l_child_inds, r_child_inds):
    var_reduction = (np.var(y[parent_inds]) -
            (((len(l_child_inds)/len(X)) * np.var(y[l_child_inds])) +
             ((len(r_child_inds)/len(X))*np.var(y[r_child_inds]))))
    return var_reduction

def criterion_loo_variance_reduction(X, y, parent_inds, l_child_inds, r_child_inds):
    min_var_reduction = float('inf')

    for i in parent_inds:
        parent_inds_excl = [idx for idx in parent_inds if idx != i]
        l_child_inds_excl = [idx for idx in l_child_inds if idx != i]
        r_child_inds_excl = [idx for idx in r_child_inds if idx != i]

        if len(parent_inds_excl) > 0:
            total_variance = np.var(y[parent_inds_excl])
            if len(l_child_inds_excl) > 0:
                left_variance = np.var(y[l_child_inds_excl])
            else:
                left_variance = 0

            if len(r_child_inds_excl) > 0:
                right_variance = np.var(y[r_child_inds_excl])
            else:
                right_variance = 0

            weighted_variance = ((len(l_child_inds_excl) / len(parent_inds_excl)) * left_variance +
                                 (len(r_child_inds_excl) / len(parent_inds_excl)) * right_variance)

            var_reduction = total_variance - weighted_variance

            if var_reduction < min_var_reduction:
                min_var_reduction = var_reduction

    return min_var_reduction

def criterion_range_reduction(X, y, parent_inds, l_child_inds, r_child_inds):
    range_reduction = (np.max(y[parent_inds] - np.min(y[parent_inds])) -
            (((len(l_child_inds)/len(X)) * (np.max(y[l_child_inds])-np.min(y[l_child_inds]))) +
             ((len(r_child_inds)/len(X))* (np.max(y[r_child_inds])-np.min(y[r_child_inds])))))
    return range_reduction

def criterion_range_reduction_uw(X, y, parent_inds, l_child_inds, r_child_inds):
    range_reduction = ((np.max(y[parent_inds] - np.min(y[parent_inds]))) -
            (((0.5) * (np.max(y[l_child_inds])-np.min(y[l_child_inds]))) +
             ((0.5)* (np.max(y[r_child_inds])-np.min(y[r_child_inds])))))

    range_reduction_mult = ( (np.max(y[parent_inds] - np.min(y[parent_inds]))) /
                   (((0.5) * (np.max(y[l_child_inds])-np.min(y[l_child_inds]))) +
             ((0.5)* (np.max(y[r_child_inds])-np.min(y[r_child_inds])))))

    return range_reduction, range_reduction_mult-1
