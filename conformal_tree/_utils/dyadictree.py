#!/usr/bin/env python3
import numpy as np
from sklearn.base import BaseEstimator

class DyadicTree:
    def __init__(self):
        self.nodes = {(0,1): 0.0}

    def __getitem__(self, key):
        if key not in self.nodes:
            raise ValueError("Tree does not contain this node")
        return self.nodes[key]

    def __setitem__(self, key, value):
        self.nodes[key] = value

    def __repr__(self):
        return f"{self.nodes}"

    def split(self, parent_key, values):
        h, i = parent_key
        left_child_key = (h+1, 2*i - 1)
        left_child_val, right_child_val = values
        right_child_key = (h+1, 2*i)
        self.nodes[left_child_key] = left_child_val
        self.nodes[right_child_key] = right_child_val

    def is_leaf(self, node):
        h, i = node
        return ((h+1,2*i-1) not in self.nodes) and ((h+1,2*i) not in self.nodes)

    def get_n_leaves(self):
        return np.sum(np.array([self.is_leaf(node) for node in self.nodes.keys()]))

    def get_max_depth(self):
        return np.max([node[0] for node in self.nodes])

class DyadicTreeRegressor(BaseEstimator):
    def __init__(self,
                 domain,
                 criterion,
                 max_depth,
                 min_samples_split,
                 min_samples_leaf,
                 max_leaf_nodes):
        self.domain = domain
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_leaf_nodes = max_leaf_nodes

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
        _tree = DyadicTree()

        current_leaves = [(0,1)]

        while current_n_leaves < self.max_leaf_nodes:
            candidate_leaves_criteria = {}
            for leaf in current_leaves:
                depth_check = leaf[0] < self.max_depth
                min_samples_split_check = len(self._indices_in_region(X, self._get_region_from_node(leaf))) >= self.min_samples_split
                min_samples_left_child_check = len(self._indices_in_region(X, self._get_region_from_node(left_child(leaf)))) >= self.min_samples_split
                min_samples_right_child_check = len(self._indices_in_region(X, self._get_region_from_node(right_child(leaf)))) >= self.min_samples_split
                if depth_check and min_samples_split_check and min_samples_left_child_check and min_samples_right_child_check:
                    candidate_leaves_criteria[leaf] = self._criterion_wrapper(X, y, leaf)

            # candidate_leaves_criteria = {leaf: self.criterion(X, y, leaf) for leaf in current_leaves if leaf[0] < self.max_depth}
            if len(candidate_leaves_criteria) == 0:
                break

            best_leaf = max(candidate_leaves_criteria, key=candidate_leaves_criteria.get)
            h, i = best_leaf
            region_left = self._get_region_from_node((h+1,2*i-1))
            region_right = self._get_region_from_node((h+1,2*i))
            leaf_vals = (np.mean(y[self._indices_in_region(X, region_left)]), np.mean(y[self._indices_in_region(X, region_right)]))
            _tree.split(best_leaf, leaf_vals)
            current_leaves = [node for node in current_leaves if node != best_leaf]

            current_leaves.append((h+1,2*i-1))
            current_leaves.append((h+1,2*i))
            current_n_leaves += 1

        self._tree = _tree

    def _get_region_from_node(self, node):
        h, i = node
        dim = self.domain.shape[0]

        lc = np.zeros(h) #1 if leftchild, 0 if rightchild
        while(h > 0): #store path of tree
            h -= 1
            lc[h] = i % 2
            i = int(np.ceil(i/2))

        region = np.copy(self.domain)


        for ind in range(lc.shape[0]):
            cdim = ind % dim
            cdimsize = region[cdim,1]-region[cdim,0]

            if(lc[ind]):
                region[cdim,1] = region[cdim,1] - (cdimsize/2)
            else:
                region[cdim,0] = region[cdim,0] + (cdimsize/2)

        return region

    def _criterion_wrapper(self, X, y, node):
        h, i = node
        region_parent = self._get_region_from_node((h,i))
        region_left_child = self._get_region_from_node((h+1,2*i-1))
        region_right_child = self._get_region_from_node((h+1,2*i))

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

    def plot(self,ax, color="skyblue", alpha=0.5):
        if self.domain.shape[0] > 1:
            raise ValueError("Plotting currently only supports one dimensional X space")
        leaves = []
        leaves = [node for node in self._tree.nodes if self._tree.is_leaf(node)]
        regions = [self._get_region_from_node(leaf) for leaf in leaves]
        for i,reg in enumerate(regions):
            val = self._tree[leaves[i]]
            ax.fill_between([reg[0][0], reg[0][1]], [val, val], color=color, alpha=alpha, step='post')

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
    min_var_reduction = float('inf')  # Initialize to a large number

    for i in parent_inds:
        parent_inds_excl = [idx for idx in parent_inds if idx != i]
        l_child_inds_excl = [idx for idx in l_child_inds if idx != i]
        r_child_inds_excl = [idx for idx in r_child_inds if idx != i]

        if len(parent_inds_excl) > 0:  # Ensure there are still data points left
            total_variance = np.var(y[parent_inds_excl])
            if len(l_child_inds_excl) > 0:
                left_variance = np.var(y[l_child_inds_excl])
            else:
                left_variance = 0  # No variance if no data points

            if len(r_child_inds_excl) > 0:
                right_variance = np.var(y[r_child_inds_excl])
            else:
                right_variance = 0  # No variance if no data points

            weighted_variance = ((len(l_child_inds_excl) / len(parent_inds_excl)) * left_variance +
                                 (len(r_child_inds_excl) / len(parent_inds_excl)) * right_variance)

            var_reduction = total_variance - weighted_variance

            # Update the minimum variance reduction found
            if var_reduction < min_var_reduction:
                min_var_reduction = var_reduction

    return min_var_reduction
