from sklearn import tree
from sklearn import ensemble
import matplotlib.pyplot as plt
import numpy as np

from . import dyadictree
from . import multidyadictree

def forest_membership(data, residuals, max_depth=20,
                    max_leaf_nodes=80,
                    min_samples_leaf=20,
                    n_estimators=100):

    rf_model = ensemble.RandomForestRegressor(
        max_depth=max_depth,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        n_estimators=n_estimators
    )

    rf_model.fit(data, residuals)

    membership = rf_model.apply(data)
    return rf_model, membership


def tree_membership(data,c_gap,  max_depth=20,
                    max_leaf_nodes=80,  min_samples_leaf=20):

    tree_model = tree.DecisionTreeRegressor(
                        max_depth=max_depth,
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf)

    tree_model.fit(data,c_gap)
    membership = tree_model.apply(data)
    return tree_model, membership

def tree_plotter(tree_model,x, c_gap, ax=None,title=""):

    x_grid = np.linspace(0, 1, 1000)
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10,5))

    tree_out =tree_model.predict(x_grid.reshape(-1,1))
    ax.bar(x,c_gap, width=0.005)
    ax.fill_between(x_grid, 0*tree_out ,tree_out , color="blue", alpha=0.3)
    ax.set_title(title)


def dyadic_tree_membership(data,c_gap, domain, max_depth=20,
                    max_leaf_nodes=80,  min_samples_leaf=50):

    # crit_ = dyadictree.criterion_range_reduction
    crit_ = dyadictree.criterion_range_reduction_uw

    tree_model = dyadictree.DyadicTreeRegressor(
                        domain=domain,
                        criterion=crit_,
                        max_depth=max_depth,
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=1)

    tree_model._fit(data,c_gap)
    membership = tree_model.apply(data)
    return tree_model, membership


def multi_dyadic_tree_membership(data,c_gap, domain, max_depth=20,
                                 max_leaf_nodes=80,  min_samples_leaf=50, threshold=0.05):

    # crit_ = dyadictree.criterion_range_reduction
    crit_ = multidyadictree.criterion_range_reduction_uw

    tree_model = multidyadictree.MultiDyadicTreeRegressor(
                        domain=domain,
                        criterion=crit_,
                        max_depth=max_depth,
                        max_leaf_nodes=max_leaf_nodes,
                        min_samples_leaf=min_samples_leaf,
                        min_samples_split=1,
                        threshold=threshold)

    tree_model._fit(data,c_gap)
    membership = tree_model.apply(data)
    return tree_model, membership
