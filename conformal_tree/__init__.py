from .forest import ConformalForest, choose_leaf_cap, stability_bound

__all__ = ['ConformalTreeRegression', 'ConformalTreeClassification',
           'ConformalForest', 'choose_leaf_cap', 'stability_bound']


def __getattr__(name):
    # The new forest needs only NumPy; load the older plotting dependencies on demand.
    if name in ('ConformalTreeRegression', 'ConformalTreeClassification'):
        from .conformal_tree import ConformalTreeRegression, ConformalTreeClassification
        return {'ConformalTreeRegression': ConformalTreeRegression,
                'ConformalTreeClassification': ConformalTreeClassification}[name]
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
