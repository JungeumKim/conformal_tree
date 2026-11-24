# Conformal Tree

Code for [Adaptive Uncertainty Quantification for Generative AI](https://arxiv.org/abs/2408.08990).

## Install

Use Python 3.12. From the repository root:

```sh
python -m pip install -e ".[benchmarks]"
```

## Usage

Use a model trained independently of the calibration and test data.

```python
import numpy as np
from conformal_tree import ConformalForest

scores = np.abs(y_cal - model.predict(X_cal))
domain = np.column_stack((X_train.min(axis=0), X_train.max(axis=0)))

forest = ConformalForest(
    min_samples_leaf=20, max_leaves=4, max_directions="sqrt",
    n_trees=100, max_depth=8, threshold=0.05, seed=123,
)
forest.fit(X_cal, scores, domain)
lower, upper = forest.predict_intervals(
    X_test, model.predict(X_test), alpha=0.1,
    aggregation="uniform", rng=np.random.default_rng(456),
)
```

This uses the hallucination procedure: each test X is included in counts, but
its score is not used. `"sqrt"` samples `ceil(sqrt(d))` directions per node.
Other choices are `"log2"`, `"all"`, or an integer. For a single multidyadic
tree, set `max_directions="all", n_trees=1`.

Use `aggregation="majority"` for majority voting. `forest.delta_` gives the
stability bound. No stability correction is applied to the quantiles automatically.
The older `ConformalTreeRegression` and `ConformalTreeClassification` classes remain available.

## Tree and forest comparison

From the repository root:

```sh
git clone https://github.com/msesia/cqr-comparison.git ../cqr-comparison
python -m benchmarks.compare_tree_forest --data-dir ../cqr-comparison/datasets --jobs 4
```

The default run compares a single tree with all directions against a forest
of 100 trees with `ceil(sqrt(d))` directions, on eight datasets over 20 paired
trials. It uses saved model scores and splits, then refits the trees and
recomputes the intervals. Settings are `alpha=0.1`, `m=20`, depth 8, and
improvement threshold 0.05. Both methods use the same leaf cap, chosen as the
largest cap up to the original benchmark cap for which the forest has `delta<=0.1`.
Responses are divided by the mean absolute training response.

Results are saved to `benchmarks/output/tree_forest/tables.txt`.
The script resumes completed trials and checks the results against the saved reference.
For alternatives, use `--directions log2`, `--aggregation majority`, or
`--refit-model` to repeat model fitting. Choose a new `--output` directory when changing settings.

## Citation

```bibtex
@misc{kim2024adaptiveuncertaintyquantificationgenerative,
  title={Adaptive Uncertainty Quantification for Generative AI},
  author={Jungeum Kim and Sean O'Hagan and Veronika Rockova},
  year={2024}, eprint={2408.08990}, archivePrefix={arXiv},
  primaryClass={stat.ME}
}
```

## Tree stability and selected-leaf coverage

- [Split criterion comparison](benchmarks/cut_criterion_stability/REPORT.md) reports how often revealing the test score changes its leaf. The eight-leaf Data 2 comparison is in the zero-threshold table.
- [Selected-leaf coverage](benchmarks/selected_leaf_coverage/REPORT.md) reports the estimated coverage bound and observed coverage for eight leaves on Data 1 and Data 2.

To reproduce both simulations on macOS or Linux, use Python 3.10 or later with NumPy, SciPy, scikit-learn, and a C++17 compiler available as `c++`. Run from the repository root:

```sh
python benchmarks/cut_criterion_stability/run.py
python benchmarks/selected_leaf_coverage/run_leaves.py
```

Both scripts generate their own data. New runs write to `benchmarks/output/`; the saved results are beside each report.

### Ordinary tree and real-data leaf coverage

The [ordinary-tree tables](benchmarks/ordinary_tree/REPORT.md) use only the calibration observations to fit the tree. To rerun the Data 2 split comparison and the two synthetic selected-leaf tables:

```sh
python benchmarks/ordinary_tree/run_split_comparison.py
python benchmarks/selected_leaf_coverage/run_leaves.py --ordinary
python benchmarks/ordinary_tree/tables.py --rerun
```

The [real-data tables](benchmarks/real_selected_leaf_coverage/REPORT.md) compare the hallucination and ordinary trees on six regression benchmarks and two classification applications. To regenerate the tables and leaf definitions from the saved counts:

```sh
python benchmarks/real_selected_leaf_coverage/tables.py
```

To repeat the underlying simulations, use the same cqr-comparison datasets as above. The classification scores are available in this checkout. For legislator, the two covariates are stored as dyadic bin indices at depth six, the deepest possible split with the seven-leaf cap.

```sh
python benchmarks/real_selected_leaf_coverage/inputs.py --data-dir ../cqr-comparison/datasets
python benchmarks/real_selected_leaf_coverage/run.py
python benchmarks/real_selected_leaf_coverage/skin_independent.py
python benchmarks/real_selected_leaf_coverage/tables.py --rerun
```

The regenerated real-data report, counts and LaTeX tables are under `benchmarks/output/real_selected_leaf_coverage/`.
