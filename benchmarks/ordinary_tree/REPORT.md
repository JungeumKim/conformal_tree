# Ordinary tree simulations

The ordinary tree is fitted to the calibration sample. Its splits and counts exclude the test observation. After fitting, we route the test covariate to its leaf. Both simulations use the same samples, seeds and fixed predictors as the score-hidden runs.

## Q2. Does adding the test observation change its leaf?

Data 2, n = 499, minimum child size 20, leaf cap 8, zero improvement threshold, 12,000 paired trials. The outcome is disagreement of the geometric regions containing the same test point. For unrestricted cuts, adding a covariate can also change the candidate cut locations.

| Cut and criterion | Score hidden | Ordinary | Changes in ordinary fits |
|---|---:|---:|---:|
| Dyadic midpoint, range | 0.58% | 0.80% | 96 |
| Dyadic midpoint, squared error | 5.67% | 5.91% | 709 |
| All cutpoints, range | 1.78% | 12.01% | 1,441 |
| All cutpoints, squared error | 16.13% | 19.56% | 2,347 |

For dyadic range, the analytical whole-tree bounds are 5.60% (score hidden) and 30.80% (ordinary, including the count term). Each also bounds query-leaf disagreement.

## Q3. Conditional coverage of selected leaves

At 90% nominal coverage, the penalty in percentage points equals 100 times the number of trials in which exactly one fit selects the cell for the query, divided by the number in which the practical fit selects it. The estimated lower bound subtracts this penalty from 90%. Both the bound estimate and observed coverage use simulation frequencies. Separate pilot runs selected the same four cells per distribution. Each validation sample has 400,000 trials.

| Data | Cell | Ordinary selections | Disagreements | Hidden penalty (pp) | Ordinary penalty (pp) | Ordinary lower bound | Observed coverage (95% interval) |
|---|---|---:|---:|---:|---:|---:|---:|
| 1 | [0, 5] | 69,005 | 156 | 0.23 | 0.23 | 89.77% | 90.02% (89.80–90.25%) |
| 1 | (2.5, 3.75] | 7,127 | 87 | 1.22 | 1.22 | 88.78% | 89.28% (88.54–89.99%) |
| 1 | (5, 10] | 36,542 | 285 | 0.78 | 0.78 | 89.22% | 90.12% (89.81–90.42%) |
| 1 | (5, 7.5] | 77,461 | 193 | 0.25 | 0.25 | 89.75% | 90.24% (90.03–90.45%) |
| 2 | [0, 0.625] | 23,976 | 350 | 0.06 | 1.46 | 88.54% | 91.25% (90.88–91.60%) |
| 2 | (5, 10] | 93,210 | 515 | 0.55 | 0.55 | 89.45% | 90.21% (90.02–90.40%) |
| 2 | (5, 7.5] | 29,513 | 341 | 1.16 | 1.16 | 88.84% | 90.36% (90.01–90.69%) |
| 2 | (5, 6.25] | 8,696 | 132 | 1.37 | 1.52 | 88.48% | 90.72% (90.09–91.32%) |

The C++ implementations passed 40 paired comparisons for all four splitting rules and 256 comparisons of the selected leaves, calibration counts and coverage indicators. In all 800,000 validation trials, the comparison tree fitted with the test observation included returned the same leaf as in the earlier simulation.

See the repository README for the reproduction commands.
