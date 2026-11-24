# Conditional coverage in selected leaves of the hallucination tree

**Data 1 and Data 2, nominal coverage 90%.** Each dataset has its own independently trained, fixed black-box predictor. Estimates average over 400,000 new calibration samples and test observations per dataset. The two fits in each trial use identical covariates and differ only in whether the test score is visible during fitting.

| DGP | Fixed cell A | Query returned A | Estimated penalty (pp) | Estimated lower bound | 95% simultaneous lower bound | Observed conditional coverage (pointwise 95% interval) |
|---|---|---:|---:|---:|---:|---:|
| Data 1 | [0, 5] | 17.25% | 0.23 | 89.77% | 89.72% | 90.02% (89.80%–90.25%) |
| Data 1 | (2.5, 3.75] | 1.78% | 1.22 | 88.78% | 88.32% | 89.28% (88.54%–89.99%) |
| Data 1 | (5, 10] | 9.14% | 0.78 | 89.22% | 89.07% | 90.12% (89.81%–90.42%) |
| Data 1 | (5, 7.5] | 19.37% | 0.25 | 89.75% | 89.69% | 90.24% (90.03%–90.45%) |
| Data 2 | [0, 0.625] | 5.98% | 0.06 | 89.94% | 89.88% | 91.26% (90.89%–91.61%) |
| Data 2 | (5, 10] | 23.30% | 0.55 | 89.45% | 89.37% | 90.21% (90.02%–90.40%) |
| Data 2 | (5, 7.5] | 7.38% | 1.16 | 88.84% | 88.64% | 90.36% (90.01%–90.69%) |
| Data 2 | (5, 6.25] | 2.17% | 1.37 | 88.63% | 88.20% | 90.72% (90.09%–91.32%) |

## Counts behind the estimates

H means the score-hidden tree returns A for the query. F means the tree fitted using all scores returns A. The numerator of the penalty counts both ways of disagreeing. The denominator counts H, including agreement. Coverage counts successes only among H.

| DGP | Cell | H | H only | F only | Estimated δ (%) | Covered among H |
|---|---|---:|---:|---:|---:|---:|
| Data 1 | [0, 5] | 69005 | 101 | 55 | 0.0390% | 62120 |
| Data 1 | (2.5, 3.75] | 7127 | 77 | 10 | 0.0217% | 6363 |
| Data 1 | (5, 10] | 36542 | 154 | 131 | 0.0713% | 32931 |
| Data 1 | (5, 7.5] | 77461 | 134 | 59 | 0.0483% | 69901 |
| Data 2 | [0, 0.625] | 23910 | 5 | 9 | 0.0035% | 21820 |
| Data 2 | (5, 10] | 93210 | 239 | 276 | 0.1288% | 84084 |
| Data 2 | (5, 7.5] | 29513 | 165 | 176 | 0.0852% | 26667 |
| Data 2 | (5, 6.25] | 8683 | 61 | 58 | 0.0298% | 7877 |

## Interpretation

Data 1's cell (2.5, 3.75] illustrates the distinction between the empirical penalty and the worst-case stability formula. There were 87 disagreements and 7,127 practical selections, so the estimated penalty is 1.22 percentage points. The depth-three analytical numerator bound is 10/500 = 2%. Dividing that worst-case numerator by the estimated selection probability and truncating gives a lower coverage bound of zero. The stronger numbers in the table use measured disagreement on these simulated DGPs.

## Simulation and uncertainty

- The estimand conditions on the event that the fitted score-hidden tree returns exactly A as the query leaf. A parent and its child can therefore both appear as examples across repeated fits. The interval boundary convention is lower-open and upper-closed, with zero included in the first region.
- Benchmark DGPs: X is uniform on [0,10], Z is independent standard normal. Data 1 uses Y = 3 sin(4/X + 0.2) + 1.5 + X Z. Data 2 uses Y = sin(X^(-3)) + 0.1 Z. For each DGP, predictor 0 from the earlier stability experiment is held fixed (100-tree random forest trained on 300 independent cases). The predictor and its scale are independent of the pilot and validation samples. Results describe these two fixed predictors.
- Tree settings are n=499, m=20, depth at most 8, and Kmax=4 (Data 1) or 8 (Data 2). The benchmark range gate requires parent range greater than 1.05 times mean child range. These are the benchmark settings, including the usual stopping rule. An independent uniform jitter of width 1e-12 is added to all residuals for tree fitting only. The cutoff uses the original calibration scores in the score-hidden leaf. With L such scores, its rank is ceil(0.90(L+1)), with an infinite cutoff for ranks exceeding L.
- Leaf selection: 12,000 independent pilot trials per DGP. Among nonroot cells returned on at least 0.5% of trials, choose distinct cells closest on the log scale to 20%, 8%, and 2% query-return probability. Add the most frequent remaining cell touching zero to cover the oscillatory region of Data 2. Only selection frequencies were used. Evaluate the locked choices on independent validation samples; outcomes from all validation trials are retained.
- Write D for disagreement count (H only plus F only), H for the practical selection count, and W for covered practical selections. Estimates are δ=D/T, ρ=H/T, penalty=D/H, lower bound=max(0,0.90-D/H), and observed conditional coverage=W/H. The plug-in lower bound has Monte Carlo error.
- The simultaneous lower column accounts for Monte Carlo estimation of the theorem's numerator and denominator. For each of the eight selected cells, compute an exact one-sided binomial upper bound on δ and lower bound on ρ, each with tail probability 0.05/16. Substitute their ratio into the theorem and truncate at zero. Bonferroni gives joint confidence of at least 95% for these eight lower bounds, conditional on the pilot choices and fixed predictors. The observed coverage intervals are exact binomial intervals conditional on H, pointwise for each leaf.
- Validation: the pilot/validation seeds are separate. Across 256 further paired samples the C++ fits matched the released Python implementation for the geometric leaves, calibration sample sizes, and coverage indicators. Across each validation dataset, summing the fixed-cell disagreement counts over all cells gives twice the number of query leaf changes.

## Reproduction

Run `python run_leaves.py` in this folder with NumPy, SciPy, scikit-learn, a C++17 compiler, the sibling `cut_criterion_stability` folder, and the package in this checkout. `selection.json` lists pilot choices. `results.json` contains per-cell validation counts. `data1.npz` and `data2.npz` preserve both leaf IDs, the observed coverage indicator, and the leaf calibration count per trial.
