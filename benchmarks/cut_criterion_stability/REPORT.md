# How often does revealing the test score change its leaf?

Each entry is a percentage of 12,000 paired calibration/test samples. We record whether the **geometric interval of covariate values containing the test point** differs between the fit that hides its score and the fit that includes it. Both fits use the same covariates, including the test point, for candidate cut locations, routing and counts. The outcome is query-leaf disagreement under hallucination, \(\Pr(\widehat A_n^{\mathrm H}(X_{n+1})\ne\widehat A_{n+1}(X_{n+1}))\). No fixed cell is specified.

| DGP | Calibration \(n\) | Dyadic range | Dyadic squared error | All cuts, range | All cuts, squared error | Dyadic range upper bound |
|---|---:|---:|---:|---:|---:|---:|
| Data 1 | 249 | 0.575% | 2.867% | 0.317% | 10.458% | 4.80% |
| Data 1 | 499 | 0.350% | 1.700% | 0.075% | 6.867% | 2.40% |
| Data 1 | 999 | 0.108% | 1.033% | 0.025% | 4.900% | 1.20% |
| Data 2 | 249 | 0.900% | 1.633% | 2.458% | 0.725% | 11.20% |
| Data 2 | 499 | 0.592% | 1.158% | 1.742% | 0.783% | 5.60% |
| Data 2 | 999 | 0.258% | 0.658% | 1.125% | 0.867% | 2.80% |

For dyadic range we use the whole-tree upper bound, which also bounds the chance that the query leaf changes. In one dimension it is \(\min\{1,(4K_{\max}-4)/(n+1)\}\). The table shows the bound only for the method covered by this argument. Empirical probabilities at three sample sizes do not establish asymptotic rates.

At \(n=499\), approximate 95% Monte Carlo half-widths, in percentage points and in the table's column order, are (0.106, 0.231, 0.049, 0.453) for Data 1 and (0.137, 0.191, 0.234, 0.158) for Data 2. Data 2's all-cuts squared-error rate exceeds dyadic range by 0.192 points, with a paired 95% half-width of 0.208 points. Standard errors use the variance of the paired outcomes within each of eight fixed predictors. They describe simulation uncertainty for that equally weighted panel.

The average numbers of leaves in the score-hidden fits at \(n=499\) were:

| DGP | Dyadic range | Dyadic squared error | All cuts, range | All cuts, squared error |
|---|---:|---:|---:|---:|
| Data 1 | 4.00 | 3.91 | 4.00 | 3.99 |
| Data 2 | 7.69 | 2.00 | 7.60 | 2.16 |

## Same samples with the minimum improvement set to zero

As a sensitivity check, we repeated all four fits at \(n=499\) with the same samples and a zero minimum improvement threshold. Every hidden-score tree reached its cap (4 or 8 leaves). The table again shows the probability that the test point changes leaves.

| DGP | Dyadic range | Dyadic squared error | All cuts, range | All cuts, squared error |
|---|---:|---:|---:|---:|
| Data 1 | 0.325% | 1.025% | 0.075% | 6.533% |
| Data 2 | 0.583% | 5.667% | 1.783% | 16.133% |

For Data 1, allowing all cuts increased disagreement under squared error but decreased it under range reduction at both thresholds. On Data 2, early stopping strongly affected the squared-error results. With all eight leaves, switching from dyadic range to dyadic squared error increased disagreement from 0.58% to 5.67%, and allowing all cuts under squared error raised it to 16.13%. These comparisons concern partition stability. Measuring prediction coverage or width would require additional outcomes.

## Design and verification

- Both DGPs come from the project's benchmark loader. Both draw \(X\sim U(0,10)\) and \(Z\sim N(0,1)\) independently. For Data 1, \(Y=3\sin(4/X+0.2)+1.5+XZ\). For Data 2, \(Y=\sin(X^{-3})+0.1Z\).
- For each DGP, eight random forest predictors were independently trained on 300 observations using the benchmark settings (100 trees, minimum leaf size 2). Each training sample's mean absolute outcome sets the response scale. Each predictor is held fixed for 1,500 fresh calibration/test samples per \(n\). All four methods receive exactly the same data in each trial. The tree uses absolute residuals from the fitted predictor with independent uniform jitter of at most \(10^{-12}\). We use the known domain \([0,10]\) for dyadic cuts.
- All four methods use best-first growth, minimum child count 20, maximum depth 8, and the paper's benchmark caps (4 leaves on Data 1 and 8 on Data 2). A dyadic node offers its midpoint. An all-cuts node offers the midpoints between consecutive distinct covariate values satisfying the count rule. Ties are resolved left to right by node and then cut. The predictor is shared across methods and is distinct from the tree fitted to the residuals.
- Range gain is parent range minus the unweighted mean child range. CART gain is parent SSE minus the sum of the two child SSEs, using visible scores about their respective means. The gate requires parent loss to exceed 1.05 times child loss, equivalently a fractional gain above \(0.05/1.05\) for positive parent loss. This matches the release's range rule and puts the squared-error stopping rule on the same relative scale. The all-cuts squared-error cell uses CART splits with these matched growth settings.
- The C++ scan uses cumulative ranges and sums for efficiency. We compared both fits for all four methods with direct exhaustive recomputation on 40 independent cases. We also compared the dyadic range query boundaries and leaf counts against the project's corrected Python implementation. The one-dimensional forest prediction lookup was checked against direct sklearn predictions at split boundaries and on fresh feature values.
- To reproduce with NumPy, scikit-learn, a C++17 compiler, and the project's corrected Python tree available, run `python run.py`. `summary.json` includes paired standard errors and partition disagreement. `trials.npz` records the outcomes for each method on each sample, in the order listed in that file.
