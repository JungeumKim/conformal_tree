
# Thursday May 02, 2024

## Idea
- Exchangeability is violated because the tree depends on calibration data but does not depend on $X_{n+1}$
- If the grid/partition were fixed ahead of time, then everything should work (this is the same as Gibbs, Wasserman)
- Using a fixed depth complete dyadic tree is essentially the same as this.
- We could get around this by essentially "marginalizing the tree" by refiting the tree many times for various realizations of $y_{n+1}$ and computing the interval similarly to full conformal prediction. However, this may be not feasible because the tree would need to be fit $N\times |G|$ times where $G$ is a grid. Such a method would likely still yield coverage, with a proof similar to that of the full conformal proof. But it would not be computationally feasible.
- Now, if we can come up with a class of tree and splitting rule such that the partition will never change from a different value of $y_{n+1}$, then we only need to fit a tree once per each test point, making it feasible.

## Todo

- [ ] Consider a dyadic tree. We need to investigate the default splitting rule, and come up with a modified splitting rule that has the property that changing $y_{n+1}$ value will not change the shape of the tree. We may need some assumptions, such as a minimum number of samples in each leaf.

- [ ] Once we have the splitting rule, we need to implement the dyadic tree with the splitting rule and test the method. We can first use a small calibration set size as it may make it more clear if there is miscoverage.

- [ ] Proof of this method that sort of "marginalizes" the tree in a full-conformal type of way will actually achieve the coverage

- Future: determine a splitting rule or criterion for general CART



# Thursday April 25, 2024

## Literature review:
 - Write a detailed list of the methodologies in the other relevant papers, and how they differ from our proposed method.
 
 Papers:
 - [ ] Jing Lei - understand the proof and comp.
 - [x] Gibbs and Candes: - understand the proof and comp + why the equivalence happens (using f)
 - [x] Chernozhukov
 - [v] Fong and Holmes
 - [ ] Why the forest average would violate while their integration(?) method would not. What is the intuition
       
## Simulation:
- Implement the classification idea
- Figure out and implement the weighted majority algorithm

## Minor things
- Tables: 2 decimal plcae with (min, max) in smaller font size
- Table 4: what about underfit?
- Fortify the point-guaranttee impossibility by adding Vovk 2012 and Barber 2020 (from Gibbs)

## Final challenge
- Later, we will need to come up with an idea to handle covariate shift.

# Thursday April 18, 2024

- [x] Expand on the distinction between conformal tree and the studentized residuals method with a tree modeling $\hat\sigma$ (write down what I explained to Jungeum in my note)


## Literature to check:
- [ ] Understand the similar approaches and why this has not bee n done before
- Fong and Holmes, Bayesian conformal [[https://proceedings.neurips.cc/paper/2021/hash/97785e0500ad16c18574c64189ccf4b4-Abstract.html]]
    - Check the paper and add to references
- [ ] Candes lecture on time series conformal prediction (check for overlap)
- [ ] Gibbs et al (2023) Grouping or overlapping?
- [ ] Rina paper with distribution shift and discretization

## Idea
- [x] Think of the averaging method
- [x] Theory for averaging method
- [x] Think about the discontinuity


## Simulations
- [x] Implement separate error metrics:
    - [x] Interval score length
    - Interval width
    - Coverage
- [x] More simulation data in multidimensional setting to better understand
    - [x] Experiment with the tree for real data
- [ ] Do quantile conformal methods for the non-filled in areas in the table
- [ ] Apply this method to studentized and cqr variants
- [x] Try the averaging method 
- [x] Add the conditional mean \hat{f} to the sin simulations
