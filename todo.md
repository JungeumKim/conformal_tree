Thursday July 18 Meeting:

** TODO Send updated manuscript to Veronika by Monday
** TODO Add outline to introduction, right before the notations section
Outline the sections and highlight contributions
** TODO Cleanup tree methodology section
- Separate the history from the tree. Use H(T) to denote the history, which gets updated in the algorithm as well. H(T) is a map from nodes to split directions
- Use \mathbb{T}_K to denote the class of binary trees with size K
- Define range reduction of a node in a specific direction
- Define candidacy outside of the algorithm box that needs the existence of one direction j such that splitting is possible (splitting is possible requires min samples leaf and requires the threshold to be surpassed)

  Formal definition of eligibility/candidacy, introduce candidate, screening rule with details outside of the algorithm box. Keep the check for K leaves inside the alg box
** TODO Remark 2: (Jungeum, what does it say?)
** TODO Reorganize the sections
- Conformal tree is adaptive can go to section 4
- Section 4: Synthetic data and real data benchmarks
- Section 5: UQ for LLM: GPT classification examples
- Add definition of the naive uncertainty quantification, then just refer to it when explaining results
- Add definition of temperature and other necessary LLM preliminaries
** TODO Figure 1
Add the percentage of points for which the interval was narrower to our method, and the coverage, and the width to all plots
** TODO Make a really nice looking figure that showcases the dyadicness of the tree structure
** TODO Ensure consistent notation in the notation section reaches throughout the whole paper, and that all notation in the appendix proofs matches that notation as well.
Separate notation section in appendix if necessary.
** TODO Remove the word split from the C in eq 3.3, eq. 3.4, eq 3.5?
** TODO Figure 4:
Add a map for NAIVE UQ
** TODO Figure 3:
Add naive UQ to the bar plot, and the empirical test coverage plot
** TODO Figure 6:
Add naive UQ to the bar plot
** TODO Write some concluding remarks preliminary for V to review and revise
- Other robust predictors are possible
- Using other methods for conditional conformal inference within groups may be possible, using our method to determine the groups
** TODO Appendix: Proof of lemma 1
Notation: define P_{X,S}, \mathcal{X}, index set I ranges over what, etc.
Needs to be explicit about everything
** TODO Figure out IID vs exchangeable
What is necessary for the ranks to be uniform, and why do papers say iid?
** TODO Table 1:
- Replace Locally weighted tree with conformal forest (CART)
- Add elaboration in body text explaining why we don't expect to beat other methods/ competitiveness is sufficient
- CART is fine here because the forest does not have a guarantee anyway bc even the Ramdas bound sacrifices to $\alpha/2$



Friday July 12

** TODO Think about the threshold
Does including a threshold to split ruin the $\delta$ factor
** TODO Think about the splitting order
Does including a splitting order also affect the $\delta$ factor?
Revise the algorithm and the notation in the text to including the direction/order (superscript i)
** TODO Do the ChatGPT multiple sampling naive method for UQ
Might need to ask Veronika via email to get the details right for that
** TODO Revise figures
Side-by-side figures, remove large figures, move the lower bound fig that is not close to appendix
** TODO Once everything has been figured out, re-run the experiments
** TODO Revise the combining intervals block
Write about the idea, in detail, involving subsampling rows/cols
** TODO Revise the Gibbs reference section
Remove a lot of the detail
** TODO Revise the tree for full conformal section
Talk about predicting the level with the center of the range, etc
** TODO Talk with people why iid. and would exchangeability make the rank uniform.



* Thursday June 13, 2024
- [ ] Forest version figure out the proof using Ramdas thing to combine the intervals -- each interval needs the coverage and then it should work. Ensure bootstrapping gives the right coverage

* Thursday June 13, 2024



** Experiments
- [x] *** Add exposition for both experiments describing the effectiveness of GPT as a base classifier, eg comparison with standard ML or with naive techniques
- [x] *** Investigate the overcoverage by trying leave-one-out coverage
- [ ] *** Recreate Figure 8 for the politics example where we split in time-periods for the time dependent model
- [x] *** Use the proportion of test points that have sets that are smaller as a metric
- Use this for all models:
- Real/synthetic data comparison
- Politics
- Diagnosis experiment
*** For politics experiment, provide more evidence that GPT knows what these scores are
- GPT defines these scores as, etc etc. Need proof that it knows what they are and that the resulting groups of states that it gives are representative of this knowledge


- [ ] The figure that jungeum wanted
- [x] recreation of the alpha plot for the politics data
- [x] Table with the proportion for the real data oone uhfufufu


** Writing
- [x] *** Algorithm 2 - include one line for stopping criterion and describe what it is
- [x] *** Change notation to use (0,0) for the root node of the tree
*** Work on the introduction
- More scientific
- compress the references
- Look for literature for Gen AI UQ w/ conformal
- Check writing style of Chernozhukov
*** Remove aleatoric/epistemic words or use a citation?
Change them to plain English descriptions of what they mean "variance increases with X" or "Heteroskedastic noise"
*** Add remark 'we do not suggest this, but if people do use it, they should have UQ etc'





** Figures
*** Figure 1 - revised into two figures
**** Fig 1.1 : Sketch of the pipeline (Jungeum)
**** Fig 1.2 : Local conformal plots (3x2 grid, see page 4 of pdf)









# Thursday May 23, 2024

- 1.) [ ] Fix and rewrite the proof that the partition will stay the same: include the binomial coefficient

- 2.) [ ] Better comparison tables with Gibbs/etc
- Write a full experiments section under regression

In main body: compare with standard CP, studentized residuals, and gibbs
In appendix: the same + quantile + raphael + wrapper of our method with their method

- 3.) [ ] Forest/aggregation idea: need to check that it makes sense, write proposition and proof
- We bootstrap to make a different tree, but then we need to use the original (unbootstrapped?) calibration data in order to fit the new tree?



# Thursday May 16, 2024

- [x] Organize the paper into a manuscript with proper sections

- [x] Clearly motivate the setting and compare with studentized + Gibbs to explain why it is needed
- [ ] Include a well-written comparison with Gibbs

- [x] Fix the issue of one additional point changing the candidacy for a node to split
- Idea 1: random splitting candidacy
- Idea 2: fitting the tree separately with each X_{n+1} included, with a made up non-extreme y-value
- Idea 3: bounding the probability that this will happen

- [ ] Write the proof formally, carefully, and clearly

- [ ] Aggregation idea - implementation and theory and simulation
- [ ] Compare with Gibbs in simulations as well

- [ ] Main body flagship experiment: deep learning experiment
- [ ] Main body simulation: comparison of results directly comparable in the same setting
- [ ] Appendix simulation: Huge comparison with all methods

Note: Classification?


# Thursday May 09, 2024

- Use a splitting criterion that is ONLY dependent on the max and min values within a leaf node, with a guaranteed minimum number of leaves in each bin $m$
- Then, with probability $1-2/m$, we have that the new point $X_{n+1},y_{n+1}$ will not be the most extreme point in the bin. Thus, since the partitioning rule does not depend on points that are not the most extreme, it will not affect the partition.
- For the theorem
$\mathbb{E}[P(y_{n+1}\in C)] = P(y_{n+1}\in C \mid y_{n+1} \in A)P(y_{n+1}\in A) + P(y_{n+1}\in C \mid y_{n+1}\not\in A)P(y_{n+1}\not\in A)$
etc


TODO
- [x] Implement the new criterion
- [x] Do numerical experiments, on the benchmark datasets
- [ ] Find a `flagship' new dataset for a cool numerical experiment -- a deep learning example where the training set is unavailable and calibration set is limited, would be ideal!!!
- [x] Write a persuave piece on why splitting on the calibration data is useful :)
- [ ] Think more about the combining intervals and forest idea


# Thursday May 02, 2024

## Idea
- Exchangeability is violated because the tree depends on calibration data but does not depend on $X_{n+1}$
- If the grid/partition were fixed ahead of time, then everything should work (this is the same as Gibbs, Wasserman)
- Using a fixed depth complete dyadic tree is essentially the same as this.
- We could get around this by essentially "marginalizing the tree" by refiting the tree many times for various realizations of $y_{n+1}$ and computing the interval similarly to full conformal prediction. However, this may be not feasible because the tree would need to be fit $N\times |G|$ times where $G$ is a grid. Such a method would likely still yield coverage, with a proof similar to that of the full conformal proof. But it would not be computationally feasible.
- Now, if we can come up with a class of tree and splitting rule such that the partition will never change from a different value of $y_{n+1}$, then we only need to fit a tree once per each test point, making it feasible.

## Todo

- [x] Consider a dyadic tree. We need to investigate the default splitting rule, and come up with a modified splitting rule that has the property that changing $y_{n+1}$ value will not change the shape of the tree. We may need some assumptions, such as a minimum number of samples in each leaf.

- [x] Once we have the splitting rule, we need to implement the dyadic tree with the splitting rule and test the method. We can first use a small calibration set size as it may make it more clear if there is miscoverage.

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
