# Conformal Tree

This repository contains an implementation of the Conformal Tree algorithm, as described in the article "Adaptive Uncertainty Quantification for Generative AI" which can be found [here](https://arxiv.org/abs/2408.08990).

## Description

Conformal prediction is a framework for obtaining well-calibrated confidence sets for arbitrary models, provided that joint distribution of all data is exchangeable.

Conformal tree is a locally adaptive method for conformal prediction that does not require access to the training dataset, making it ideal for modern deep learning applications in which the training data is obscured from the user.

## Usage

The basic usage pattern for conformalizing a regression model is

```python
from conformal_tree import ConformalTreeRegression

y_calib_pred = my_model.predict(X_calib)
y_test_pred = my_model.predict(X_test)

ct = ConformalTreeRegression(domain=np.array([[0.0,1.0]]))
conf_scores = np.abs(y_calib, y_calib_pred)
ct.calibrate(X_calib, conf_scores)

test_sets = ct.test_set(X_calib, y_test_pred)
```

Full examples can be found in the demonstration notebooks in this repository, `regression_demo.ipynb` and `classification_demo.ipynb`.

## Attribution

This code was jointly developed by Jungeum Kim and Sean O'Hagan, as part of a project that is a joint work with Veronika Rockova. If you wish to cite this work, please use 

```bibtex
@misc{kim2024adaptiveuncertaintyquantificationgenerative,
      title={Adaptive Uncertainty Quantification for Generative AI}, 
      author={Jungeum Kim and Sean O'Hagan and Veronika Rockova},
      year={2024},
      eprint={2408.08990},
      archivePrefix={arXiv},
      primaryClass={stat.ME},
      url={https://arxiv.org/abs/2408.08990}, 
}
```
