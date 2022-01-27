sensitivity\_bounds
-------------------------------

Description
^^^^^^^^^^^^
Bounds on the strength of unobserved confounders using observed covariates, as in Cinelli and Hazlett (2020).
The main generic function is ovb_bounds, which can compute both the bounds on the strength of confounding
as well as the adjusted estimates, standard errors, t-values and confidence intervals.

Other functions that compute only the bounds on the strength of confounding are also provided. These functions
may be useful when computing benchmarks for using only summary statistics from papers you see in print.

Currently it implements only the bounds based on partial R2. Other bounds will be implemented soon.

Reference
^^^^^^^^^^
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias." Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Examples
^^^^^^^^^
Load example dataset

>>> from sensemakr import data
>>> darfur = data.load_darfur()

Fit a statsmodels OLSResults object ("fitted_model")

>>> import statsmodels.formula.api as smf
>>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar +pastvoted + hhsize_darfur + female + village', data=darfur)
>>> fitted_model = model.fit()

Bounds on the strength of confounders 1, 2, or 3 times as strong as female
and 1, 2, or 3 times as strong as pastvoted

>>> from sensemakr import sensitivity_bounds
>>> sensitivity_bounds.ovb_bounds(model = fitted_model, treatment = "directlyharmed", benchmark_covariates = ["female", "pastvoted"], kd = [1, 2, 3]) # doctest: +SKIP

Functions
^^^^^^^^^^
.. autofunction:: sensemakr.ovb_bounds

.. autofunction:: sensemakr.ovb_partial_r2_bound
