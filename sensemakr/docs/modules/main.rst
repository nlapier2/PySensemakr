Sensemakr
---------------------------
Description
^^^^^^^^^^^^^^
Sensitivity analysis to unobserved confounders.

This class performs sensitivity analysis to omitted variables as discussed in Cinelli and Hazlett (2020).
It returns an object of class Sensemakr with several pre-computed sensitivity statistics for reporting.
After creating the object, you may directly use the plot and summary methods of the returned object.

Sensemakr is a convenience class. You may use the other sensitivity functions of the package directly,
such as the functions for sensitivity plots (ovb_contour_plot, ovb_extreme_plot), the functions for
computing bias-adjusted estimates and t-values (adjusted_estimate, adjusted_t), the functions for computing the
robustness value and partial R2 (robustness_value, partial_r2),  or the functions for bounding the strength
of unobserved confounders (ovb_bounds), among others.

Parameters
^^^^^^^^^^
First argument should either be

* a statsmodels OLSResults object ("fitted_model"); or
* the numerical estimated value of the coefficient, along with the numeric values of standard errors and
  degrees of freedom (arguments estimate, se and df).

Return
^^^^^^^
An object of class Sensemakr, containing:

* sensitivity_stats : A Pandas DataFrame with the sensitivity statistics for the treatment variable,
  as computed by the function sensitivity_stats.

* bounds : A pandas DataFrame with bounds on the strength of confounding according to some benchmark covariates,
  as computed by the function ovb_bounds.

Reference
^^^^^^^^^^
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Examples
^^^^^^^^
>>> # Load example dataset:
>>> import sensemakr as smkr
>>> darfur = smkr.load_darfur()
>>> # Fit a statsmodels OLSResults object ("fitted_model")
>>> import statsmodels.formula.api as smf
>>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
>>> fitted_model = model.fit()
>>> # Runs sensemakr for sensitivity analysis
>>> sensitivity = smkr.Sensemakr(fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3])
>>> # Manual bounds example:
>>> sensitivity_manual = smkr.Sensemakr(model = fitted_model, treatment = "directlyharmed", r2dz_x = [0.1, 0.2], r2yz_dx= [0.2, 0.3], bound_label= ["bound 1", "bound 2"])
>>> # List of Lists example:
>>> sensitivity_lists = smkr.Sensemakr(model = fitted_model, treatment = "directlyharmed", benchmark_covariates = [['female'], ['pastvoted','female']])
>>> # Dictionary Example:
>>> sensitivity_dict = smkr.Sensemakr(model = fitted_model, treatment = "directlyharmed", benchmark_covariates = {'Fem':['female'],"FemPast":["female","pastvoted"]})
>>> # Description of results
>>> sensitivity.summary() # doctest: +SKIP

.. autoclass:: sensemakr.Sensemakr
    :members: summary, plot, print, ovb_minimal_reporting
    :special-members:
