**Description**

PySensemakr is the Python version of the R package sensemakr.
The sensemakr package implements a suite of sensitivity analysis tools that makes it easier to
understand the impact of omitted variables in linear regression models, as discussed in Cinelli and Hazlett (2020).

R's linear model ("lm") and data.frame objects most naturally translate to statsmodels OLS/OLSResults and pandas
DataFrame/Series objects, thus PySensemakr makes heavy use of those packages.
In this documention, several examples are included to demonstrate how to use these objects to run this package successfully.


The main function of the package is sensemakr.Sensemakr, which computes the most common sensitivity analysis results.
After creating an object of the Sensemakr class, you may directly use the plot, summary, and print methods of the object.

You may also use the other sensitivity functions of the package directly, such as: (i) functions for sensitivity plots
(`ovb_contour_plot`, `ovb_extreme_plot`); (ii) functions for computing bias-adjusted estimates and t-values
(`adjusted_estimate`, `adjusted_t`); (iii) functions for computing the robustness value and partial R2 directly (`robustness_value`,
`partial_r2`); and, (iv) functions for bounding the strength of unobserved confounders (`ovb_bounds`), among others. These functions are
in the modules `ovb_plots`, `bias_functions`, `sensitivity_stats`, and `ovb_bounds`. `PySensemakr` also comes with example data sets, found
in the module `data`.

More information can be found on this documentation and related papers.

**Reference**

Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
Journal of the Royal Statistical Society, Series B (Statistical Methodology).

**Examples**

>>> # Load example dataset:
>>> from sensemakr import data
>>> darfur = data.load_darfur()
>>> # Fit a statsmodels OLSResults object ("fitted_model")
>>> import statsmodels.formula.api as smf
>>> model = smf.ols(formula='peacefactor ~ \
        directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
>>> fitted_model = model.fit()
>>> # Runs sensemakr for sensitivity analysis
>>> from sensemakr import sensemakr
>>> sensitivity = sensemakr.Sensemakr(
        fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3])
>>> # Description of results
>>> sensitivity.summary()
>>> # Import plot module
>>> from sensemakr import ovb_plots
>>> # Plot bias contour of point estimate
>>> ovb_plots.plot(sensitivity,plot_type = "contour")
>>> # Plot extreme scenario
>>> ovb_plots.plot(sensitivity, plot_type = "extreme")
>>> #Pandas DataFrame with sensitivity statistics
>>> sensitivity.sensitivity_stats
>>> # Pandas DataFrame with bounds on the strength of confounders
>>> sensitivity.bounds
>>> # Using sensitivity functions directly
>>> from sensemakr import sensitivity_stats
>>> from sensemakr import ovb_bounds
>>> from sensemakr import bias_functions
>>> # Robustness value of directly harmed q = 1 (reduce estimate to zero)
>>> sensitivity_stats.robustness_value(model = fitted_model, covariates = 'directlyharmed')
>>> # Robustness value of directly harmed q = 1/2 (reduce estimate in half)
>>> sensitivity_stats.robustness_value(model = fitted_model, covariates = 'directlyharmed', q = 0.5)
>>> # Robustness value of directly harmed q = 1/2, alpha = 0.05
>>> sensitivity_stats.robustness_value(model = fitted_model, covariates = 'directlyharmed', q = 0.5, alpha = 0.05)
>>> # Partial R2 of directly harmed with peacefactor
>>> sensitivity_stats.partial_r2(model = fitted_model, covariates = "directlyharmed")
>>> # Partial R2 of female with peacefactor
>>> sensitivity_stats.partial_r2(model = fitted_model, covariates = "female")
>>> # Pandas DataFrame with sensitivity statistics
>>> sensitivity_stats.sensitivity_stats(model = fitted_model, treatment = "directlyharmed")
>>> # Bounds on the strength of confounders using female and age
>>> ovb_bounds.ovb_bounds(model = fitted_model, treatment = "directlyharmed",
    benchmark_covariates = ["female", "age"], kd = [1, 2, 3])
>>> # Adjusted estimate given hypothetical strength of confounder
>>> bias_functions.adjusted_estimate(model = fitted_model, treatment = "directlyharmed", r2dz_x = 0.1, r2yz_dx = 0.1)
>>> # Adjusted t-value given hypothetical strength of confounder
>>> bias_functions.adjusted_t(model = fitted_model, treatment = "directlyharmed", r2dz_x = 0.1, r2yz_dx = 0.1)
>>> # Bias contour plot directly from model
>>> ovb_plots.ovb_contour_plot(model=fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3])
>>> # Extreme scenario plot directly from model
>>> ovb_plots.ovb_extreme_plot(model=fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3], lim = 0.05)



sensemakr\.sensemakr
---------------------------

.. automodule:: sensemakr.sensemakr
    :members:
    :undoc-members:
    :show-inheritance:

sensemakr\.ovb\_plots
----------------------------

Description
^^^^^^^^^^^^
Sensitivity analysis plots for sensemakr. This module provides the contour and extreme scenario sensitivity
plots of the sensitivity analysis results obtained with the function Sensemakr or model or maunually input statistics.

Reference
^^^^^^^^^^

Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Example
^^^^^^^^
See specific functions below.

.. autofunction:: sensemakr.ovb_plots.ovb_contour_plot

.. autofunction:: sensemakr.ovb_plots.add_bound_to_contour

.. autofunction:: sensemakr.ovb_plots.ovb_extreme_plot

.. autofunction:: sensemakr.ovb_plots.plot

sensemakr\.bias\_functions
---------------------------------

.. automodule:: sensemakr.bias_functions
    :members:
    :undoc-members:
    :show-inheritance:

sensemakr\.sensitivity\_stats
------------------------------------

Description
^^^^^^^^^^^^
Computes the sensitivity statistics: robustness value, partial R2, and Cohen's f2; plus helper functions

Reference
^^^^^^^^^^
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias." Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Example
^^^^^^^^^
See specific functions below.

.. autofunction:: sensemakr.sensitivity_stats.robustness_value

.. autofunction:: sensemakr.sensitivity_stats.sensitivity_stats

.. autofunction:: sensemakr.sensitivity_stats.partial_r2

.. autofunction:: sensemakr.sensitivity_stats.partial_f2

.. autofunction:: sensemakr.sensitivity_stats.group_partial_r2

sensemakr\.ovb\_bounds
-----------------------------

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

Example
^^^^^^^^
See specific functions below.

.. autofunction:: sensemakr.ovb_bounds.ovb_bounds

.. autofunction:: sensemakr.ovb_bounds.ovb_partial_r2_bound

sensemakr\.data
----------------------

.. automodule:: sensemakr.data
    :members:
    :undoc-members:
    :show-inheritance:
