bias\_functions
---------------------------------
Description
^^^^^^^^^^^^^
Compute bias-adjusted estimates, standard-errors, and t-values.

All methods in the script below have similar purposes and parameters, so they are all described here.

These functions compute bias adjusted estimates (adjusted_estimate), standard-errors (adjusted_se),
and t-values (adjusted_t), given a hypothetical strength of the confounder in the partial R2 parameterization.

The functions work either with a statsmodels OLSResults object, or directly passing in numerical inputs, such as the
current coefficient estimate, standard error and degrees of freedom.

They return a numpy array with the adjusted estimate, standard error, or t-value for each partial R2 passed in.

Internally, we also have functions defined to compute the bias and relative_bias, given the same arguments. We also
define internal functions to compute the bias function and relative bias function for the partial R2 parameters.

Finally, in the python version of the package, there is a param_check method which validates all the parameters, since
they are roughly the same for each method.

Parameters
^^^^^^^^^^^
For all methods, r2dz_x and r2yz_dx are required. For all methods other than bf, either model and treatment
or estimate, se, and dof are also required, except adjused_se and bias which do not accept the estimate parameter.

r2dz_x : float or list of floats
    a float or list of floats with the partial R^2 of a putative unobserved confounder "z" with the treatment variable "d", with observed covariates "x" partialed out.
r2yz_dx : float or list of floats
    a float or list of floats with the  partial R^2 of a putative unobserved confounder "z" with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.
model : statsmodels OLSResults object
    a fitted statsmodels OLSResults object for the restricted regression model you have provided.
treatment : string
    a string with the name of the "treatment" variable, e.g. the independent variable of interest.
estimate : float
    a float with the unadjusted estimate of the coefficient for the independent variable of interest.
se : float
    a float with the unadjusted standard error of the regression.
dof : int
    an int with the degrees of freedom of the regression.
reduce : boolean
    whether to reduce (True, default) or increase (False) the estimate due to putative confounding.

Reference
^^^^^^^^^^
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias." Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Example
^^^^^^^^
>>> # Load example dataset and fit a statsmodels OLSResults object ("fitted_model")
>>> from sensemakr import data
>>> darfur = data.load_darfur()
>>> # Fit a statsmodels OLSResults object ("fitted_model")
>>> import statsmodels.formula.api as smf
>>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
>>> fitted_model = model.fit()
>>> # Import this module
>>> from sensemakr import bias_functions
>>> # Computes adjusted estimate for confounder with  r2dz_x = 0.05, r2yz_dx = 0.05
>>> bias_functions.adjusted_estimate(model = fitted_model, treatment = "directlyharmed", r2dz_x = 0.05, r2yz_dx = 0.05)  # doctest: +NUMBER
0.06393
>>> # Computes adjusted SE for confounder with  r2dz_x = 0.05, r2yz_dx = 0.05
>>> bias_functions.adjusted_se(model = fitted_model, treatment = "directlyharmed", r2dz_x = 0.05, r2yz_dx = 0.05)  # doctest: +NUMBER
0.02327
>>> # Computes adjusted t-value for confounder with  r2dz_x = 0.05, r2yz_dx = 0.05
>>> bias_functions.adjusted_t(model = fitted_model, treatment = "directlyharmed", r2dz_x = 0.05, r2yz_dx = 0.05)  # doctest: +NUMBER
2.74724
>>> # Alternatively, pass in numerical values directly.
>>> bias_functions.adjusted_estimate(estimate = 0.09731582, se = 0.02325654, dof = 783, r2dz_x = 0.05, r2yz_dx = 0.05)  # doctest: +NUMBER
0.06393
>>> bias_functions.adjusted_se(se = 0.02325654, dof = 783, r2dz_x = 0.05, r2yz_dx = 0.05) # doctest: +NUMBER
0.02327
>>> bias_functions.adjusted_t(estimate = 0.09731582, se = 0.02325654, dof = 783, r2dz_x = 0.05, r2yz_dx = 0.05)  # doctest: +NUMBER
2.74724

Functions
^^^^^^^^^^

.. autofunction:: sensemakr.adjusted_estimate

.. autofunction:: sensemakr.adjusted_se

.. autofunction:: sensemakr.adjusted_t

.. autofunction:: sensemakr.adjusted_partial_r2

.. autofunction:: sensemakr.bias

.. autofunction:: sensemakr.relative_bias
