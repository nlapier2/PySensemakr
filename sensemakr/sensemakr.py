"""
Sensemakr: extending omitted variable bias

####
NOTE: This is a Python version of the original R sensemakr package, which can be found here:
https://github.com/carloscinelli/sensemakr
R's linear model ("lm") and data.frame objects most naturally translate into statsmodels OLS/OLSResults and pandas
    DataFrame/Series objects, so we have used those in the python version. These packages are not universally familiar
    to python users, but they were the most straightforward and faithful way to translate the R package into Python.
    Examples are included below to demonstrate how to use these objects to run this package successfully.
There are some features of the original package that have not yet been implemented. For instance, R's notion of
    "formulas" for linear models are less used by and less familiar to python programmers, and thus have not yet been
    included in the Python version. This could change if there is sufficient demand.
####

The sensemakr package implements a suite of sensitivity analysis tools that makes it easier to
understand the impact of omitted variables in linear regression models, as discussed in Cinelli and Hazlett (2020).

This package defines a class called Sensemakr, which computes the most common sensitivity analysis results.
After creating an object of the Sensemakr class, you may directly use the plot and print methods of the object.

You may also use the other sensitivity functions of the package directly, such as the functions for sensitivity plots
(ovb_contour_plot, ovb_extreme_plot) the functions for computing bias-adjusted estimates and t-values
(adjusted_estimate, adjusted_t), the functions for computing the robustness value and partial R2 (robustness_value,
partial_r2),  or the functions for bounding the strength of unobserved confounders (ovb_bounds), among others.

More information can be found on the help documentation and related papers.

Reference:
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
    Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Examples:

# load example dataset and fit a statsmodels OLSResults object ("fitted_model")
import pandas as pd
darfur = pd.read_csv('data/darfur.csv')

# fit a statsmodels OLSResults object ("fitted_model")
import statsmodels.formula.api as smf
model = smf.ols(formula='peacefactor ~ \
    directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
fitted_model = model.fit()

# runs sensemakr for sensitivity analysis
from sensemakr import sensemakr
sensitivity = sensemakr.Sensemakr(
    fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3])

# description of results
sensitivity.summary()

#### plot bias contour of point estimate
###plot(sensitivity)

#### plot bias contour of t-value
###plot(sensitivity, sensitivity_of = "t-value")

#### plot extreme scenario
###plot(sensitivity, type = "extreme")

# Pandas DataFrame with sensitivity statistics
sensitivity.sensitivity_stats

# Pandas DataFrame with bounds on the strength of confounders
sensitivity.bounds

### Using sensitivity functions directly ###
from sensemakr import sensitivity_stats
from sensemakr import ovb_bounds
from sensemakr import bias_functions

# robustness value of directly harmed q = 1 (reduce estimate to zero)
sensitivity_stats.robustness_value(model = fitted_model, covariates = 'directlyharmed')

# robustness value of directly harmed q = 1/2 (reduce estimate in half)
sensitivity_stats.robustness_value(model = fitted_model, covariates = 'directlyharmed', q = 0.5)

# robustness value of directly harmed q = 1/2, alpha = 0.05
sensitivity_stats.robustness_value(model = fitted_model, covariates = 'directlyharmed', q = 0.5, alpha = 0.05)

# partial R2 of directly harmed with peacefactor
sensitivity_stats.partial_r2(model = fitted_model, covariates = "directlyharmed")

# partial R2 of female with peacefactor
sensitivity_stats.partial_r2(model = fitted_model, covariates = "female")

# pandas DataFrame with sensitivity statistics
sensitivity_stats.sensitivity_stats(model = fitted_model, treatment = "directlyharmed")

# bounds on the strength of confounders using female and age
ovb_bounds.ovb_bounds(model = fitted_model, treatment = "directlyharmed",
    benchmark_covariates = ["female", "age"], kd = [1, 2, 3])

# adjusted estimate given hypothetical strength of confounder
bias_functions.adjusted_estimate(model = fitted_model, treatment = "directlyharmed", r2dz_x = 0.1, r2yz_dx = 0.1)

# adjusted t-value given hypothetical strength of confounder
bias_functions.adjusted_t(model = fitted_model, treatment = "directlyharmed", r2dz_x = 0.1, r2yz_dx = 0.1)

#### bias contour plot directly from model
###ovb_contour_plot(fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3])

#### extreme scenario plot directly from model
###ovb_extreme_plot(model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3], lim = 0.05)

"""

import sys
import pandas as pd
from scipy.stats import t
import numpy as np
from . import sensitivity_stats
from . import bias_functions
from . import ovb_bounds


class Sensemakr:
    """
    Sensitivity analysis to unobserved confounders

    This function performs sensitivity analysis to omitted variables as discussed in Cinelli and Hazlett (2020).
    It returns an object of class Sensemakr with several pre-computed sensitivity statistics for reporting.
    After creating the object, you may directly use the plot and summary methods of the returned object.

    Sensemakr is a convenience class. You may use the other sensitivity functions of the package directly, 
    such as the functions for sensitivity plots (ovb_contour_plot, ovb_extreme_plot), the functions for
    computing bias-adjusted estimates and t-values (adjusted_estimate, adjusted_t), the functions for computing the
    robustness value and partial R2 (robustness_value, partial_r2),  or the functions for bounding the strength
    of unobserved confounders (ovb_bounds), among others.

    @param ... arguments passed to other methods. First argument should either be
    (i)  an lm model with the outcome regression (argument model); or
    (ii) the numerical estimated value of the coefficient, along with the numeric values of standard errors and
         degrees of freedom (arguments estimate, se and df).


    Reference:
    Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
        Journal of the Royal Statistical Society, Series B (Statistical Methodology).

    Examples:
    # loads dataset
    data("darfur")

    # runs regression model
    model <- lm(peacefactor ~ directlyharmed + age + farmer_dar + herder_dar +
                             pastvoted + hhsize_darfur + female + village, data = darfur)

    # runs sensemakr for sensitivity analysis
    sensitivity <- sensemakr(model, treatment = "directlyharmed",
                                   benchmark_covariates = "female",
                                   kd = [1, 2, 3])
    # short description of results
    sensitivity

    # long description of results
    summary(sensitivity)

    # plot bias contour of point estimate
    plot(sensitivity)

    # plot bias contour of t-value
    plot(sensitivity, sensitivity_of = "t-value")

    # plot extreme scenario
    plot(sensitivity, type = "extreme")

    :return:
    An object of class Sensemakr, containing:
        sensitivity_stats: A Pandas DataFrame with the sensitivity statistics for the treatment variable,
                           as computed by the function sensitivity_stats.
        bounds: A pandas DataFrame with bounds on the strength of confounding according to some benchmark covariates,
                as computed by the function ovb_bounds.
    """

    def __init__(self, model=None, treatment=None, estimate=None, se=None, dof=None, benchmark_covariates=None, kd=1,
                 ky=None, q=1, alpha=0.05, r2dz_x=None, r2yz_dx=None, r2dxj_x=None, r2yxj_dx=None,
                 bound_label="Manual Bound", reduce=True):
        """
        The constructor for a Sensemakr object. Parameter descriptions are below. For usage and info, see the
        description of the class above.

        Required parameters: model and treatment, or estimate, se, and dof
        :param model: a fitted statsmodels OLSResults object for the restricted regression model you have provided
        :param treatment: a string with the name of the "treatment" variable, e.g. the independent variable of interest
        :param estimate: a float with the estimate of the coefficient for the independent variable of interest
        :param se: a float with the standard error of the regression
        :param dof: an int with the degrees of freedom of the regression
        :param benchmark_covariates: a string or list of strings with
            the names of the variables to use for benchmark bounding
        :param kd: a float or list of floats with each being a multiple of the strength of association between a
            benchmark variable and the treatment variable to test with benchmark bounding
        :param ky: same as kd except measured in terms of strength of association with the outcome variable
        :param q: a float with the percent to reduce the point estimate by for the robustness value RV_q
        :param alpha: a float with the significance level for the robustness value RV_qa to render the
            estimate not significant
        :param r2dz_x: a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
            with the treatment variable "d", with observed covariates "x" partialed out. In this case, you are manually
            specifying a putative confounder's strength rather than benchmarking.
        :param r2yz_dx: a float or list of floats with the  partial R^2 of a putative unobserved confounder "z"
            with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.
            In this case, you are manually specifying a putative confounder's strength rather than benchmarking.
        :param r2dxj_x: float with the partial R2 of covariate Xj with the treatment D
            (after partialling out the effect of the remaining covariates X, excluding Xj).
        :param r2yxj_dx: float with the partial R2 of covariate Xj with the outcome Y
            (after partialling out the effect of the remaining covariates X, excluding Xj).
        :param bound_label: a string what to call the name of a bounding variable, for printing and plotting purposes
        :param reduce: whether to reduce (True, default) or increase (False) the estimate due to putative confounding
        """
        if (model is None or treatment is None) and (estimate is None or se is None or dof is None):
            sys.exit('Error: Sensemakr object requires either a statsmodels OLSResults object and treatment name or an '
                     'estimate, standard error, and degrees of freedom.')
        # Set this object's attributes based on user input
        self.benchmark_covariates = benchmark_covariates
        if type(kd) is list:
            self.kd = np.array(kd)
        else:
            self.kd = kd
        if ky is None:
            self.ky = self.kd
        elif type(ky) is list:
            self.ky = np.array(ky)
        else:
            self.ky = ky
        self.q = q
        self.alpha = alpha
        self.r2dz_x = r2dz_x
        if r2yz_dx is None:
            self.r2yz_dx = self.r2dz_x
        else:
            self.r2yz_dx = r2yz_dx
        self.r2dxj_x = r2dxj_x
        if r2yxj_dx is None:
            self.r2yxj_dx = self.r2dxj_x
        else:
            self.r2yxj_dx = r2yxj_dx
        self.bound_label = bound_label
        self.reduce = reduce

        # Compute sensitivity statistics for this model
        if model is not None:
            self.model = model
            self.treatment = treatment
            self.sensitivity_stats = sensitivity_stats.sensitivity_stats(model=self.model, treatment=self.treatment,
                                                                         q=self.q, alpha=self.alpha, reduce=self.reduce)
            self.estimate = self.sensitivity_stats['estimate']
            self.se = self.sensitivity_stats['se']
            self.dof = self.model.df_resid
        else:
            self.sensitivity_stats = sensitivity_stats.sensitivity_stats(estimate=self.estimate, se=se, dof=dof,
                                                                         q=self.q, alpha=self.alpha, reduce=self.reduce)
            self.estimate = estimate
            self.se = se
            self.dof = dof
            self.treatment = 'D'
        if reduce:
            self.h0 = self.estimate * (1 - self.q)
        else:
            self.h0 = self.estimate * (1 + self.q)

        # Compute omitted variable bias (ovb) bounds
        if self.r2dz_x is None:
            self.bounds = None
        else:
            if model is not None:
                self.r2dz_x, self.r2yz_dx = sensitivity_stats.check_r2(self.r2dz_x, self.r2yz_dx)
                # Compute adjusted parameter estimate, standard error, and t statistic
                self.adjusted_estimate = bias_functions.adjusted_estimate(model=self.model, treatment=self.treatment,
                                                                          r2dz_x=self.r2dz_x, r2yz_dx=self.r2yz_dx,
                                                                          reduce=self.reduce)
                self.adjusted_se = bias_functions.adjusted_se(model=self.model, treatment=self.treatment,
                                                              r2dz_x=self.r2dz_x, r2yz_dx=self.r2yz_dx)
                self.adjusted_t = bias_functions.adjusted_t(model=self.model, treatment=self.treatment,
                                                            r2dz_x=self.r2dz_x, r2yz_dx=self.r2yz_dx,
                                                            h0=self.h0, reduce=self.reduce)
            else:
                self.r2dz_x, self.r2yz_dx = sensitivity_stats.check_r2(self.r2dz_x, self.r2yz_dx)
                # Compute adjusted parameter estimate, standard error, and t statistic
                self.adjusted_estimate = bias_functions.adjusted_estimate(estimate=self.estimate, se=self.se,
                                                                          dof=self.dof, treatment=self.treatment,
                                                                          r2dz_x=self.r2dz_x, r2yz_dx=self.r2yz_dx,
                                                                          reduce=self.reduce)
                self.adjusted_se = bias_functions.adjusted_se(se=self.se, dof=self.dof, treatment=self.treatment,
                                                              r2dz_x=self.r2dz_x, r2yz_dx=self.r2yz_dx)
                self.adjusted_t = bias_functions.adjusted_t(estimate=self.estimate, se=self.se, dof=self.dof,
                                                            treatment=self.treatment, r2dz_x=self.r2dz_x,
                                                            r2yz_dx=self.r2yz_dx, h0=self.h0, reduce=self.reduce)

            # Compute confidence interval
            se_multiple = t.ppf(alpha / 2, self.dof)  # number of SEs within CI based on alpha
            self.adjusted_lower_CI = self.adjusted_estimate - se_multiple * self.adjusted_se
            self.adjusted_upper_CI = self.adjusted_estimate + se_multiple * self.adjusted_se

            # Place results in a DataFrame
            self.bounds = pd.DataFrame(data={'r2dz_x': self.r2dz_x,
                                             'r2yz_dx': self.r2yz_dx,
                                             'bound_label': self.bound_label,
                                             'treatment': self.treatment,
                                             'adjusted_estimate': self.adjusted_estimate,
                                             'adjusted_se': self.adjusted_se,
                                             'adjusted_t': self.adjusted_t,
                                             'adjusted_lower_CI': self.adjusted_lower_CI,
                                             'adjusted_upper_CI': self.adjusted_upper_CI})

        if self.benchmark_covariates is not None and self.model is not None:
            self.bench_bounds = ovb_bounds.ovb_bounds(self.model, self.treatment,
                                                      benchmark_covariates=self.benchmark_covariates, kd=self.kd,
                                                      ky=self.ky, alpha=self.alpha, h0=self.h0, reduce=self.reduce)
        elif self.r2dxj_x is not None and self.estimate is not None:
            self.benchmark_covariates = 'manual_benchmark'
            # bound_label = ovb_bounds.label_maker(benchmark_covariate=self.benchmark_covariates, kd=kd, ky=ky)
            bench_bounds = ovb_bounds.ovb_partial_r2_bound(r2dxj_x=self.r2dxj_x, r2yxj_dx=self.r2yxj_dx, kd=kd, ky=ky)
            bench_bounds['adjusted_estimate'] = bias_functions.adjusted_estimate(self.r2dz_x, self.r2yz_dx,
                                                                                 estimate=self.estimate, se=self.se,
                                                                                 reduce=self.reduce)
            bench_bounds['adjusted_se'] = bias_functions.adjusted_estimate(self.r2dz_x, self.r2yz_dx,
                                                                           se=self.se, reduce=self.reduce)
            bench_bounds['adjusted_t'] = bias_functions.adjusted_t(self.r2dz_x, self.r2yz_dx, estimate=self.estimate,
                                                                   se=self.se, reduce=self.reduce)
            se_multiple = abs(t.ppf(alpha / 2, model.model.df_resid))  # number of SEs within CI based on alpha
            bench_bounds['adjusted_lower_CI'] = bench_bounds['adjusted_estimate'] - \
                se_multiple * bench_bounds['adjusted_se']
            bench_bounds['adjusted_upper_CI'] = bench_bounds['adjusted_estimate'] + \
                se_multiple * bench_bounds['adjusted_se']

        if self.bounds is None:
            self.bounds = self.bench_bounds
        else:
            self.bounds.append(self.bench_bounds)

    def summary(self, digits=3):
        """
        Print a summary of the sensitivity results for a Sensemakr object, including robustness value, extreme
        confounding scenario, and benchmark bounding. digits is the number of digits to round numbers to; default is 3.
        Following the example above, if you have a Sensemakr object called sensitivity, call this method using
        sensitivity.summary(). To round to 5 digits instead of 3, call sensitivity.summary(digits=5).
        """
        if self.reduce:
            h0 = round(self.estimate * (1 - self.q), digits)
            direction = "reduce"
        else:
            h0 = round(self.estimate * (1 + self.q), digits)
            direction = "increase"

        print("Sensitivity Analysis to Unobserved Confounding\n")
        if self.model is not None:
            model_formula = self.model.model.endog_names + ' ~ ' + ' + '.join(self.model.model.exog_names)
            print("Model Formula: " + model_formula + "\n")
        print("Null hypothesis: q =", self.q, "and reduce =", self.reduce, "\n")
        print("-- This means we are considering biases that", direction, "the absolute value of the current estimate.")
        print("-- The null hypothesis deemed problematic is H0:tau =", h0, "\n")

        print("Unadjusted Estimates of '", self.treatment, "':")
        print("  Coef. estimate:", round(self.estimate, digits))
        print("  Standard Error:", round(self.sensitivity_stats['se'], digits))
        print("  t-value:", round(self.sensitivity_stats['t_statistic'], digits), "\n")

        print("Sensitivity Statistics:")
        print("  Partial R2 of treatment with outcome:", round(self.sensitivity_stats['r2yd_x'], digits))
        print("  Robustness Value, q =", self.q, ":", round(self.sensitivity_stats['rv_q'], digits))
        print("  Robustness Value, q =", self.q, "alpha =", self.alpha, ":",
              round(self.sensitivity_stats['rv_qa'], digits), "\n")

        print("Verbal interpretation of sensitivity statistics:\n")
        print("-- Partial R2 of the treatment with the outcome: an extreme confounder (orthogonal to the covariates) ",
              "that explains 100% of the residual variance of the outcome, would need to explain at least",
              100.0 * self.sensitivity_stats['r2yd_x'], "% of the residual variance of the treatment "
                                                        "to fully account for the observed estimated effect.\n")

        print("-- Robustness Value,", "q =", self.q, ": unobserved confounders (orthogonal to the covariates) that ",
              "explain more than", 100.0 * self.sensitivity_stats['rv_q'], "% of the residual variance",
              "of both the treatment and the outcome are strong enough to bring the point estimate to", h0,
              "(a bias of", 100.0 * self.q, "% of the original estimate). Conversely, unobserved confounders that "
              "do not explain more than", 100.0 * self.sensitivity_stats['rv_q'], "% of the residual variance",
              "of both the treatment and the outcome are not strong enough to bring the point estimate to", h0, ".\n")

        print("-- Robustness Value,", "q =", self.q, ",", "alpha =", self.alpha, ": unobserved confounders (orthogonal "
              "to the covariates) that explain more than", 100.0 * self.sensitivity_stats['rv_qa'], "% of the residual "
              "variance of both the treatment and the outcome are strong enough to bring the estimate to a range where "
              "it is no longer 'statistically different' from", h0, "(a bias of", 100.0 * self.q, "% of the original "
              "estimate), at the significance level of alpha =", self.alpha, ".", "Conversely, unobserved confounders "
              "that do not explain more than", 100.0 * self.sensitivity_stats['rv_qa'], "% of the residual variance",
              "of both the treatment and the outcome are not strong enough to bring the estimate to a range where "
              "it is no longer 'statistically different' from", h0, ", at the significance level of alpha =",
              self.alpha, ".\n")

        print("Bounds on omitted variable bias:\n--The table below shows the maximum strength of unobserved confounders"
              " with association with the treatment and the outcome bounded by a multiple of the observed explanatory"
              " power of the chosen benchmark covariate(s).\n")
        print(self.bounds)
