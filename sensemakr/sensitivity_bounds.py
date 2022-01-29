"""
Bounds on the strength of unobserved confounders using observed covariates, as in Cinelli and Hazlett (2020).

The main generic function is ovb_bounds, which can compute both the bounds on the strength of confounding
as well as the adjusted estimates, standard errors, t-values and confidence intervals.

Other functions that compute only the bounds on the strength of confounding are also provided. These functions
may be useful when computing benchmarks for using only summary statistics from papers you see in print.

Currently it implements only the bounds based on partial R2. Other bounds will be implemented soon.

Reference:
------------
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias." Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Example:
------------
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
------------
"""
# Computes bounds on the strength of unobserved confounders using observed covariates
import sys
from . import bias_functions
from . import sensitivity_statistics
from scipy.stats import t
import pandas as pd
import numpy as np
import statsmodels.api as sm

def ovb_bounds(model, treatment, benchmark_covariates=None, kd=1, ky=None, alpha=0.05, h0=0, reduce=True,
               bound='partial r2', adjusted_estimates=True):
    """
    Provide bounds on the strength of unobserved confounders using observed covariates, as in Cinelli and Hazlett (2020).

    The main generic function is ovb_bounds, which can compute both the bounds on the strength of confounding
    as well as the adjusted estimates, standard errors, t-values and confidence intervals.

    Other functions that compute only the bounds on the strength of confounding are also provided. These functions
    may be useful when computing benchmarks for using only summary statistics from papers you see in print.

    Currently it implements only the bounds based on partial R2. Other bounds will be implemented soon.

    :Required parameters: model and treatment.

    Parameters
    ----------
    model : statsmodels OLSResults object
        a fitted statsmodels OLSResults object for the restricted regression model you have provided.
    treatment : string
        a string with the name of the "treatment" variable, e.g. the independent variable of interest.
    benchmark_covariates : string or list of strings
        a string or list of strings with names of the variables to use for benchmark bounding.
    kd : float or list of floats
        a float or list of floats with each being a multiple of the strength of association between a
        benchmark variable and the treatment variable to test with benchmark bounding (Default value = 1).
    ky : float or list of floats
        same as kd except measured in terms of strength of association with the outcome variable.
    alpha : float
        a float with the significance level for the robustness value RV_qa to render the
        estimate not significant (Default value = 0.05).
    h0 : float
        a float with the null hypothesis effect size; defaults to 0.
    reduce : boolean
        whether to reduce (True, default) or increase (False) the estimate due to putative confounding.
    bound : string
        type of bound to perform; as of now, only partial R^2 bounding is allowed (Default value = 'partial r2').
    adjusted_estimates : boolean
        whether to compute bias-adjusted estimates, standard errors, and t-statistics (Default value = True).

    Returns
    -------
    Pandas DataFrame

        A Pandas DataFrame containing the following variables:

        **treatment** : the name of the provided treatment variable.

        **bound_label** : a string created by label_maker to serve as a label for the bound for printing & plotting purposes.

        **r2dz_x** : a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
        with the treatment variable "d", with observed covariates "x" partialed out, as implied by z being kd-times
        as strong as the benchmark_covariates.

        **r2yz_dx** : a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
        with the outcome variable "y", with observed covariates "x" and the treatment variable "d" partialed out,
        as implied by z being ky-times as strong as the benchmark_covariates.

        **adjusted_estimate** : the bias-adjusted estimate adjusted for a confounder with the given r2dz_x and r2yz_dx above.

        **adjusted_se** : the bias-adjusted standard error adjusted for a confounder with the given r2dz_x and r2yz_dx above.

        **adjusted_t** : the bias-adjusted t-statistic adjusted for a confounder with the given r2dz_x and r2yz_dx above.


    Example
    -------

    >>> # Load example dataset
    >>> from sensemakr import data
    >>> darfur = data.load_darfur()
    >>> # Fit a statsmodels OLSResults object ("fitted_model")
    >>> import statsmodels.formula.api as smf
    >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
    >>> fitted_model = model.fit()
    >>> # Bounds on the strength of confounders 1, 2, or 3 times as strong as female
    >>> # and 1, 2, or 3 times as strong as pastvoted
    >>> from sensemakr import sensitivity_bounds
    >>> sensitivity_bounds.ovb_bounds(model = fitted_model, treatment = "directlyharmed", benchmark_covariates = ["female", "pastvoted"], kd = [1, 2, 3]) # doctest: +SKIP
    """
    if ky is None:
        ky = kd
    if bound != 'partial r2':
        sys.exit('Only partial r2 is implemented as of now.')
    bounds = ovb_partial_r2_bound(model=model, treatment=treatment,
                                  benchmark_covariates=benchmark_covariates, kd=kd, ky=ky)

    if adjusted_estimates:
        bounds['treatment'] = treatment
        bounds['adjusted_estimate'] = bias_functions.adjusted_estimate(bounds['r2dz_x'], bounds['r2yz_dx'], model=model,
                                                                       treatment=treatment, reduce=reduce)
        bounds['adjusted_se'] = bias_functions.adjusted_se(bounds['r2dz_x'], bounds['r2yz_dx'], model=model,
                                                           treatment=treatment)
        bounds['adjusted_t'] = bias_functions.adjusted_t(bounds['r2dz_x'], bounds['r2yz_dx'], model=model,
                                                         treatment=treatment, reduce=reduce, h0=h0)

        se_multiple = abs(t.ppf(alpha / 2, model.model.df_resid))  # number of SEs within CI based on alpha
        bounds['adjusted_lower_CI'] = bounds['adjusted_estimate'] - se_multiple * bounds['adjusted_se']
        bounds['adjusted_upper_CI'] = bounds['adjusted_estimate'] + se_multiple * bounds['adjusted_se']
    return bounds


def ovb_partial_r2_bound(model=None, treatment=None, r2dxj_x=None, r2yxj_dx=None,
                         benchmark_covariates=None, kd=1, ky=None):
    """
    Provide a Pandas DataFrame with the bounds on the strength of the unobserved confounder.

    Adjusted estimates, standard errors and t-values (among other quantities) need to be computed
    manually by the user using those bounds with the functions adjusted_estimate, adjusted_se and adjusted_t.

    :Required parameters: (model and treatment) or (r2dxj_x and r2yxj_dx).

    Parameters
    ----------
    model : statsmodels OLSResults object
        a fitted statsmodels OLSResults object for the restricted regression model you have provided.
    treatment : string
        a string with the name of the "treatment" variable, e.g. the independent variable of interest.
    r2dxj_x : float
        float with the partial R2 of covariate Xj with the treatment D (after partialling out the effect of the remaining covariates X, excluding Xj).
    r2yxj_dx : float
        float with the partial R2 of covariate Xj with the outcome Y (after partialling out the effect of the remaining covariates X, excluding Xj).
    benchmark_covariates : string or list of strings
        a string or list of strings with names of the variables to use for benchmark bounding.
    kd : float or list of floats
        a float or list of floats with each being a multiple of the strength of association between a
        benchmark variable and the treatment variable to test with benchmark bounding (Default value = 1).
    ky : float or list of floats
        same as kd except measured in terms of strength of association with the outcome variable (Default value = None).

    Returns
    -------
    Pandas DataFrame

        A Pandas DataFrame containing the following variables:

        **bound_label** : a string created by label_maker to serve as a label for the bound for printing & plotting purposes.

        **r2dz_x** : a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
        with the treatment variable "d", with observed covariates "x" partialed out, as implied by z being kd-times
        as strong as the benchmark_covariates.

        **r2yz_dx** : a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
        with the outcome variable "y", with observed covariates "x" and the treatment variable "d" partialed out,
        as implied by z being ky-times as strong as the benchmark_covariates.



    Examples
    ---------
        Let's construct bounds from summary statistics only. Suppose you didn't have access to the data, but only to the treatment and outcome regression tables.
        You can still compute the bounds.

    >>> # First import the necessary libraries.
    >>> from sensemakr import *
    >>> # Use the t statistic of female in the outcome regression to compute the partial R2 of female with the outcome.
    >>> r2yxj_dx = partial_r2(t_statistic = -9.789, dof = 783)
    >>> # Use the t-value of female in the *treatment* regression to compute the partial R2 of female with the treatment.
    >>> r2dxj_x = partial_r2(t_statistic = -2.680, dof = 783)
    >>> # Compute manually bounds on the strength of confounders 1, 2, or 3 times as strong as female.
    >>> bounds = ovb_partial_r2_bound(r2dxj_x = r2dxj_x, r2yxj_dx = r2yxj_dx,kd = [1, 2, 3], ky = [1, 2, 3])
    >>> # Compute manually adjusted estimates.
    >>> bound_values = adjusted_estimate(estimate = 0.0973, se = 0.0232, dof = 783, r2dz_x = bounds['r2dz_x'], r2yz_dx = bounds['r2yz_dx'])
    >>> # Plot contours and bounds.
    >>> ovb_contour_plot(estimate = 0.0973, se = 0.0232, dof = 783)
    >>> add_bound_to_contour(bounds=bounds, bound_value = bound_values)
    """
    if (model is None or treatment is None) and (r2dxj_x is None or r2yxj_dx is None):
        sys.exit('Error: ovb_partial_r2_bound requires either a statsmodels OLSResults object and a treatment name'
                 'or the partial R^2 values with the benchmark covariate, r2dxj_x and r2yxj_dx.')
    if (treatment is not None and type(treatment) is not str):
        sys.exit('Error: treatment must be a single string.')
    if ((benchmark_covariates is None) and (r2dxj_x is not None)) :
        #return None
        benchmark_covariates=['manual']
    elif(benchmark_covariates is None):
        return None
    elif type(benchmark_covariates) is str:
        benchmark_covariates = [benchmark_covariates]
    else:
        if ((type(benchmark_covariates) is not list) and (type(benchmark_covariates) is not dict)):
            sys.exit('Benchmark covariates must be a string, list of strings, 2d list containing only strings or dict containing only strings and list of strings.')
        if (type(benchmark_covariates) is list):
            for i in benchmark_covariates:
                if type(i) is not str and (type(i) is not list or any(type(j) is not str for j in i)):
                    sys.exit('Benchmark covariates must be a string, list of strings, 2d list containing only strings or dict containing only strings and list of strings.')
        else: #benchmark_covariates is a dict
            for i in benchmark_covariates:
                if(type(benchmark_covariates[i]) is not str and (type(benchmark_covariates[i]) is not list or any(type(j) is not str for j in benchmark_covariates[i]))):
                    sys.exit('Benchmark covariates must be a string, list of strings, 2d list containing only strings or dict containing only strings and list of strings.')

    if model is not None:
        m = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
        d = np.array(m[treatment])
        non_treatment = m.drop(columns=treatment)  # all columns except treatment
        non_treatment.insert(0, 0, 1)  # add constant term for regression
        treatment_model = sm.OLS(d, non_treatment)
        treatment_results = treatment_model.fit()

        if type(benchmark_covariates) is str:
            # r2yxj_dx = partial R^2 with outcome; r2dxj_x = partial R^2 with treatment
            r2yxj_dx = [sensitivity_statistics.partial_r2(model, covariates=benchmark_covariates)]
            r2dxj_x = [sensitivity_statistics.partial_r2(treatment_results, covariates=benchmark_covariates)]
        elif(type(benchmark_covariates) is list):
            r2yxj_dx, r2dxj_x = [], []
            for b in benchmark_covariates:
              	r2yxj_dx.append(sensitivity_statistics.group_partial_r2(model, covariates=b))
              	r2dxj_x.append(sensitivity_statistics.group_partial_r2(treatment_results, covariates=b))
     	# Group Benchmark
        elif(type(benchmark_covariates) is dict):
            r2yxj_dx, r2dxj_x = [], []
            for b in benchmark_covariates:
                r2yxj_dx.append(sensitivity_statistics.group_partial_r2(model, benchmark_covariates[b]))
                r2dxj_x.append(sensitivity_statistics.group_partial_r2(treatment_results, benchmark_covariates[b]))
    elif r2dxj_x is not None:
        if np.isscalar(r2dxj_x):
            r2dxj_x = [r2dxj_x]
        if np.isscalar(r2yxj_dx):
            r2yxj_dx = [r2yxj_dx]

    bounds = pd.DataFrame()
    for i in range(len(benchmark_covariates)):
        r2dxj_x[i], r2yxj_dx[i] = sensitivity_statistics.check_r2(r2dxj_x[i], r2yxj_dx[i])
        if type(kd) is list:
            kd = np.array(kd)
        if ky is None:
            ky=kd
        r2dz_x = kd * (r2dxj_x[i] / (1 - r2dxj_x[i]))
        if (np.isscalar(r2dz_x) and r2dz_x >= 1) or (not np.isscalar(r2dz_x) and any(i >= 1 for i in r2dz_x)):
            sys.exit("Implied bound on r2dz.x >= 1. Impossible kd value. Try a lower kd.")
        r2zxj_xd = kd * (r2dxj_x[i] ** 2) / ((1 - kd * r2dxj_x[i]) * (1 - r2dxj_x[i]))
        if (np.isscalar(r2zxj_xd) and r2zxj_xd >= 1) or (not np.isscalar(r2zxj_xd) and any(i >= 1 for i in r2zxj_xd)):
            sys.exit("Impossible kd value. Try a lower kd.")
        r2yz_dx = ((np.sqrt(ky) + np.sqrt(r2zxj_xd)) / np.sqrt(1 - r2zxj_xd)) ** 2 * (r2yxj_dx[i] / (1 - r2yxj_dx[i]))
        if (np.isscalar(r2yz_dx) and r2yz_dx > 1) or (not np.isscalar(r2yz_dx) and any(i > 1 for i in r2yz_dx)):
            print('Warning: Implied bound on r2yz.dx greater than 1, try lower kd and/or ky. Setting r2yz.dx to 1.')
            r2yz_dx[r2yz_dx > 1] = 1
        if(type(benchmark_covariates) is not dict):
            if np.isscalar(kd):
                bound_label = label_maker(benchmark_covariate=benchmark_covariates[i], kd=kd, ky=ky)
                bounds = bounds.append({'bound_label': bound_label, 'r2dz_x': r2dz_x, 'r2yz_dx': r2yz_dx},
                                       ignore_index=True)
            else:
                for j in range(len(kd)):
                    bound_label = label_maker(benchmark_covariate=benchmark_covariates[i], kd=kd[j], ky=ky[j])
                    bounds = bounds.append({'bound_label': bound_label, 'r2dz_x': r2dz_x[j], 'r2yz_dx': r2yz_dx[j]},
                                           ignore_index=True)
        else:
            if np.isscalar(kd):
                bound_label = label_maker(benchmark_covariate=list(benchmark_covariates)[i], kd=kd, ky=ky)
                bounds = bounds.append({'bound_label': bound_label, 'r2dz_x': r2dz_x, 'r2yz_dx': r2yz_dx},
                                       ignore_index=True)
            else:
                for j in range(len(kd)):
                    bound_label = label_maker(benchmark_covariate=list(benchmark_covariates)[i], kd=kd[j], ky=ky[j])
                    bounds = bounds.append({'bound_label': bound_label, 'r2dz_x': r2dz_x[j], 'r2yz_dx': r2yz_dx[j]},
                                           ignore_index=True)

    return bounds


def label_maker(benchmark_covariate, kd, ky, digits=2):
    """
    Return a string created by appending the covariate name to the multiplier(s) ky and (if applicable) kd.

    Parameters
    ----------
    benchmark_covariates : string or list of strings
        a string or list of strings with names of the variables to use for benchmark bounding.
    kd : float or list of floats
        a float or list of floats with each being a multiple of the strength of association between a
        benchmark variable and the treatment variable to test with benchmark bounding (Default value = 1).
    ky : float or list of floats
        same as kd except measured in terms of strength of association with the outcome variable (Default value = None).
    digits : int
        rouding digit of ky/kd shown in the string (Default value = 2).

    Returns
    -------

    """
    if benchmark_covariate is None:
        return 'manual'
    else:
        variable_text = ' ' + str(benchmark_covariate)
    if ky == kd:
        multiplier_text = str(round(ky, digits))
    else:
        multiplier_text = str(round(kd, digits)) + '/' + str(round(ky, digits))
    bound_label = multiplier_text + 'x' + variable_text
    return bound_label
