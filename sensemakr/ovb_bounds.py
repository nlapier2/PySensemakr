# Computes bounds on the strength of unobserved confounders using observed covariates
import sys
from . import bias_functions
from . import sensitivity_stats
from scipy.stats import t
import pandas as pd
import numpy as np
import statsmodels.api as sm


def ovb_bounds(model, treatment, benchmark_covariates=None, kd=1, ky=None, alpha=1.0, h0=0, reduce=True,
               bound='partial r2', adjusted_estimates=True):
    """
    Bounds on the strength of unobserved confounders using observed covariates, as in Cinelli and Hazlett (2020).
    The main generic function is ovb_bounds, which can compute both the bounds on the strength of confounding
    as well as the adjusted estimates, standard errors, t-values and confidence intervals.

    Other functions that compute only the bounds on the strength of confounding are also provided. These functions
    may be useful when computing benchmarks for using only summary statistics from papers you see in print.

    Currently it implements only the bounds based on partial R2. Other bounds will be implemented soon.

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

    # bounds on the strength of confounders 1, 2, or 3 times as strong as female
    # and 1,2, or 3 times as strong as pastvoted
    from sensemakr import ovb_bounds
    ovb_bounds.ovb_bounds(model = fitted_model, treatment = "directlyharmed",
               benchmark_covariates = ["female", "pastvoted"], kd = [1, 2, 3])

    Required parameters: model and treatment
    :param model: a fitted statsmodels OLSResults object for the restricted regression model you have provided
    :param treatment: a string with the name of the "treatment" variable, e.g. the independent variable of interest
    :param benchmark_covariates: a string or list of strings with names of the variables to use for benchmark bounding
    :param kd: a float or list of floats with each being a multiple of the strength of association between a
            benchmark variable and the treatment variable to test with benchmark bounding
    :param ky: same as kd except measured in terms of strength of association with the outcome variable
    :param alpha: a float with the significance level for the robustness value RV_qa to render the
            estimate not significant
    :param h0: a float with the null hypothesis effect size; defaults to 0
    :param reduce: whether to reduce (True, default) or increase (False) the estimate due to putative confounding
    :param bound: type of bound to perform; as of now, only partial R^2 bounding is allowed
    :param adjusted_estimates: whether to compute bias-adjusted estimates, standard errors, and t-statistics
    :return: A Pandas DataFrame containing the following variables:
      * treatment: the name of the provided treatment variable
      * bound_label: a string created by label_maker to serve as a label for the bound for printing & plotting purposes
      * r2dz_x: a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
            with the treatment variable "d", with observed covariates "x" partialed out, as implied by z being kd-times
            as strong as the benchmark_covariates
      * r2yz_dx: a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
            with the outcome variable "y", with observed covariates "x" and the treatment variable "d" partialed out,
            as implied by z being ky-times as strong as the benchmark_covariates
      * adjusted_estimate: the bias-adjusted estimate adjusted for a confounder with the given r2dz_x and r2yz_dx above
      * adjusted_se: the bias-adjusted standard error adjusted for a confounder with the given r2dz_x and r2yz_dx above
      * adjusted_t: the bias-adjusted t-statistic adjusted for a confounder with the given r2dz_x and r2yz_dx above
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
                         benchmark_covariates=None, kd=None, ky=None):
    """
    The function `ovb_partial_r2_bound()` returns only a Pandas DataFrame with the bounds on the strength of the
    unobserved confounder. Adjusted estimates, standard errors and t-values (among other quantities) need to be computed
    manually by the user using those bounds with the functions adjusted_estimate, adjusted_se and adjusted_t.

    Reference:
    Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
        Journal of the Royal Statistical Society, Series B (Statistical Methodology).

    Examples:
    #########################################################
    ## Let's construct bounds from summary statistics only ##
    #########################################################
    # Suppose you didn't have access to the data, but only to
    # the treatment and outcome regression tables.
    # You can still compute the bounds.

    # first import the necessary libraries
    from sensemakr import sensitivity_stats
    from sensemakr import bias_functions
    from sensemakr import ovb_bounds

    # Use the t statistic of female in the outcome regression
    # to compute the partial R2 of female with the outcome.
    r2yxj_dx = sensitivity_stats.partial_r2(t_statistic = -9.789, dof = 783)

    # Use the t-value of female in the *treatment* regression
    # to compute the partial R2 of female with the treatment
    r2dxj_x = sensitivity_stats.partial_r2(t_statistic = -2.680, dof = 783)

    #### Compute manually bounds on the strength of confounders 1, 2, or 3
    #### times as strong as female
    ###bounds = ovb_bounds.ovb_partial_r2_bound(r2dxj_x = r2dxj_x, r2yxj_dx = r2yxj_dx,
    ###                              kd = [1, 2, 3], ky = [1, 3, 3], bound_label = "[1, 2, 3]x female")
    #### Compute manually adjusted estimates
    ###bound_values = bias_functions.adjusted_estimate(estimate = 0.0973, se = 0.0232, dof = 783,
    ###                                 r2dz_x = bounds['r2dz_x'], r2yz_dx = bounds['r2yz_dx'])

    #### Plot contours and bounds
    ###ovb_contour_plot(estimate = 0.0973, se = 0.0232, dof = 783)
    ###add_bound_to_contour(bounds, bound_value = bound_values)

    
    Required parameters: model and treatment or r2dxj_x and r2yxj_dx
    :param model: a fitted statsmodels OLSResults object for the restricted regression model you have provided
    :param treatment: a string with the name of the "treatment" variable, e.g. the independent variable of interest
    :param r2dxj_x: float with the partial R2 of covariate Xj with the treatment D
        (after partialling out the effect of the remaining covariates X, excluding Xj).
    :param r2yxj_dx: float with the partial R2 of covariate Xj with the outcome Y
        (after partialling out the effect of the remaining covariates X, excluding Xj).
    :param benchmark_covariates: a string or list of strings with names of the variables to use for benchmark bounding
    :param kd: a float or list of floats with each being a multiple of the strength of association between a
            benchmark variable and the treatment variable to test with benchmark bounding
    :param ky: same as kd except measured in terms of strength of association with the outcome variable
    :return: A Pandas DataFrame containing the following variables:
      * bound_label: a string created by label_maker to serve as a label for the bound for printing & plotting purposes
      * r2dz_x: a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
            with the treatment variable "d", with observed covariates "x" partialed out, as implied by z being kd-times
            as strong as the benchmark_covariates
      * r2yz_dx: a float or list of floats with the partial R^2 of a putative unobserved confounder "z"
            with the outcome variable "y", with observed covariates "x" and the treatment variable "d" partialed out,
            as implied by z being ky-times as strong as the benchmark_covariates
    """
    if (model is None or treatment is None) and (r2dxj_x is None or r2yxj_dx is None):
        sys.exit('Error: ovb_partial_r2_bound requires either a statsmodels OLSResults object and a treatment name'
                 'or the partial R^2 values with the benchmark covariate, r2dxj_x and r2yxj_dx.')
    if type(treatment) is not str:
        sys.exit('Error: treatment must be a single string.')
    if benchmark_covariates is None:
        return None
    elif type(benchmark_covariates) is str:
        benchmark_covariates = [benchmark_covariates]
    else:
        if type(benchmark_covariates) is not list:
            sys.exit('Benchmark covariates must be a string, list of strings or 2d list containing only strings.')
        for i in benchmark_covariates:
            if type(i) is not str and (type(i) is not list or any(type(j) is not str for j in i)):
                sys.exit('Benchmark covariates must be a string, list of strings or 2d list containing only strings.')

    if model is not None:
        m = pd.DataFrame(model.model.exog, columns=model.model.exog_names)
        d = np.array(m[treatment])
        non_treatment = m.drop(columns=treatment)  # all columns except treatment
        non_treatment.insert(0, 0, 1)  # add constant term for regression
        treatment_model = sm.OLS(d, non_treatment)
        treatment_results = treatment_model.fit()

        if type(benchmark_covariates) is str:
            # r2yxj_dx = partial R^2 with outcome; r2dxj_x = partial R^2 with treatment
            r2yxj_dx = [sensitivity_stats.partial_r2(model, covariates=benchmark_covariates)]
            r2dxj_x = [sensitivity_stats.partial_r2(treatment_results, covariates=benchmark_covariates)]
        else:
            r2yxj_dx, r2dxj_x = [], []
            for b in benchmark_covariates:
                r2yxj_dx.append(sensitivity_stats.group_partial_r2(model, covariates=b))
                r2dxj_x.append(sensitivity_stats.group_partial_r2(treatment_results, covariates=b))
    elif r2dxj_x is not None:
        if type(r2dxj_x) is int or type(r2dxj_x) is float:
            r2dxj_x = [r2dxj_x]
        if type(r2yxj_dx) is int or type(r2yxj_dx) is float:
            r2yxj_dx = [r2yxj_dx]

    bounds = pd.DataFrame()
    for i in range(len(benchmark_covariates)):
        r2dxj_x[i], r2yxj_dx[i] = sensitivity_stats.check_r2(r2dxj_x[i], r2yxj_dx[i])
        if type(kd) is list:
            kd = np.array(kd)
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
        if type(kd) is int or type(kd) is float:
            bound_label = label_maker(benchmark_covariate=benchmark_covariates[i], kd=kd, ky=ky)
            bounds = bounds.append({'bound_label': bound_label, 'r2dz_x': r2dz_x, 'r2yz_dx': r2yz_dx},
                                   ignore_index=True)
        else:
            for j in range(len(kd)):
                bound_label = label_maker(benchmark_covariate=benchmark_covariates[i], kd=kd[j], ky=ky[j])
                bounds = bounds.append({'bound_label': bound_label, 'r2dz_x': r2dz_x[j], 'r2yz_dx': r2yz_dx[j]},
                                       ignore_index=True)
    return bounds


def label_maker(benchmark_covariate, kd, ky, digits=2):
    """ Returns a string created by appending the covariate name to the multiplier(s) ky and (if applicable) kd. """
    if benchmark_covariate is None:
        variable_text = '\n'
    else:
        variable_text = ' ' + str(benchmark_covariate)
    if ky == kd:
        multiplier_text = str(round(ky, digits))
    else:
        multiplier_text = str(round(kd, digits)) + '/' + str(round(ky, digits))
    bound_label = multiplier_text + 'x' + variable_text
    return bound_label
