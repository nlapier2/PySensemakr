"""
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
----------
For all methods, r2dz_x and r2yz_dx are required. For all methods other than bf, either model and treatment
or estimate, se, and dof are also required, except adjused_se and bias which do not accept the estimate parameter.

List of parameters
^^^^^^^^^^^^^^^^^^^^
r2dz_x :
    a float or list of floats with the partial R^2 of a putative unobserved confounder "z" with the treatment variable "d", with observed covariates "x" partialed out.
r2yz_dx :
    a float or list of floats with the  partial R^2 of a putative unobserved confounder "z" with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.
model :
    a fitted statsmodels OLSResults object for the restricted regression model you have provided.
treatment :
    a string with the name of the "treatment" variable, e.g. the independent variable of interest.
estimate :
    a float with the unadjusted estimate of the coefficient for the independent variable of interest.
se :
    a float with the unadjusted standard error of the regression.
dof :
    an int with the degrees of freedom of the regression.
reduce :
    whether to reduce (True, default) or increase (False) the estimate due to putative confounding.

Parameters only used in param_check
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
function_name :
    string with the name of the calling function, used to print the function name in error messages.
estimate_is_param :
    flag for whether estimate should be a required parameter for the calling function.
reduce_is_param :
    flag for whether reduce is a parameter for the calling function.

Reference
------------
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias." Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Example
------------
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
"""


from . import sensitivity_statistics
import sys
import numpy as np


def adjusted_estimate(r2dz_x, r2yz_dx, model=None, treatment=None, estimate=None, se=None, dof=None, reduce=True):
    """Compute the bias-adjusted coefficient estimate.

    Parameters
    ----------
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

    Returns
    -------
    float
        the bias-adjusted coefficient estimate

    """
    r2dz_x, r2yz_dx, estimate, se, dof = param_check('adjusted_estimate', r2dz_x, r2yz_dx, model=model,
                                                     treatment=treatment, estimate=estimate,
                                                     se=se, dof=dof, reduce=reduce)
    if reduce:
        return np.sign(estimate) * (abs(estimate) - bias(r2dz_x, r2yz_dx, se=se, dof=dof))
    else:
        return np.sign(estimate) * (abs(estimate) + bias(r2dz_x, r2yz_dx, se=se, dof=dof))


def adjusted_se(r2dz_x, r2yz_dx, model=None, treatment=None, se=None, dof=None):
    """Compute the bias-adjusted regression standard error.

    Parameters
    ----------
    r2dz_x : float or list of floats
        a float or list of floats with the partial R^2 of a putative unobserved confounder "z" with the treatment variable "d", with observed covariates "x" partialed out.
    r2yz_dx : float or list of floats
        a float or list of floats with the  partial R^2 of a putative unobserved confounder "z" with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.
    model : statsmodels OLSResults object
        a fitted statsmodels OLSResults object for the restricted regression model you have provided.
    treatment : string
        a string with the name of the "treatment" variable, e.g. the independent variable of interest.
    se : float
        a float with the unadjusted standard error of the regression.
    dof : int
        an int with the degrees of freedom of the regression.

    Returns
    -------
    float
        bias-adjusted regression standard error
    """
    r2dz_x, r2yz_dx, estimate, se, dof = param_check('adjusted_se', r2dz_x, r2yz_dx, model=model, treatment=treatment,
                                                     se=se, dof=dof, estimate_is_param=False, reduce_is_param=False)
    new_se = np.sqrt((1 - r2yz_dx) / (1 - r2dz_x)) * se * np.sqrt(dof / (dof - 1))
    return new_se


def adjusted_t(r2dz_x, r2yz_dx, model=None, treatment=None, estimate=None, se=None, dof=None, reduce=True, h0=0):
    """Compute bias-adjusted t-statistic, (adjusted_estimate - h0) / adjusted_se.

    Parameters
    ----------
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
    h0 : float
        the test value for null hypothesis.

    Returns
    -------
    float
        bias-adjusted t-statistic, (adjusted_estimate - h0) / adjusted_se.

    """
    r2dz_x, r2yz_dx, estimate, se, dof = param_check('adjusted_t', r2dz_x, r2yz_dx, model=model, treatment=treatment,
                                                     estimate=estimate, se=se, dof=dof, reduce=reduce)
    new_estimate = adjusted_estimate(estimate=estimate, r2yz_dx=r2yz_dx, r2dz_x=r2dz_x, se=se, dof=dof, reduce=reduce)
    new_t = (new_estimate - h0) / adjusted_se(r2yz_dx=r2yz_dx, r2dz_x=r2dz_x, se=se, dof=dof)
    return new_t  # , h0


def adjusted_partial_r2(r2dz_x, r2yz_dx, model=None, treatment=None, estimate=None, se=None, dof=None,
                        reduce=True, h0=0):
    """Compute the bias-adjusted partial R2, based on adjusted_t.

    Parameters
    ----------
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
    h0 : float
        the test value for null hypothesis.

    Returns
    -------
    float
        the bias-adjusted partial R2, based on adjusted_t.

    """
    r2dz_x, r2yz_dx, estimate, se, dof = param_check('adjusted_partial_r2', r2dz_x, r2yz_dx, model=model,
                                                     treatment=treatment, estimate=estimate,
                                                     se=se, dof=dof, reduce=reduce)
    new_t = adjusted_t(estimate=estimate, r2yz_dx=r2yz_dx, r2dz_x=r2dz_x, se=se, dof=dof, reduce=reduce, h0=h0)
    return sensitivity_statistics.partial_r2(t_statistic=new_t, dof=dof-1)


def bias(r2dz_x, r2yz_dx, model=None, treatment=None, se=None, dof=None):
    """Compute the omitted variable bias for the partial R2 parameterization.

    Parameters
    ----------
    r2dz_x : float or list of floats
        a float or list of floats with the partial R^2 of a putative unobserved confounder "z" with the treatment variable "d", with observed covariates "x" partialed out.
    r2yz_dx : float or list of floats
        a float or list of floats with the  partial R^2 of a putative unobserved confounder "z" with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.
    model : statsmodels OLSResults object
        a fitted statsmodels OLSResults object for the restricted regression model you have provided.
    treatment : string
        a string with the name of the "treatment" variable, e.g. the independent variable of interest.
    se : float
        a float with the unadjusted standard error of the regression.
    dof : int
        an int with the degrees of freedom of the regression.

    Returns
    -------
    float
        the omitted variable bias for the partial R2 parameterization.

    """
    r2dz_x, r2yz_dx, estimate, se, dof = param_check('bias', r2dz_x, r2yz_dx, model=model, treatment=treatment,
                                                     se=se, dof=dof, estimate_is_param=False, reduce_is_param=False)
    bias_val = bf(r2dz_x, r2yz_dx) * se * np.sqrt(dof)  # numpy array
    return bias_val


def relative_bias(r2dz_x, r2yz_dx, model=None, treatment=None, estimate=None, se=None, dof=None):
    """Compute the relative bias for the partial R2 parameterization.

    Parameters
    ----------
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

    Returns
    -------
    float
        the relative bias for the partial R2 parameterization.

    """
    r2dz_x, r2yz_dx, estimate, se, dof = param_check('relative_bias', r2dz_x, r2yz_dx, model=model, treatment=treatment,
                                                     estimate=estimate, se=se, dof=dof, reduce_is_param=False)
    t_statistic = abs(estimate / se)
    f = sensitivity_statistics.partial_f(t_statistic=t_statistic, dof=dof)
    bf_val = bf(r2dz_x, r2yz_dx)
    q = bf_val / f
    return q


def rel_bias(r_est, est):
    """Compute the relative bias for any estimator and the truth value.

    Parameters
    ----------
    r_est : float
        a float or list of floats of a reference value.
    est : float
        a float or list of floats of an estimator.

    Returns
    -------

    """
    r_est, est = np.array(r_est), np.array(est)
    return (r_est - est) / r_est


def bf(r2dz_x, r2yz_dx):
    """Compute the bias function for the partial R2 parameters. See description at top for details.

    Parameters
    ----------
    r2dz_x : float or list of floats
        a float or list of floats with the partial R^2 of a putative unobserved confounder "z" with the treatment variable "d", with observed covariates "x" partialed out.
    r2yz_dx : float or list of floats
        a float or list of floats with the  partial R^2 of a putative unobserved confounder "z" with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.


    Returns
    -------

    """
    r2dz_x, r2yz_dx = np.array(r2dz_x), np.array(r2yz_dx)
    return np.sqrt((r2yz_dx * r2dz_x) / (1 - r2dz_x))


def param_check(function_name, r2dz_x, r2yz_dx,
                model=None, treatment=None, estimate=None, se=None, dof=None, reduce=True,
                estimate_is_param=True, reduce_is_param=True):
    """Helper method that checks whether the required parameters have been passed in and have valid values.

    Also extracts data from a statsmodels OLSResults object (if one is given) for use in numerical formulae.
    See description at top for details.

    Parameters
    ----------
    function_name : string
        name of the function to check parameters.
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
    h0 : float
        the test value for null hypothesis.
    estimate_is_param : boolean
         whether or not estimate is a parameter to check.
    reduce_is_param : boolean
         whether or not reduce is a parameter to check.

    Returns
    -------

    """
    if estimate_is_param:
        if (model is None or treatment is None) and (estimate is None or se is None or dof is None):
            sys.exit('Error: in addition to r2yz_dx & r2dz_x, ' + function_name +
                     ' requires either a statsmodels OLSResults object and a treatment variable'
                     ' or the current estimate, standard error, and degrees of freedom.')
    else:
        if (model is None or treatment is None) and (se is None or dof is None):
            sys.exit('Error: in addition to r2yz_dx & r2dz_x, ' + function_name +
                     ' requires either a statsmodels OLSResults object and a treatment variable'
                     ' or the current standard error and degrees of freedom.')

    if model is not None:
        if type(treatment) is not str:
            sys.exit('Error in ' + function_name + ' method: must provide only one treatment variable.')
        model_data = sensitivity_statistics.model_helper(model, covariates=treatment)  # extracts model data
        estimate = list(model_data['estimate'])[0]  # extract the raw float value
        se = list(model_data['se'])[0]
        dof = int(model_data['dof'])
    sensitivity_statistics.check_se(se)
    sensitivity_statistics.check_dof(dof)
    r2dz_x, r2yz_dx = sensitivity_statistics.check_r2(r2dz_x, r2yz_dx)
    if estimate_is_param and type(estimate) is not float and type(estimate) is not int:
        sys.exit('Error in ' + function_name + ' method: provided estimate must be a single number.')
    if reduce_is_param and type(reduce) is not bool:
        sys.exit('Error in ' + function_name + ' method: reduce must be True or False boolean value.')
    return r2dz_x, r2yz_dx, estimate, se, dof
