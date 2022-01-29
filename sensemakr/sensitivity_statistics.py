"""
Computes the sensitivity statistics: robustness value, partial R2, and Cohen's f2; plus helper functions.

Reference:
------------
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias." Journal of the Royal Statistical Society, Series B (Statistical Methodology).

Example:
------------
See specific functions below.

Functions
------------
"""
# Computes the sensitivity statistics: robustness value, partial R2, and Cohen's f2; plus helper functions
import sys
from scipy.stats import t
import numpy as np
import pandas as pd


def robustness_value(model=None, covariates=None, t_statistic=None, dof=None, q=1, alpha=1.0):
    """
    Compute the robustness value of a regression coefficient.

    The robustness value describes the
    minimum strength of association (parameterized in terms of partial R2) that omitted variables would need to have
    both with the treatment and with the outcome to change the estimated coefficient by a certain amount
    (for instance, to bring it down to zero).

    For instance, a robustness value of 1% means that an unobserved confounder that explain 1% of the residual variance
    of the outcome and 1% of the residual variance of the treatment is strong enough to explain away the estimated
    effect. Whereas a robustness value of 90% means that any unobserved confounder that explain less than 90% of the
    residual variance of both the outcome and the treatment assignment cannot fully account for the observed effect.
    You may also compute robustness value taking into account sampling uncertainty.
    See details in Cinelli and Hazlett (2020).

    The function robustness_value can take as input a statsmodels OLSResults object or you may directly pass
    the t-value and degrees of freedom.

    **Required parameters:** either model or t_statistic and dof.

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    covariates : string
        a string or list of strings with the names of the variables to use for benchmark bounding.
    t_statistic : float
        a float with the t_statistic for the restricted model regression.
    dof : int
        an int with the degrees of freedom of the restricted regression.
    q : float
        a float with the percent to reduce the point estimate by for the robustness value RV_q (Default value = 1).
    alpha : float
        a float with the significance level for the robustness value RV_qa to render the estimate not significant (Default value = 1.0).

    Returns
    -------
    numpy array
        a numpy array with the robustness value

    Examples
    --------
    >>> # Load example dataset
    >>> from sensemakr import data
    >>> darfur = data.load_darfur()
    >>> # Fit a statsmodels OLSResults object ("fitted_model")
    >>> import statsmodels.formula.api as smf
    >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
    >>> fitted_model = model.fit()
    >>> from sensemakr import sensitivity_statistics
    >>> # Robustness value of directly harmed q =1 (reduce estimate to zero):
    >>> sensitivity_statistics.robustness_value(model = fitted_model, covariates = "directlyharmed") # doctest: +SKIP
    >>> # Robustness value of directly harmed q = 1/2 (reduce estimate in half):
    >>> sensitivity_statistics.robustness_value(model = fitted_model, covariates = "directlyharmed", q = 1/2) # doctest: +SKIP
    >>> # Robustness value of directly harmed q = 1/2, alpha = 0.05 (reduce estimate in half, with 95% confidence):
    >>> sensitivity_statistics.robustness_value(model = fitted_model, covariates = "directlyharmed", q = 1/2, alpha = 0.05) # doctest: +SKIP
    >>> # You can also provide the statistics directly:
    >>> sensitivity_statistics.robustness_value(t_statistic = 4.18445, dof = 783) # doctest: +SKIP
    """
    if model is None and (t_statistic is None or dof is None):
        sys.exit('Error: robustness_value requires either a statsmodels OLSResults object '
                 'or a t-statistic and degrees of freedom.')
    check_q(q)
    check_alpha(alpha)

    if model is not None:
        model_data = model_helper(model, covariates=covariates)
        t_statistic = model_data['t_statistics']
        dof = int(model_data['dof'])
    elif type(t_statistic) is float or type(t_statistic) is int:
        t_statistic = pd.Series(t_statistic)

    fq = q * abs(t_statistic / np.sqrt(dof))  # Cohen's f for given q value
    f_crit = abs(t.ppf(alpha / 2, dof - 1)) / np.sqrt(dof - 1)  # computes critical f
    fqa = fq - f_crit  # f for q and alpha values

    rv = 0.5 * (np.sqrt(fqa**4 + (4 * fqa**2)) - fqa**2)  # constraint binding case
    rvx = (fq**2 - f_crit**2)/(1 + fq**2)  # constraint not binding case

    # combined results
    rv_out = rv
    rv_out[fqa < 0] = 0
    rv_out[(fqa > 0) & (fq > 1 / f_crit)] = rvx[(fqa > 0) & (fq > 1 / f_crit)]

    # set attributes and return
    # rv_out['q'] = q
    # rv_out['alpha'] = alpha
    return rv_out


def partial_r2(model=None, covariates=None, t_statistic=None, dof=None):
    r"""
    Compute the partial R2 for a linear regression model.

    The partial R2 describes how much of the residual variance of the outcome (after partialing out
    the other covariates) a covariate explains.

    The partial R2 can be used as an extreme-scenario sensitivity analysis to omitted variables.
    Considering an unobserved confounder that explains 100% of the residual variance of the outcome,
    the partial R2 describes how strongly associated with the treatment this unobserved confounder would need to be
    in order to explain away the estimated effect.

    For details see Cinelli and Hazlett (2020).

    **Required parameters:** either model or t_statistic and dof.

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    covariates : string or list of strings
        a string or list of strings with the covariates used to compute the t_statistic and dof
        from the model. If not specified, defaults to all variables.
    t_statistic : float
        a float with the t_statistic for the restricted model regression.
    dof : int
        an int with the degrees of freedom of the restricted regression.

    Returns
    -------
    float
        a float with the computed partial R^2.



    Examples
    ---------
        This function takes as input a statsmodels OLSResults object or you may pass directly t-value & degrees of freedom.
        For partial R2 of groups of covariates, check group_partial_r2.

    >>> # Load example dataset:
    >>> from sensemakr import data
    >>> darfur = data.load_darfur()
    >>> # Fit a statsmodels OLSResults object ("fitted_model"):
    >>> import statsmodels.formula.api as smf
    >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
    >>> fitted_model = model.fit()
    >>> # Load this module:
    >>> from sensemakr import sensitivity_statistics
    >>> # Partial R2 of directly harmed with peacefactor:
    >>> sensitivity_statistics.partial_r2(model = fitted_model, covariates = "directlyharmed")  # doctest: +NUMBER
    0.02187
    >>> # Partial R2 of female with peacefactor:
    >>> sensitivity_statistics.partial_r2(model = fitted_model, covariates = "female")  # doctest: +NUMBER
    0.10903
    >>> # You can also provide the statistics directly:
    >>> sensitivity_statistics.partial_r2(t_statistic = 4.18445, dof = 783)  # doctest: +NUMBER
    0.021873
    """
    if model is None and (t_statistic is None or dof is None):
        sys.exit('Error: partial_r2 requires either a statsmodels OLSResults object '
                 'or a t-statistic and degrees of freedom.')

    if model is not None:
        model_data = model_helper(model, covariates=covariates)
        t_statistic = model_data['t_statistics']
        dof = model_data['dof']
        return (t_statistic ** 2 / (t_statistic ** 2 + dof))[0]  # extracts float
    else:
        return t_statistic ** 2 / (t_statistic ** 2 + dof)


def partial_f2(model=None, covariates=None, t_statistic=None, dof=None):
    r"""
    Compute the partial (Cohen's) f2 for a linear regression model.

    The partial (Cohen's) f2 is a common measure of effect size (a transformation of the partial R2) that can
    also be used directly for sensitivity analysis using a bias factor table.
    For details see Cinelli and Hazlett (2020).

    This function takes as input a statsmodels OLSResults object or you may pass directly t-value & degrees of freedom.

    **Required parameters:** either model or (t_statistic and dof).

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    covariates : string or list of strings
        a string or list of strings with the covariates used to compute the t_statistic and dof
        from the model. If not specified, defaults to all variables.
    t_statistic : float
        a float with the t_statistic for the restricted model regression.
    dof : int
        an int with the degrees of freedom of the restricted regression.

    Returns
    -------
    float
        a float with the computed partial f^2.

    Examples
    ---------
    >>> # Load example dataset:
    >>> from sensemakr import data
    >>> darfur = data.load_darfur()
    >>> # Fit a statsmodels OLSResults object ("fitted_model"):
    >>> import statsmodels.formula.api as smf
    >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
    >>> fitted_model = model.fit()
    >>> # Load this module:
    >>> from sensemakr import sensitivity_statistics
    >>> # Partial f2 of directly harmed with peacefactor:
    >>> sensitivity_statistics.partial_f2(model = fitted_model, covariates = "directlyharmed") # doctest: +SKIP
    >>> # Partial f2 of female with peacefactor:
    >>> sensitivity_statistics.partial_f2(model = fitted_model, covariates = "female") # doctest: +SKIP
    >>> # You can also provide the statistics directly:
    >>> sensitivity_statistics.partial_f2(t_statistic = 4.18445, dof = 783) # doctest: +NUMBER
    0.022362
    """
    if model is None and (t_statistic is None or dof is None):
        sys.exit('Error: partial_f2 requires either a statsmodels OLSResults object '
                 'or a t-statistic and degrees of freedom.')

    if model is not None:
        model_data = model_helper(model, covariates=covariates)
        t_statistic = model_data['t_statistics']
        dof = model_data['dof']

    return t_statistic ** 2 / dof


def partial_f(model=None, covariates=None, t_statistic=None, dof=None):
    """
    Calculate the square root of the partial_f2 function described above.

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    covariates : string or list of strings
        a string or list of strings with the covariates used to compute the t_statistic and dof
        from the model. If not specified, defaults to all variables.
    t_statistic : float
        a float with the t_statistic for the restricted model regression.
    dof : int
        an int with the degrees of freedom of the restricted regression.

    Returns
    -------
    float
        a float with the computed partial f.
    """
    return np.sqrt(partial_f2(model, covariates, t_statistic, dof))


def group_partial_r2(model=None, covariates=None, f_statistic=None, p=None, dof=None):
    r"""
    Partial R2 of groups of covariates in a linear regression model.

    This function computes the partial R2 of a group of covariates in a linear regression model. Multivariate version
    of the partial_r2 function; see that for more details.

    **Required parameters:** either model or (f_statistic, p, and dof).

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    covariates : string or list of strings
        a string or list of strings with the covariates used to compute the t_statistic and dof
        from the model. If not specified, defaults to all variables.
    f_statistic : float
        a float with the f_statistic for the restricted model regression.
    p : int
        an int with the number of parameters in the model.
    dof : int
        an int with the degrees of freedom of the restricted regression.

    Returns
    -------
    float
        a float with the computed group partial R^2.

    Examples
    ---------
    >>> # Load example dataset:
    >>> from sensemakr import data
    >>> darfur = data.load_darfur()
    >>> # Fit a statsmodels OLSResults object ("fitted_model"):
    >>> import statsmodels.formula.api as smf
    >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
    >>> fitted_model = model.fit()
    >>> from sensemakr import sensitivity_statistics
    >>> sensitivity_statistics.group_partial_r2(model = fitted_model, covariates = ["female", "pastvoted"]) # doctest: +NUMBER
    0.11681
    """
    if (model is None or covariates is None) and (f_statistic is None or p is None or dof is None):
        sys.exit('Error: group_partial_r2 requires either a statsmodels OLSResults object and covariates or an '
                 'f-statistic, number of parameters, and degrees of freedom.')
    if((f_statistic is None or p is None or dof is None)):
        params = model.params
        check_covariates(model.model.exog_names, covariates)
        params = params[covariates]
        if np.isscalar(params):
            return partial_r2(model=model, covariates=covariates, t_statistic=f_statistic, dof=dof)
        v = model.cov_params().loc[covariates, :][covariates]  # variance-covariance matrix
        dof = model.df_resid
        p = len(params)
        f_statistic = np.matmul(np.matmul(params.values.T, np.linalg.inv(v.values)), params.values) / p
    r2 = f_statistic * p / (f_statistic * p + dof)
    return r2


def sensitivity_stats(model=None, treatment=None, estimate=None, se=None, dof=None, q=1, alpha=0.05, reduce=True):
    r"""
    Computes the robustness_value, partial_r2 and partial_f2 of the coefficient of interest.

    **Required parameters:** either model and treatment, or (estimate, se, and dof).

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    treatment : string
        a string with treatment variable name.
    estimate : float
        a float with the coefficient estimate of the restricted regression.
    se : float
        a float with the standard error of the restricted regression.
    dof : int
        an int with the degrees of freedom of the restricted regression.
    q : float
        a float with the percent to reduce the point estimate by for the robustness value RV_q (Default value = 1).
    alpha : float
        a float with the significance level for the robustness value RV_qa to render the estimate not significant (Default value = 0.05).
    reduce : boolean
        whether to reduce or increase the estimate due to confounding (Default value = True).

    Returns
    -------
    Pandas DataFrame
        a Pandas DataFrame containing the following quantities:

        **treatment** : a string with the name of the treatment variable.

        **estimate** : a float with the estimated effect of the treatment.

        **se** : a float with the estimated standard error of the treatment effect.

        **t_statistics** : a float with  the t-value of the treatment.

        **r2yd_x** : a float with the partial R2 of the treatment and the outcome, see details in partial_r2.

        **rv_q** : a float the robustness value of the treatment, see details in robustness_value.

        **rv_qa** : a float with the robustness value of the treatment considering statistical significance, see details in robustness_value.

        **f2yd_x** : a float with the partial (Cohen's) f2 of the treatment with the outcome, see details in partial_f2.

        **dof** : an int with the degrees of freedom of the model.

    Examples
    ---------
    >>> # Load example dataset:
    >>> from sensemakr import data
    >>> darfur = data.load_darfur()
    >>> # Fit a statsmodels OLSResults object ("fitted_model"):
    >>> import statsmodels.formula.api as smf
    >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
    >>> fitted_model = model.fit()
    >>> from sensemakr import sensitivity_statistics
    >>> # Sensitivity stats for directly harmed:
    >>> sensitivity_statistics.sensitivity_stats(model = fitted_model, treatment = "directlyharmed") # doctest: +SKIP
    >>> # You can  also pass the numeric values directly:
    >>> sensitivity_statistics.sensitivity_stats(estimate = 0.09731582, se = 0.02325654, dof = 783) # doctest: +SKIP
    """
    if (model is None or treatment is None) and (estimate is None or se is None or dof is None):
        sys.exit('Error: sensitivity_stats requires either a statsmodels OLSResults object and treatment name or an '
                 'estimate, standard error, and degrees of freedom.')
    if model is not None:
        if type(treatment) is not str:
            sys.exit('Error: must provide only one treatment variable.')
        model_data = model_helper(model, covariates=treatment)
        estimate = list(model_data['estimate'])[0]
        se = list(model_data['se'])[0]
        dof = int(model_data['dof'])

    check_q(q)
    check_alpha(alpha)
    check_se(se)
    check_dof(dof)

    if reduce:
        h0 = estimate * (1 - q)
    else:
        h0 = estimate * (1 + q)
    original_t = estimate / se
    t_statistic = (estimate - h0) / se
    r2yd_x = partial_r2(t_statistic=original_t, dof=dof)
    rv_q = list(robustness_value(t_statistic=original_t, dof=dof, q=q))[0]
    rv_qa = list(robustness_value(t_statistic=original_t, dof=dof, q=q, alpha=alpha))[0]
    f2yd_x = partial_f2(t_statistic=original_t, dof=dof)
    sensitivity_stats_df = {'estimate': estimate, 'se': se, 't_statistic': t_statistic,
                            'r2yd_x': r2yd_x, 'rv_q': rv_q, 'rv_qa': rv_qa, 'f2yd_x': f2yd_x, 'dof': dof}
    return sensitivity_stats_df


# Helper function for quickly extracting properties from a model, allowing specification of a subset of covariates
def model_helper(model, covariates=None):
    """
    Internal function for extracting info from a statsmodels OLSResults object and returning it in a dict.

    Parameters
    ----------
    model : statsmodels OLSResults object
        a statsmodels OLSResults object containing the restricted regression.
    covariates : string or list of strings
        a string or list of strings with the covariates used to compute the t_statistic and dof
        from the model. If not specified, defaults to all variables.

    Returns
    -------

    """
    error_if_no_dof(model)  # check to make sure there aren't zero residual degrees of freedom for this model
    if covariates is not None:
        covariates = check_covariates(model.model.exog_names, covariates)
        used_variables = covariates
    else:
        used_variables = model.model.exog_names  # use all variables if no covariates specified
    model_info = {
        'covariates': used_variables,
        'estimate': model.params[used_variables],
        'se': model.bse[used_variables],
        't_statistics': model.tvalues[used_variables],
        'dof': int(model.df_resid)
    }
    return model_info


# Variable validators for sensitivity stats and sensemakr

def check_r2(r2dz_x, r2yz_dx):
    """
    Ensure that r2dz_x and r2yz_dx are numpy scalars or arrays.

    Parameters
    ----------
    r2dz_x : float or list of floats
        a float or list of floats with the partial R^2 of a putative unobserved
        confounder "z" with the treatment variable "d", with observed covariates "x" partialed out.
    r2yz_dx : float or list of floats
        a float or list of floats with the  partial R^2 of a putative unobserved
        confounder "z" with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.


    Returns
    -------

    """
    if r2dz_x is None:
        return r2dz_x, r2yz_dx
    if type(r2dz_x) is float or type(r2dz_x) is int:
        r2dz_x = np.float64(r2dz_x)
    elif type(r2dz_x) is list:
        r2dz_x = np.array(r2dz_x)
    if type(r2yz_dx) is float or type(r2yz_dx) is int:
        r2yz_dx = np.float64(r2yz_dx)
    elif type(r2yz_dx) is list:
        r2yz_dx = np.array(r2yz_dx)
    for r in [r2dz_x, r2yz_dx]:
        if np.isscalar(r) and not np.issubdtype(r, np.number):
            sys.exit('Partial R^2 must be a number or array of numbers between zero and one.')
        elif not np.isscalar(r):
            r = np.array(r)
            if not(all(np.issubdtype(i, np.number) and 0 <= i <= 1 for i in r)):
                sys.exit('Partial R^2 must be a number or array of numbers between zero and one.')
    return r2dz_x, r2yz_dx


def check_q(q):
    """
    Ensure that q, the percent reduction to the point estimate for RV_q, is a float or int greater than 0.

    Parameters
    ----------
    q : float
        a float with the percent to reduce the point estimate by for the robustness value RV_q (Default value = 1).

    Returns
    -------

    """
    if (type(q) is not float and type(q) is not int) or q < 0:
        sys.exit('Error: the q parameter must be a single number greater than 0. q was: ' + str(q))


def check_alpha(alpha):
    """
    Ensure that alpha, the significance level for RV_qa, is a float between 0 and 1.

    Parameters
    ----------
    alpha : float
        a float with the significance level for the robustness value RV_qa to
        render the estimate not significant (Default value = 0.05).

    Returns
    -------

    """
    if type(alpha) is not float or alpha < 0 or alpha > 1:
        sys.exit('Error: alpha must be between 0 and 1. alpha was: ' + str(alpha))


def check_se(se):
    """
    Ensure that standard error is a float greater than zero.

    Parameters
    ----------
    se : float
        a float with the standard error of the restricted regression.

    Returns
    -------

    """
    if (type(se) is not float and type(se) is not int) or se < 0:
        sys.exit('Standard error provided must be a single non-negative number. SE was: ' + str(se))


def check_dof(dof):
    """
    Ensure that the degrees of freedom for a regression is a positive integer.

    Parameters
    ----------
    dof : int
        an int with the degrees of freedom of the restricted regression.

    Returns
    -------

    """
    dof = float(dof)
    if type(dof) is float and dof.is_integer():
        dof = int(dof)
    if type(dof) is not int or dof <= 0:
        sys.exit('Error: degrees of freedom provided must be a single positive integer. DOF was: ' + str(dof))


def error_if_no_dof(model):
    """
    For a given statsmodels OLSResults object, ensure that its degrees of freedom is not zero.

    Parameters
    ----------
    model : statsmodels OLSResults object.

    Returns
    -------

    """
    if model.df_resid == 0:
        sys.exit('Error: There are 0 residual degrees of freedom in the regression model provided.')


def check_covariates(all_names, covariates):
    """
    Ensure that all provided covariates are strings and are in the regression model.

    Parameters
    ----------
    all_names : list of strings.

    covariates : string or list of strings.

    Returns
    -------

    """
    if covariates is not None:
        if type(covariates) is str:
            covariates = [covariates]  # make into a list if it's only a single string
        if not all(type(i) is str for i in covariates):
            sys.exit('Error: Treatment and covariates names must be strings.')
        not_found = [i for i in covariates if i not in all_names]
        if len(not_found) > 0:
            sys.exit('Variables not found in model: ' + ', '.join(not_found))
    return covariates
