r"""
Sensitivity analysis to unobserved confounders.

This class performs sensitivity analysis to omitted variables as discussed in Cinelli and Hazlett (2020).
It returns an object of class Sensemakr with several pre-computed sensitivity statistics for reporting.
After creating the object, you may directly use the plot and summary methods of the returned object.

Sensemakr is a convenience class. You may use the other sensitivity functions of the package directly,
such as the functions for sensitivity plots (ovb_contour_plot, ovb_extreme_plot), the functions for
computing bias-adjusted estimates and t-values (adjusted_estimate, adjusted_t), the functions for computing the
robustness value and partial R2 (robustness_value, partial_r2),  or the functions for bounding the strength
of unobserved confounders (ovb_bounds), among others.

**Parameters:**
arguments passed to other methods. First argument should either be

* a statsmodels OLSResults object ("fitted_model"); or
* the numerical estimated value of the coefficient, along with the numeric values of standard errors and
  degrees of freedom (arguments estimate, se and df).

**Return:**
An object of class Sensemakr, containing:

* sensitivity_stats : A Pandas DataFrame with the sensitivity statistics for the treatment variable,
  as computed by the function sensitivity_stats.

* bounds : A pandas DataFrame with bounds on the strength of confounding according to some benchmark covariates,
  as computed by the function ovb_bounds.

**Reference:**
Cinelli, C. and Hazlett, C. (2020), "Making Sense of Sensitivity: Extending Omitted Variable Bias."
Journal of the Royal Statistical Society, Series B (Statistical Methodology).


Examples
--------
>>> # Load example dataset:
>>> from sensemakr import data
>>> darfur = data.load_darfur()
>>> # Fit a statsmodels OLSResults object ("fitted_model")
>>> import statsmodels.formula.api as smf
>>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
>>> fitted_model = model.fit()
>>> # Runs sensemakr for sensitivity analysis
>>> from sensemakr import main
>>> sensitivity = main.Sensemakr(fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3])
>>> # Description of results
>>> sensitivity.summary() # doctest: +SKIP
"""


import sys
import pandas as pd
from scipy.stats import t
import numpy as np
from . import sensitivity_statistics
from . import bias_functions
from . import sensitivity_bounds
from . import sensitivity_plots

class Sensemakr:
    r"""
    The constructor for a Sensemakr object.

    Parameter descriptions are below. For usage and info, see the
    description of the class.

    Required parameters: model and treatment, or estimate, se, and dof.

    Parameters
    ----------
    model: statsmodels OLSResults object
        a fitted statsmodels OLSResults object for the restricted regression model you have provided.
    treatment: string
        a string with the name of the "treatment" variable, e.g. the independent variable of interest.
    estimate: float
        a float with the estimate of the coefficient for the independent variable of interest.
    se: float
        a float with the standard error of the regression.
    dof: int
        an int with the degrees of freedom of the regression.
    benchmark_covariates: string or list of strings
        a string or list of strings with the names of the variables to use for benchmark bounding.
    kd: float or list of floats
        a float or list of floats with each being a multiple of the strength of association between a
        benchmark variable and the treatment variable to test with benchmark bounding.
    ky: float or list of floats
        same as kd except measured in terms of strength of association with the outcome variable.
    q: float
        a float with the percent to reduce the point estimate by for the robustness value RV_q.
    alpha: float
        a float with the significance level for the robustness value RV_qa to render the estimate not significant.
    r2dz_x: float or list of floats
        a float or list of floats with the partial R^2 of a putative unobserved confounder "z" with the treatment variable "d", with observed covariates "x" partialed out. In this case, you are manually
        specifying a putative confounder's strength rather than benchmarking.
    r2yz_dx: float or list of floats
        a float or list of floats with the  partial R^2 of a putative unobserved confounder "z"
        with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.
        In this case, you are manually specifying a putative confounder's strength rather than benchmarking.
    r2dxj_x: float
        float with the partial R2 of covariate Xj with the treatment D
        (after partialling out the effect of the remaining covariates X, excluding Xj).
    r2yxj_dx: float
        float with the partial R2 of covariate Xj with the outcome Y
        (after partialling out the effect of the remaining covariates X, excluding Xj).
    bound_label: string
        a string what to call the name of a bounding variable, for printing and plotting purposes.
    reduce: boolean
        whether to reduce (True, default) or increase (False) the estimate due to putative confounding.
    """

    def __init__(self, model=None, treatment=None, estimate=None, se=None, dof=None, benchmark_covariates=None, kd=1,
                 ky=None, q=1, alpha=0.05, r2dz_x=None, r2yz_dx=None, r2dxj_x=None, r2yxj_dx=None,
                 bound_label="Manual Bound", reduce=True):
        r"""
        Construct for a Sensemakr object.

        Parameter descriptions are below. For usage and info, see the
        description of the class.

        Required parameters: model and treatment, or estimate, se, and dof

        Parameters
        ----------
        model: statsmodels OLSResults object
            a fitted statsmodels OLSResults object for the restricted regression model you have provided
        treatment: string
            a string with the name of the "treatment" variable, e.g. the independent variable of interest
        estimate: float
            a float with the estimate of the coefficient for the independent variable of interest
        se: float
            a float with the standard error of the regression
        dof: int
            an int with the degrees of freedom of the regression
        benchmark_covariates: string or list of strings
            a string or list of strings with the names of the variables to use for benchmark bounding
        kd: float or list of floats
            a float or list of floats with each being a multiple of the strength of association between a
            benchmark variable and the treatment variable to test with benchmark bounding
        ky: float or list of floats
            same as kd except measured in terms of strength of association with the outcome variable
        q: float
            a float with the percent to reduce the point estimate by for the robustness value RV_q
        alpha: float
            a float with the significance level for the robustness value RV_qa to render the estimate not significant
        r2dz_x: float or list of floats
            a float or list of floats with the partial R^2 of a putative unobserved confounder "z" with the treatment variable "d", with observed covariates "x" partialed out. In this case, you are manually
            specifying a putative confounder's strength rather than benchmarking.
        r2yz_dx: float or list of floats
            a float or list of floats with the  partial R^2 of a putative unobserved confounder "z"
            with the outcome variable "y", with observed covariates "x" and treatment variable "d" partialed out.
            In this case, you are manually specifying a putative confounder's strength rather than benchmarking.
        r2dxj_x: float
            float with the partial R2 of covariate Xj with the treatment D
            (after partialling out the effect of the remaining covariates X, excluding Xj).
        r2yxj_dx: float
            float with the partial R2 of covariate Xj with the outcome Y
            (after partialling out the effect of the remaining covariates X, excluding Xj).
        bound_label: string
            a string what to call the name of a bounding variable, for printing and plotting purposes
        reduce: boolean
            whether to reduce (True, default) or increase (False) the estimate due to putative confounding
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
            self.sensitivity_stats = sensitivity_statistics.sensitivity_stats(model=self.model, treatment=self.treatment,
                                                                         q=self.q, alpha=self.alpha, reduce=self.reduce)
            self.estimate = self.sensitivity_stats['estimate']
            self.se = self.sensitivity_stats['se']
            self.dof = self.model.df_resid
        else:
            self.model=None
            self.estimate = estimate
            self.sensitivity_stats = sensitivity_statistics.sensitivity_stats(estimate=self.estimate, se=se, dof=dof,
                                                                         q=self.q, alpha=self.alpha, reduce=self.reduce)
            #self.estimate = estimate
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
                self.r2dz_x, self.r2yz_dx = sensitivity_statistics.check_r2(self.r2dz_x, self.r2yz_dx)
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
                self.r2dz_x, self.r2yz_dx = sensitivity_statistics.check_r2(self.r2dz_x, self.r2yz_dx)
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
            if(not np.isscalar(self.r2dz_x)):
	            self.bounds = pd.DataFrame(data={'r2dz_x': self.r2dz_x,
	                                             'r2yz_dx': self.r2yz_dx,
	                                             'bound_label': self.bound_label,
	                                             'treatment': self.treatment,
	                                             'adjusted_estimate': self.adjusted_estimate,
	                                             'adjusted_se': self.adjusted_se,
	                                             'adjusted_t': self.adjusted_t,
	                                             'adjusted_lower_CI': self.adjusted_lower_CI,
	                                             'adjusted_upper_CI': self.adjusted_upper_CI})
            else:
                self.bounds = pd.DataFrame(data={'r2dz_x': self.r2dz_x,
	                                             'r2yz_dx': self.r2yz_dx,
	                                             'bound_label': self.bound_label,
	                                             'treatment': self.treatment,
	                                             'adjusted_estimate': self.adjusted_estimate,
	                                             'adjusted_se': self.adjusted_se,
	                                             'adjusted_t': self.adjusted_t,
	                                             'adjusted_lower_CI': self.adjusted_lower_CI,
	                                             'adjusted_upper_CI': self.adjusted_upper_CI},index=[0])

        if self.benchmark_covariates is not None and self.model is not None:
            self.bench_bounds = sensitivity_bounds.ovb_bounds(self.model, self.treatment,
                                                      benchmark_covariates=self.benchmark_covariates, kd=self.kd,
                                                      ky=self.ky, alpha=self.alpha, h0=self.h0, reduce=self.reduce)
        elif self.r2dxj_x is not None and self.estimate is not None:
            if self.benchmark_covariates is None:
                self.benchmark_covariates = 'manual_benchmark'
            # bound_label = ovb_bounds.label_maker(benchmark_covariate=self.benchmark_covariates, kd=kd, ky=ky)
            self.bench_bounds = sensitivity_bounds.ovb_partial_r2_bound(r2dxj_x=self.r2dxj_x, r2yxj_dx=self.r2yxj_dx, kd=kd, ky=ky,benchmark_covariates=self.benchmark_covariates)
            if (self.bench_bounds is not None):
                self.r2dz_x=self.bench_bounds['r2dz_x'].values
                self.r2yz_dx=self.bench_bounds['r2yz_dx'].values
                self.bench_bounds['adjusted_estimate'] = bias_functions.adjusted_estimate(self.r2dz_x, self.r2yz_dx, estimate=self.estimate, se=self.se, dof=self.dof,reduce=self.reduce)
                self.bench_bounds['adjusted_se'] = bias_functions.adjusted_se(self.r2dz_x, self.r2yz_dx,se=self.se,dof=self.dof)
                self.bench_bounds['adjusted_t'] = bias_functions.adjusted_t(self.r2dz_x, self.r2yz_dx, estimate=self.estimate,se=self.se, reduce=self.reduce,dof=self.dof)
                se_multiple = abs(t.ppf(alpha / 2, self.dof))  # number of SEs within CI based on alpha
                self.bench_bounds['adjusted_lower_CI'] = self.bench_bounds['adjusted_estimate'] - \
                    se_multiple * self.bench_bounds['adjusted_se']
                self.bench_bounds['adjusted_upper_CI'] = self.bench_bounds['adjusted_estimate'] + \
                    se_multiple * self.bench_bounds['adjusted_se']
        else:
            self.bench_bounds = None

        if self.bounds is None:
            self.bounds = self.bench_bounds
        else:
            self.bounds = self.bounds.append(self.bench_bounds).reset_index()

    def __repr__(self):
        """Print a short summary of the sensitivity results for a Sensemakr object, including formula, hypothesis, and sensitivity analysis.

        Following the example above, if you have a Sensemakr object called sensitivity, call this method using
        print(sensitivity) or just sensitivity.

        """
        digits=3
        if self.reduce:
            h0 = round(self.estimate * (1 - self.q), digits)
            direction = "reduce"
        else:
            h0 = round(self.estimate * (1 + self.q), digits)
            direction = "increase"

        s="Sensitivity Analysis to Unobserved Confounding\n\n"
        if self.model is not None:
            #model_formula = self.model.model.endog_names + ' ~ ' + ' + '.join(self.model.model.exog_names)
            #print("Model Formula: " + model_formula + "\n")
            s+="Model Formula: "+self.model.model.formula+"\n\n"
        s+="Null hypothesis: q = "+str(self.q)+ " and reduce = "+str(self.reduce)+"\n\n"

        s+="Unadjusted Estimates of '"+str(self.treatment)+ "':\n"
        s+="  Coef. estimate: "+str(round(self.estimate, digits))+"\n"
        s+="  Standard Error: "+str(round(self.sensitivity_stats['se'], digits))+'\n'
        s+="  t-value: "+str(round(self.sensitivity_stats['t_statistic'], digits))+ "\n\n"

        s+="Sensitivity Statistics: \n"
        s+="  Partial R2 of treatment with outcome: "+str(round(self.sensitivity_stats['r2yd_x'], digits))+'\n'
        s+="  Robustness Value, q = "+str(self.q)+ " : "+str(round(self.sensitivity_stats['rv_q'], digits))+'\n'
        s+="  Robustness Value, q = "+str(self.q)+ " alpha = "+ str(self.alpha)+ " : "+ \
        str(round(self.sensitivity_stats['rv_qa'], digits))+ "\n"

        return s


    def summary(self, digits=3):
        """
        Print a summary of the sensitivity results for a Sensemakr object, including robustness value, extreme confounding scenario, and benchmark bounding.

        digits is the number of digits to round numbers to; default is 3.
        Following the example above, if you have a Sensemakr object called sensitivity, call this method using
        sensitivity.summary(). To round to 5 digits instead of 3, call sensitivity.summary(digits=5).

        Parameters
        ----------
        digits : int
             Rounding digit for summary (Default value = 3).

        Returns
        -------

        """
        if self.reduce:
            h0 = round(self.estimate * (1 - self.q), digits)
            direction = "reduce"
        else:
            h0 = round(self.estimate * (1 + self.q), digits)
            direction = "increase"

        print("Sensitivity Analysis to Unobserved Confounding\n")
        if self.model is not None:
            #model_formula = self.model.model.endog_names + ' ~ ' + ' + '.join(self.model.model.exog_names)
            #print("Model Formula: " + model_formula + "\n")
            print("Model Formula: "+self.model.model.formula+"\n")
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
              round(100.0 * self.sensitivity_stats['r2yd_x'],digits), "% of the residual variance of the treatment "
                                                        "to fully account for the observed estimated effect.\n")

        print("-- Robustness Value,", "q =", self.q, ": unobserved confounders (orthogonal to the covariates) that ",
              "explain more than", round(100.0 * self.sensitivity_stats['rv_q'],digits), "% of the residual variance",
              "of both the treatment and the outcome are strong enough to bring the point estimate to", h0,
              "(a bias of", 100.0 * self.q, "% of the original estimate). Conversely, unobserved confounders that "
              "do not explain more than", round(100.0 * self.sensitivity_stats['rv_q'],digits), "% of the residual variance",
              "of both the treatment and the outcome are not strong enough to bring the point estimate to", h0, ".\n")

        print("-- Robustness Value,", "q =", self.q, ",", "alpha =", self.alpha, ": unobserved confounders (orthogonal "
              "to the covariates) that explain more than", round(100.0 * self.sensitivity_stats['rv_qa'],digits), "% of the residual "
              "variance of both the treatment and the outcome are strong enough to bring the estimate to a range where "
              "it is no longer 'statistically different' from", h0, "(a bias of", 100.0 * self.q, "% of the original "
              "estimate), at the significance level of alpha =", self.alpha, ".", "Conversely, unobserved confounders "
              "that do not explain more than", round(100.0 * self.sensitivity_stats['rv_qa'],digits), "% of the residual variance",
              "of both the treatment and the outcome are not strong enough to bring the estimate to a range where "
              "it is no longer 'statistically different' from", h0, ", at the significance level of alpha =",
              self.alpha, ".\n")

        if self.bounds is not None:
            print("Bounds on omitted variable bias:\n--The table below shows the maximum strength of unobserved confounders"
                  " with association with the treatment and the outcome bounded by a multiple of the observed explanatory"
                  " power of the chosen benchmark covariate(s).\n")
            print(self.bounds)
    def plot(self, plot_type = "contour", sensitivity_of = 'estimate', **kwargs):
        """
        Provide the contour and extreme scenario sensitivity plots of the sensitivity analysis results obtained with the function Sensemakr.

        They are basically dispatchers
        to the core plot functions ovb_contour_plot and ovb_extreme_plot.

        This function takes as input a sensemakr object and one of the plot type "contour" or "extreme". Optional arguments
        can be found in sensitivity_plots documentation including col_contour, col_thr_line etc.

        Parameters
        ----------
        plot_type : string
            Either "extreme" or "contour" (Default value = "contour").
        sensitivity_of : string
            Either "estimate" or "t-value" (Default value = 'estimate').
        **kwargs :
            Arbitrary keyword arguments. See ovb_contour_plot and ovb_extreme_plot.

        Returns
        -------


        Examples
        ---------
        >>> # Load example dataset:
        >>> from sensemakr import data
        >>> darfur = data.load_darfur()
        >>> # Fit a statsmodels OLSResults object ("fitted_model"):
        >>> import statsmodels.formula.api as smf
        >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
        >>> fitted_model = model.fit()
        >>> # Runs sensemakr for sensitivity analysis
        >>> from sensemakr import main
        >>> sensitivity = main.Sensemakr(fitted_model, treatment = "directlyharmed", benchmark_covariates = "female", kd = [1, 2, 3])
        """
        if plot_type == 'contour':
            sensitivity_plots.ovb_contour_plot(sense_obj=self,sensitivity_of=sensitivity_of,**kwargs)
        elif (plot_type == 'extreme') and (sensitivity_of == 't-value'):
            sys.exit('Error: extreme plot for t-value has not been implemented yet')
        elif plot_type == 'extreme':
            sensitivity_plots.ovb_extreme_plot(sense_obj=self,**kwargs)
        else:
            sys.exit('Error: "plot_type" argument must be "contour" or "extreme"')


    def print(self, digits=3):
        """Print a short summary of the sensitivity results for a Sensemakr object, including formula, hypothesis, and sensitivity analysis.

        digits is the number of digits to round numbers to; default is 3.
        Following the example above, if you have a Sensemakr object called sensitivity, call this method using
        sensitivity.print(). To round to 5 digits instead of 3, call sensitivity.print(digits=5).

        Parameters
        ----------
        digits : int
             Rounding digits for print (Default value = 3).

        Returns
        -------

        """
        if self.reduce:
            h0 = round(self.estimate * (1 - self.q), digits)
            direction = "reduce"
        else:
            h0 = round(self.estimate * (1 + self.q), digits)
            direction = "increase"

        print("Sensitivity Analysis to Unobserved Confounding\n")
        if self.model is not None:
            #model_formula = self.model.model.endog_names + ' ~ ' + ' + '.join(self.model.model.exog_names)
            #print("Model Formula: " + model_formula + "\n")
            print("Model Formula: "+self.model.model.formula+"\n")
        print("Null hypothesis: q =", self.q, "and reduce =", self.reduce, "\n")

        print("Unadjusted Estimates of '", self.treatment, "':")
        print("  Coef. estimate:", round(self.estimate, digits))
        print("  Standard Error:", round(self.sensitivity_stats['se'], digits))
        print("  t-value:", round(self.sensitivity_stats['t_statistic'], digits), "\n")

        print("Sensitivity Statistics:")
        print("  Partial R2 of treatment with outcome:", round(self.sensitivity_stats['r2yd_x'], digits))
        print("  Robustness Value, q =", self.q, ":", round(self.sensitivity_stats['rv_q'], digits))
        print("  Robustness Value, q =", self.q, "alpha =", self.alpha, ":",
              round(self.sensitivity_stats['rv_qa'], digits), "\n")

    def ovb_minimal_reporting(self, format = 'html', digits = 3, display = True):
        """
        ovb_minimal_reporting returns the LaTeX/HTML code for a table summarizing the sensemakr object.

        This function takes as input a sensemakr object, the digit to round number, one of the format type "latex" or "html",
        and a boolean whether to display the output or not. The default is round 3 digits, 'html' format and display the table.

        Parameters
        ----------
        format : string
            Either "latex" or "html" (Default value = 'html').
        display : boolean
            Default is True, to display the table.
        digits : int
            Rounding digit for the table (Default value = 3).

        Returns
        -------
        string
            LaTex/HTML code for creating the table summarizing the sensemakr object.

        Examples
        ---------
        >>> # Load example dataset:
        >>> from sensemakr import data
        >>> darfur = data.load_darfur()
        >>> # Fit a statsmodels OLSResults object ("fitted_model"):
        >>> import statsmodels.formula.api as smf
        >>> model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village', data=darfur)
        >>> fitted_model = model.fit()
        >>> # Runs sensemakr for sensitivity analysis
        >>> from sensemakr import main
        >>> sensitivity = main.Sensemakr(model=fitted_model, treatment = "directlyharmed", q=1.0, alpha=0.05, reduce=True)
        >>> # Gets HTML code and table
        >>> result=sensitivity.ovb_minimal_reporting() # doctest: +SKIP
        >>> # Prints raw html code
        >>> print(result) # doctest: +SKIP
        """
        if(format=='latex'):
            result='\\begin{table}[!h] \n\\centering \n\\begin{tabular}{lrrrrrr} \n'+\
            "\\multicolumn{7}{c}{Outcome: \\textit{"+str(self.model.model.endog_names)+"}} \\\\\n"+\
            "\\hline \\hline \nTreatment: & Est. & S.E. & t-value & $R^2_{Y \\sim D |{\\bf X}}$"+\
            " & $RV_{q ="+str(self.q)+"}$"+ "& $RV_{q = "+str(self.q)+ ", \\alpha = "+str(self.alpha)+ "}$ " + " \\\\ \n"+ "\\hline \n"+\
            "\\textit{"+str(self.treatment)+ "} &"+str(round(self.sensitivity_stats['estimate'], digits))+ " & "+\
              str(round(self.sensitivity_stats['se'], digits))+ " & "+\
              str(round(self.sensitivity_stats['t_statistic'], digits))+" & "+\
              str(round(self.sensitivity_stats['r2yd_x']*100, digits-2))+ "\\% & "+\
              str(round(self.sensitivity_stats['rv_q']*100, digits-2))+ "\\% & "+\
              str(round(self.sensitivity_stats['rv_qa']*100, digits-2))+ "\\% \\\\ \n"+\
            "\\hline \n" + "df = "+str(self.sensitivity_stats['dof'])+ " & & "+"\\multicolumn{5}{r}{ "+( "}\n"+\
            "\\end{tabular}\n"+\
            "\\end{table}" if (self.bounds is None) else "\\small"+\
            "\\textit{Bound ("+str(self.bounds['bound_label'][0])+ ")}: "+\
            "$R^2_{Y\\sim Z| {\\bf X}, D}$ = "+\
            str(round(self.bounds['r2yz_dx'][0]*100, digits-2))+\
            "\\%, $R^2_{D\\sim Z| {\\bf X} }$ = "+\
            str(round(self.bounds['r2dz_x'][0]*100, digits-2))+\
            "\\%""}\\\\\n"+\
            "\\end{tabular}\n"+\
            "\\end{table}")

            if(display==True):
                from IPython.display import display_latex
                display_latex(result, raw=True)
            return result

        if(format=='html'):
            result="<table style='align:center'>\n"+"<thead>\n"+\
            "<tr>\n"+\
            '\t<th style="text-align:left;border-bottom: 1px solid transparent;border-top: 1px solid black"> </th>\n'+\
            '\t<th colspan = 6 style="text-align:center;border-bottom: 1px solid black;border-top: 1px solid black"> Outcome: '+\
            str(self.model.model.endog_names)+'</th>\n'+\
            "</tr>\n"+\
            "<tr>\n"+\
            '\t<th style="text-align:left;border-top: 1px solid black"> Treatment </th>\n'+\
            '\t<th style="text-align:right;border-top: 1px solid black"> Est. </th>\n'+\
            '\t<th style="text-align:right;border-top: 1px solid black"> S.E. </th>\n'+\
            '\t<th style="text-align:right;border-top: 1px solid black"> t-value </th>\n'+\
            '\t<th style="text-align:right;border-top: 1px solid black"> R<sup>2</sup><sub>Y~D|X</sub> </th>\n'+\
            '\t<th style="text-align:right;border-top: 1px solid black">  RV<sub>q = '+\
            str(self.q)+ '</sub> </th>\n'+\
            '\t<th style="text-align:right;border-top: 1px solid black"> RV<sub>q = '+\
            str(self.q)+ ", &alpha; = "+\
            str(self.alpha)+ "</sub> </th>\n"+\
            "</tr>\n"+\
            "</thead>\n"+\
            "<tbody>\n <tr>\n"+\
            '\t<td style="text-align:left; border-bottom: 1px solid black"><i>'+\
            str(self.treatment)+ "</i></td>\n"+\
            '\t<td style="text-align:right;border-bottom: 1px solid black">'+\
            str(round(self.sensitivity_stats['estimate'], digits))+ " </td>\n"+\
            '\t<td style="text-align:right;border-bottom: 1px solid black">'+\
            str(round(self.sensitivity_stats['se'], digits))+ " </td>\n"+\
            '\t<td style="text-align:right;border-bottom: 1px solid black">'+\
            str(round(self.sensitivity_stats['t_statistic'], digits-2))+" </td>\n"+\
            '\t<td style="text-align:right;border-bottom: 1px solid black">'+\
            str(round(self.sensitivity_stats['r2yd_x']*100, digits-2))+ "% </td>\n"+\
            '\t<td style="text-align:right;border-bottom: 1px solid black">'+\
            str(round(self.sensitivity_stats['rv_q']*100, digits-2))+ "% </td>\n"+\
            '\t<td style="text-align:right;border-bottom: 1px solid black">'+\
            str(round(self.sensitivity_stats['rv_qa']*100, digits-2))+"% </td>\n"+\
            "</tr>\n</tbody>\n"+\
            ('<tr>\n'+\
            "<td colspan = 7 style='text-align:right;border-top: 1px solid black;border-bottom: 1px solid transparent;font-size:11px'>"+\
            "Note: df = "+str(self.sensitivity_stats['dof'])+\
            "</td>\n"+\
            "</tr>\n"+\
            "</table>" if (self.bounds is None) else "<tr>\n"+\
            "<td colspan = 7 style='text-align:right;border-top: 1px solid black;border-bottom: 1px solid transparent;font-size:11px'>"+\
            "Note: df = "+ str(self.sensitivity_stats['dof'])+ "; "+\
            "Bound ( "+str(self.bounds['bound_label'][0])+ " ):  "+\
            "R<sup>2</sup><sub>Y~Z|X,D</sub> =  "+\
            str(round(self.bounds['r2yz_dx'][0]*100, digits-2))+\
            "%, R<sup>2</sup><sub>D~Z|X</sub> ="+\
            str(round(self.bounds['r2dz_x'][0]*100, digits-2))+\
            "%"+\
            "</td>\n"+\
            "</tr>\n"+\
            "</table>")

            if(display==True):
                from IPython.display import display_html
                display_html(result, raw=True)

            return result
