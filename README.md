# PySensemakr - sensemakr for Python

![PyPI](https://img.shields.io/pypi/v/Pysensemakr)
[![CI](https://github.com/nlapier2/PySensemakr/actions/workflows/ci.yml/badge.svg)](https://github.com/nlapier2/PySensemakr/actions/workflows/ci.yml)
[![Codecov](https://img.shields.io/codecov/c/gh/nlapier2/PySensemakr)](https://app.codecov.io/gh/nlapier2/PySensemakr)
[![Documentation Status](https://readthedocs.org/projects/pysensemakr/badge/?version=latest)](https://pysensemakr.readthedocs.io/en/latest/?badge=latest)

`sensemakr` for Python (`PySensemakr`) implements a suite of sensitivity analysis tools that
extends the traditional omitted variable bias framework and makes it
easier to understand the impact of omitted variables in regression
models, as discussed in [Cinelli, C. and Hazlett, C. (2020) “Making
Sense of Sensitivity: Extending Omitted Variable Bias.” Journal of the
Royal Statistical Society, Series B (Statistical
Methodology).](https://doi.org/10.1111/rssb.12348)

## Related Packages
-   The R version of the package can be downloaded here: <https://github.com/carloscinelli/sensemakr/>.

-   The Stata version of the package can be downloaded here: <https://github.com/resonance1/sensemakr-stata>.

-   The Shiny App is available at: <https://carloscinelli.shinyapps.io/robustness_value/>.

## Details

For theoretical details, [please see the JRSS-B
paper](https://www.researchgate.net/publication/322509816_Making_Sense_of_Sensitivity_Extending_Omitted_Variable_Bias).

For practical details of the package, see the the [package documentation](https://pysensemakr.readthedocs.io/en/latest/).

## Installation

Make sure you have Python 3.8, or higher, installed.

To install the latest development version on GitHub, run:

```
pip3 install git+https://github.com/nlapier2/PySensemakr
```

A user version on PyPI can be installed via:

```
pip3 install PySensemakr
```

## Example Usage


```python
# Imports
import sensemakr as smkr
import statsmodels.formula.api as smf
```


```python
# loads data
darfur = smkr.load_darfur()
```


```python
# runs regression model
reg_model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + '\
                    'pastvoted + hhsize_darfur + female + village', data=darfur)
darfur_model = reg_model.fit()
```

```python
# Create a sensemakr object and print summary of results
darfur_sense = smkr.Sensemakr(model = darfur_model,
                              treatment = "directlyharmed",
                              benchmark_covariates = ["female"],
                              kd = [1,2,3])
darfur_sense.summary()
```

    Sensitivity Analysis to Unobserved Confounding

    Model Formula: peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + pastvoted + hhsize_darfur + female + village

    Null hypothesis: q = 1.0 and reduce = True

    -- This means we are considering biases that reduce the absolute value of the current estimate.
    -- The null hypothesis deemed problematic is H0:tau = 0.0

    Unadjusted Estimates of ' directlyharmed ':
      Coef. estimate: 0.097
      Standard Error: 0.023
      t-value: 4.184

    Sensitivity Statistics:
      Partial R2 of treatment with outcome: 0.022
      Robustness Value, q = 1.0 : 0.139
      Robustness Value, q = 1.0 alpha = 0.05 : 0.076

    Verbal interpretation of sensitivity statistics:

    -- Partial R2 of the treatment with the outcome: an extreme confounder (orthogonal to the covariates)  that explains 100% of the residual variance of the outcome, would need to explain at least 2.187 % of the residual variance of the treatment to fully account for the observed estimated effect.

    -- Robustness Value, q = 1.0 : unobserved confounders (orthogonal to the covariates) that  explain more than 13.878 % of the residual variance of both the treatment and the outcome are strong enough to bring the point estimate to 0.0 (a bias of 100.0 % of the original estimate). Conversely, unobserved confounders that do not explain more than 13.878 % of the residual variance of both the treatment and the outcome are not strong enough to bring the point estimate to 0.0 .

    -- Robustness Value, q = 1.0 , alpha = 0.05 : unobserved confounders (orthogonal to the covariates) that explain more than 7.626 % of the residual variance of both the treatment and the outcome are strong enough to bring the estimate to a range where it is no longer 'statistically different' from 0.0 (a bias of 100.0 % of the original estimate), at the significance level of alpha = 0.05 . Conversely, unobserved confounders that do not explain more than 7.626 % of the residual variance of both the treatment and the outcome are not strong enough to bring the estimate to a range where it is no longer 'statistically different' from 0.0 , at the significance level of alpha = 0.05 .

    Bounds on omitted variable bias:
    --The table below shows the maximum strength of unobserved confounders with association with the treatment and the outcome bounded by a multiple of the observed explanatory power of the chosen benchmark covariate(s).

      bound_label    r2dz_x   r2yz_dx       treatment  adjusted_estimate  \
    0   1x female  0.009164  0.124641  directlyharmed           0.075220   
    1   2x female  0.018329  0.249324  directlyharmed           0.052915   
    2   3x female  0.027493  0.374050  directlyharmed           0.030396   

       adjusted_se  adjusted_t  adjusted_lower_CI  adjusted_upper_CI  
    0     0.021873    3.438904           0.032283           0.118158  
    1     0.020350    2.600246           0.012968           0.092862  
    2     0.018670    1.628062          -0.006253           0.067045  



```python
# contour plot for the estimate
darfur_sense.plot()
```



![png](https://github.com/nlapier2/PySensemakr/blob/main/images/output_6_0.png)




```python
# contour plot for the t-value
darfur_sense.plot(sensitivity_of = 't-value')
```



![png](https://github.com/nlapier2/PySensemakr/blob/main/images/output_22_0.png)




```python
# extreme scenarios plot
darfur_sense.plot(plot_type = 'extreme')
```



![png](https://github.com/nlapier2/PySensemakr/blob/main/images/output_7_0.png)
