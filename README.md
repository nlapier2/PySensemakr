# PySensemakr

![PyPI](https://img.shields.io/pypi/v/Pysensemakr)
[![CI](https://github.com/KennyZhang-17/PySensemakr/actions/workflows/ci.yml/badge.svg)](https://github.com/KennyZhang-17/PySensemakr/actions/workflows/ci.yml)
[![Codecov](https://img.shields.io/codecov/c/gh/KennyZhang-17/PySensemakr)](https://app.codecov.io/gh/KennyZhang-17/PySensemakr)
## Installation

```
pip install Pysensemakr
```

## Development Version

```
pip3 install git+https://github.com/KennyZhang-17/PySensemakr
```

## Example Usage


```python
# Imports
from sensemakr import data
from sensemakr import sensemakr
import statsmodels.formula.api as smf
```


```python
# loads data
darfur = data.load_darfur()
darfur.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>wouldvote</th>
      <th>peacefactor</th>
      <th>peace_formerenemies</th>
      <th>peace_jjindiv</th>
      <th>peace_jjtribes</th>
      <th>gos_soldier_execute</th>
      <th>directlyharmed</th>
      <th>age</th>
      <th>farmer_dar</th>
      <th>herder_dar</th>
      <th>pastvoted</th>
      <th>hhsize_darfur</th>
      <th>village</th>
      <th>female</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1.000000</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>23</td>
      <td>Abdel Khair</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.706831</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>20</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>5</td>
      <td>Abdi Dar</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>45</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>15</td>
      <td>Abu Sorog</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.495178</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>55</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>Abu Dejaj</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.000000</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>7</td>
      <td>Abu Dejaj</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# runs regression model
reg_model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + '\
                    'pastvoted + hhsize_darfur + female + village', data=darfur)
model = reg_model.fit()
```


```python
# Define parameters for sensemakr
treatment = "directlyharmed"
q = 1.0
alpha = 0.05
reduce = True
benchmark_covariates=["female"]
kd = [1, 2, 3]
ky = kd
```


```python
# Create a sensemakr object and print summary of results
s = sensemakr.Sensemakr(model, treatment, q=q, 
                        alpha=alpha, reduce=reduce, benchmark_covariates=benchmark_covariates, kd=kd)
s.summary()
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
# Make a contour plot for the estimate
s.plot(plot_type='contour',sensitivity_of='estimate')
```


    
![png](/images/output_6_0.png)
    



```python
s.plot(plot_type='extreme',sensitivity_of='estimate')
```


    
![png](/images/output_7_0.png)
    

