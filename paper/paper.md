---
title: '``PySensemakr``: Sensitivity Analysis Tools for Regression Models in Python'
tags:
  - Python
  - causal inference
  - regression
  - sensitivity analysis
authors:
  - name: Zhehao Zhang
    affiliation: 2
  - name: Nathan LaPierre
  	affiliation: 1
  - name: Brian Hill
  	affiliation: 1
  - name: Carlos Cinelli
  	affiliation: 2
affiliations:
 - name: University of California, Los Angeles
   index: 1
 - name: University of Washington
   index: 2
date: 29 Dec 2021
bibliography: paper.bib
---

# Summary

Regression has been widely applied in statistical modeling in social science, biology etc. While regression can often help making predictions or forecasting, it is also widely used to determine causal relationships between different variables, especially in treatment-control settings. @cinelli2020making proposes a suite of sensitivity analysis tools that extends the traditional omitted variable bias framework and makes it easier to understand the causal impact of omitted variables in regression models. We want to answer questions like how strong unobserved confounders need to be to overturn our research hypothesis and how robust are the results to all unobserved confounders acting together, possibly non-linearly. ``PySensemakr`` is a Python package to address these questions based on regression output without further model assumptions. `PySensemakr` is build upon the python package ``statsmodels``, which performs regression analysis. ``Pysensemakr`` reports some key causal quantities based on the routine regression output and provides visualization and causal interpretation of these quantities. We suggest using ``Pysensemakr`` for routine reporting sensitivity analysis on regression methods to assist research on causal relationships.

This package includes unit and integration tests made using the pytest framework. The repo containing the latest project code is integrated with continuous integration using Github Actions. Code coverage is monitored via codecov and is presently above 90%. The package website contains detailed description of methods in this package as well as a quick start guide and examples for users.



# Statement of Need

While regression is widely used in Python, to the best of the author’s knowledge, there is no sensitivity analysis tool in Python that builds upon the regression output, especially for causal interest. Most causal inference packages in Python rely on additional model assumptions and constraints and are complicated in nature. ``Pysensemakr`` provides an easy way to do sensitivity analysis based on regression output and provides causal interpretation, plots, tables for researchers to use without posing further assumptions on models. The quick start guide on package website is very friendly to users who has few causal background.


# Methods

```python
# Imports
from sensemakr import data
from sensemakr import sensemakr
import statsmodels.formula.api as smf
from statsmodels.iolib.summary2 import summary_col
```


```python
# loads data
darfur = data.load_darfur()
```


```python
# runs regression model
reg_model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + '\
            'herder_dar + pastvoted + hhsize_darfur + female + '\
            'village', data=darfur)
darfur_model = reg_model.fit()

# regression results for directlyharmed
summary_col(darfur_model, regressor_order=["directlyharmed"], drop_omitted=True)
```




<table class="simpletable">
<tr>
         <td></td>        <th>peacefactor</th>
</tr>
<tr>
  <th>directlyharmed</th>   <td>0.0973</td>   
</tr>
<tr>
  <th></th>                <td>(0.0233)</td>  
</tr>
<tr>
  <th>R-squared</th>        <td>0.5115</td>   
</tr>
<tr>
  <th>R-squared Adj.</th>   <td>0.2046</td>   
</tr>
</table>




```python
# Create a sensemakr object and print summary of results
darfur_sense = sensemakr.Sensemakr(model = darfur_model,
                                   treatment = "directlyharmed",
                                   benchmark_covariates = ["female"],
                                   kd = [1,2,3])

# minimal reporting table
html_code = darfur_sense.ovb_minimal_reporting(format = "html")
```


<table style='align:center'>
<thead>
<tr>
	<th style="text-align:left;border-bottom: 1px solid transparent;border-top: 1px solid black"> </th>
	<th colspan = 6 style="text-align:center;border-bottom: 1px solid black;border-top: 1px solid black"> Outcome: peacefactor</th>
</tr>
<tr>
	<th style="text-align:left;border-top: 1px solid black"> Treatment </th>
	<th style="text-align:right;border-top: 1px solid black"> Est. </th>
	<th style="text-align:right;border-top: 1px solid black"> S.E. </th>
	<th style="text-align:right;border-top: 1px solid black"> t-value </th>
	<th style="text-align:right;border-top: 1px solid black"> R<sup>2</sup><sub>Y~D|X</sub> </th>
	<th style="text-align:right;border-top: 1px solid black">  RV<sub>q = 1</sub> </th>
	<th style="text-align:right;border-top: 1px solid black"> RV<sub>q = 1, &alpha; = 0.05</sub> </th>
</tr>
</thead>
<tbody>
 <tr>
	<td style="text-align:left; border-bottom: 1px solid black"><i>directlyharmed</i></td>
	<td style="text-align:right;border-bottom: 1px solid black">0.097 </td>
	<td style="text-align:right;border-bottom: 1px solid black">0.023 </td>
	<td style="text-align:right;border-bottom: 1px solid black">4.2 </td>
	<td style="text-align:right;border-bottom: 1px solid black">2.2% </td>
	<td style="text-align:right;border-bottom: 1px solid black">13.9% </td>
	<td style="text-align:right;border-bottom: 1px solid black">7.6% </td>
</tr>
</tbody>
<tr>
<td colspan = 7 style='text-align:right;border-top: 1px solid black;border-bottom: 1px solid transparent;font-size:11px'>Note: df = 783; Bound ( 1x female ):  R<sup>2</sup><sub>Y~Z|X,D</sub> =  12.5%, R<sup>2</sup><sub>D~Z|X</sub> =0.9%</td>
</tr>
</table>



```python
# contour plot for the estimate
darfur_sense.plot(plot_type = 'contour',sensitivity_of = 'estimate')
```



![png](output_4_0.png)




```python
# contour plot for the t-value
darfur_sense.plot(plot_type='contour',sensitivity_of='t-value')
```



![png](output_5_0.png)




```python
# extreme scenarios plot
darfur_sense.plot(plot_type = 'extreme',sensitivity_of = 'estimate')
```



![png](output_6_0.png)




@cinelli2020making extends the traditional omitted variable bias (OVB) framework for sensitivity analysis using partial R2 representation. Based on this method, the package implements partial R2 of the treatment with the outcome, robustness value, bounds on the strength of confounding using observed covariates, multiple or non-linear confounders to quantify the strength of unobserved confounders that could potentially change the research conclusion. We provide example data of Darfur [@hazlett2020angry] to illustrate the usage of these methods and provide interpretation of each quantities in the package website. @cinelli2020sensemakr contains R and Stata version of the package and we use some implementation ideas from these packages.



# Acknowledgements




# References
