"""
sensemakr for Python (PySensemakr) implements a suite of sensitivity
analysis tools that makes it easier to understand the impact of omitted variables in linear regression models,
as discussed in Cinelli and Hazlett (2020).

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

from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from sensemakr.main import Sensemakr
from sensemakr.sensitivity_plots import ovb_contour_plot, ovb_extreme_plot, add_bound_to_contour
from sensemakr.bias_functions import adjusted_estimate, adjusted_t, adjusted_se, adjusted_partial_r2, bias, relative_bias, rel_bias
from sensemakr.sensitivity_bounds import ovb_bounds, ovb_partial_r2_bound
from sensemakr.sensitivity_statistics import robustness_value, partial_r2, partial_f2, partial_f, group_partial_r2,sensitivity_stats
from sensemakr.data import load_darfur

__all__= ['Sensemakr','ovb_contour_plot', 'ovb_extreme_plot',
'add_bound_to_contour','adjusted_estimate', 'adjusted_t', 'adjusted_se',
'adjusted_partial_r2', 'bias', 'relative_bias', 'rel_bias', 'ovb_bounds', 'ovb_partial_r2_bound',
'robustness_value', 'partial_r2', 'partial_f2', 'partial_f', 'group_partial_r2','load_darfur', 'sensitivity_stats'
]
