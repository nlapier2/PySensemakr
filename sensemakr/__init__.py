from ._version import get_versions
__version__ = get_versions()['version']
del get_versions

from sensemakr.main import Sensemakr
from sensemakr.sensitivity_plots import ovb_contour_plot, ovb_extreme_plot, add_bound_to_contour
from sensemakr.bias_functions import adjusted_estimate, adjusted_t, adjusted_se, adjusted_partial_r2, bias, relative_bias, rel_bias
from sensemakr.sensitivity_bounds import ovb_bounds, ovb_partial_r2_bound
from sensemakr.sensitivity_stats import robustness_value, partial_r2, partial_f2, partial_f, group_partial_r2
from sensemakr.data import load_darfur

__all__ = [
    "Sensemakr",
    "ovb_contour_plot", 
    "ovb_extreme_plot", 
    "add_bound_to_contour",
    "adjusted_estimate", 
    "adjusted_t", 
    "adjusted_se", 
    "adjusted_partial_r2", 
    "bias", 
    "relative_bias", 
    "rel_bias",
    "ovb_bounds", 
    "ovb_partial_r2_bound",
    "robustness_value", 
    "partial_r2", 
    "partial_f2", 
    "partial_f", 
    "group_partial_r2",
    "load_darfur"
]

# module level doc-string
__doc__ = """
sensemakr for Python (PySensemakr) implements a suite of sensitivity analysis tools that makes it easier 
to understand the impact of omitted variables in linear regression models, as discussed in Cinelli and Hazlett (2020).

The main class is Sensemakr. It does...


Users can also call other functions directly. 
"""