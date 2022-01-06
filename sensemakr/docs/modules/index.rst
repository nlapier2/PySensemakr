Modules
=========


Main class (Sensemakr)

The main class of the sensemakr package is Sensemakr. These functions will likely suffice for most users, for most of the time. The main workflow consists of fitting a linear model with lm() and running the sensitivity analysis with sensemakr(). This function returns an object of class sensemakr with all main sensitivity results, which one can then explore the results with the print, plot and summary methods.

.. toctree::
   :maxdepth: 1
   Sensemakr

Sensitivity plots
These functions provide direct access to sensitivity contour plots and extreme sensitivity plots for customization.
.. toctree::
   :maxdepth: 1
   sensitivity_plots
   sensitivity_bounds
   sensitivity_stats
   bias_functions
   data
