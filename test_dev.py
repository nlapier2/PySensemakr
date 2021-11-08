# # Imports
# from sensemakr import sensemakr
# from sensemakr import sensitivity_stats
# from sensemakr import bias_functions
# from sensemakr import ovb_bounds
# from sensemakr import ovb_plots
# import statsmodels.api as sm
# import statsmodels.formula.api as smf
# import numpy as np
# import pandas as pd

# %load_ext autoreload
# %autoreload 2

# loads data
# darfur = pd.read_csv("data/darfur.csv")
# darfur.head()

# #runs regression model
# reg_model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar + '\
#                     'pastvoted + hhsize_darfur + female + village', data=darfur)
# model = reg_model.fit()



# #Define parameters for sensemakr
# treatment = "directlyharmed"
# q = 1.0
# alpha = 0.25
# reduce = True
# benchmark_covariates=["female"]
# kd = [1, 2, 3]
# ky = kd


# #Create a sensemakr object and print summary of results
# s = sensemakr.Sensemakr(model, treatment, q=q, alpha=alpha, reduce=reduce, benchmark_covariates=benchmark_covariates, kd=kd)
# s.summary()

# #Make a contour plot for the estimate
# ovb_plots.ovb_contour_plot(sense_obj=s, sensitivity_of='estimate')
