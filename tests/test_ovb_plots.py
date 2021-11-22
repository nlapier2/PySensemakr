import sys
from scipy.stats import t
import numpy as np
import pandas as pd
from sensemakr.sensitivity_stats import *
from sensemakr.bias_functions import *
from sensemakr.ovb_plots import *
from sensemakr.ovb_bounds import *
import statsmodels.formula.api as smf
from sensemakr import sensemakr
import pytest
import os

path=os.path.join(os.path.dirname(__file__), '../data/darfur.csv')
darfur = pd.read_csv(path)

model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar +\
                pastvoted + hhsize_darfur + female + village', data=darfur).fit()
treatment = "directlyharmed"
q = 1.0
alpha = 0.05
reduce = True
benchmark_covariates=["female"]
kd = [1, 2, 3]
ky = kd
s = sensemakr.Sensemakr(model, treatment, q=q, 
                        alpha=alpha, reduce=reduce, benchmark_covariates=benchmark_covariates, kd=kd)
s2 = sensemakr.Sensemakr(model, treatment, q=q, 
                        alpha=alpha, reduce=False, benchmark_covariates=benchmark_covariates, kd=kd)
def test_plots():
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1)
	ovb_contour_plot(model=model,treatment='directlyharmed',list_par=None)
	plot(s2,'extreme')
	#ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,benchmark_covariates='female')
	ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',reduce=False)
	ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',lim=0.2,lim_y=0.3)
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,benchmark_covariates='female',kd=[1,2,3])
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=[0.1,0.2],benchmark_covariates='female',kd=[1,2,3])
	ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',kd=[1,2,3])
	assert(True)