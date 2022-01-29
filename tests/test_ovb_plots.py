import sys
from scipy.stats import t
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sensemakr.sensitivity_statistics import *
from sensemakr.bias_functions import *
from sensemakr.sensitivity_plots import *
from sensemakr.sensitivity_bounds import *
import statsmodels.formula.api as smf
from sensemakr import main
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
s = main.Sensemakr(model, treatment, q=q,
                        alpha=alpha, reduce=reduce, benchmark_covariates=benchmark_covariates, kd=kd)
s2 = main.Sensemakr(model, treatment, q=q,
                        alpha=alpha, reduce=False, benchmark_covariates=benchmark_covariates, kd=kd)
def test_plots():
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1)
	ovb_contour_plot(model=model,treatment='directlyharmed')
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,benchmark_covariates='female')
	ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',reduce=False)
	ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',lim=0.2,lim_y=0.3)
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,benchmark_covariates='female',kd=[1,2,3])
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=[0.1,0.2],benchmark_covariates='female',kd=[1,2,3])
	ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',kd=[1,2,3])
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,lim=1.5,lim_y=1.2)
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,lim=-0.5,lim_y=-0.2)
	ovb_extreme_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,lim=-0.5)
	ovb_extreme_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,lim=1.2)
	ovb_extreme_plot(model=model,treatment='directlyharmed',r2dz_x=0.1,lim=1.2,lim_y=0.5)
	plt.close('all')
	assert(True)
def test_plot_errors():
	with pytest.raises(SystemExit):
		ovb_contour_plot(model=model,sensitivity_of='p-value')
	with pytest.raises(SystemExit):
		ovb_contour_plot(estimate=2,se=10)
	with pytest.raises(SystemExit):
		ovb_extreme_plot(estimate=2,se=10)
	with pytest.raises(SystemExit):
		ovb_contour_plot(model=model,treatment=['directlyharmed','female'])
	with pytest.raises(SystemExit):
		ovb_contour_plot(estimate='none',se=3,dof=100)
	with pytest.raises(SystemExit):
		ovb_contour_plot(estimate=2,se=3,dof=100,ky=[1,2,3],kd=[2,3])
	with pytest.raises(SystemExit):
		ovb_contour_plot(estimate=2,se=3,dof=100,ky='none')
	with pytest.raises(SystemExit):
		ovb_contour_plot(estimate=2,se=3,dof=100,ky=[1,2,3],kd=[2,'none'])
	with pytest.raises(SystemExit):
		ovb_contour_plot(estimate=2,se=3,dof=100,ky=2,kd=[2,3])
	with pytest.raises(SystemExit):
		ovb_contour_plot(estimate=None,se=3,dof=100)
	with pytest.raises(SystemExit):
		add_bound_to_contour(benchmark_covariates='female',r2dz_x=0.2)
	plt.close('all')
