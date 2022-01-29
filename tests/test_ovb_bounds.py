import sys
from scipy.stats import t
import numpy as np
import pandas as pd
from sensemakr.sensitivity_statistics import *
from sensemakr.bias_functions import *
from sensemakr.sensitivity_plots import *
from sensemakr.sensitivity_bounds import *
import statsmodels.formula.api as smf
from sensemakr import main
import pytest
import os

def resid_maker(n,df):
    N=np.random.normal(0,1,n)
    form='N~' +'+'.join(df.columns)
    df['N']=N
    model=smf.ols(formula=form,data=df).fit()
    e = model.resid
    e = (e-np.mean(e))/np.std(e)
    return(e)

path=os.path.join(os.path.dirname(__file__), '../data/darfur.csv')
darfur = pd.read_csv(path)

model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar +\
                pastvoted + hhsize_darfur + female + village', data=darfur).fit()
def test_bounds():
	out=ovb_bounds(model=model,treatment='directlyharmed',benchmark_covariates='female')

	data=["1x female",0.00916428667504862, 0.12464092303637, "directlyharmed",0.0752202712144491,0.0218733277437572,3.43890386024675,0.0322829657274445, 0.118157576701454]
	df = pd.DataFrame([data], columns = ["bound_label", "r2dz_x", "r2yz_dx", 'treatment',
	                                                 "adjusted_estimate", "adjusted_se", "adjusted_t",
	                                                 "adjusted_lower_CI", "adjusted_upper_CI"])

	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']] =\
	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']].astype(float)
	assert(out.round(6).equals(df.round(6)))

	out=ovb_bounds(model=model,treatment='directlyharmed',benchmark_covariates='female',alpha=0.2)
	data=["1x female", 0.00916428667504862, 0.12464092303637, "directlyharmed",0.0752202712144491,0.0218733277437572,  3.43890386024675,0.0471648038348768,  0.103275738594021]
	df = pd.DataFrame([data], columns = ["bound_label", "r2dz_x", "r2yz_dx", 'treatment',
	                                                 "adjusted_estimate", "adjusted_se", "adjusted_t",
	                                                 "adjusted_lower_CI", "adjusted_upper_CI"])

	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']] =\
	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']].astype(float)
	assert(out.round(6).equals(df.round(6)))

	out = ovb_bounds(model=model, treatment = "directlyharmed", benchmark_covariates = "female", alpha = 1)
	assert(out['adjusted_estimate'].values==out['adjusted_upper_CI'].values)
	assert(out['adjusted_estimate'].values==out['adjusted_lower_CI'].values)

	out=ovb_partial_r2_bound(model=model,treatment='directlyharmed',benchmark_covariates='female')
	np.testing.assert_allclose(out['r2dz_x'].values,0.0091642,atol=1e-6)

def test_partial_r2():
	b=ovb_partial_r2_bound(r2dxj_x=0.1,r2yxj_dx=0.1)
	ovb_contour_plot(model=model,treatment='directlyharmed')
	add_bound_to_contour(bound_label=b['bound_label'],r2dz_x=b['r2dz_x'],r2yz_dx=b['r2yz_dx'])
	assert(True)

def test_group_bench():
	# exact
	n=1000
	X=list(range(1,n+1))
	z1 = resid_maker(n,pd.DataFrame({'x':X}))
	z2 = resid_maker(n, pd.DataFrame({'Z1':z1}))
	x1 = resid_maker(n,pd.DataFrame({'Z1':z1, 'Z2':z2}))
	x2 = resid_maker(n,pd.DataFrame({'Z1':z1, 'Z2':z2,'X1':x1}))
	d = 2*x1+x2+2*z1+z2+resid_maker(n,pd.DataFrame({'Z1':z1, 'Z2':z2,'X1':x1, "X2":x2}))*5
	y = 2*x1+x2+2*z1+z2+resid_maker(n,pd.DataFrame({'Z1':z1, 'Z2':z2,'X1':x1, "X2":x2,'D':d}))*5
	df=pd.DataFrame({'z1':z1,'z2':z2,'x1':x1,'x2':x2,'d':d,'y':y})
	model=smf.ols(formula='y~d+x1+x2',data=df).fit()
	model_dz=smf.ols(formula='y~d+z1+z2',data=df).fit()
	r2yx=group_partial_r2(model=model,covariates=['x1','x2'])
	r2yz=group_partial_r2(model=model_dz,covariates=['z1','z2'])
	ky=r2yz/r2yx
	np.testing.assert_allclose(ky,1,atol=1e-7)
	model_d=smf.ols(formula='d~x1+x2',data=df).fit()
	model_dz=smf.ols(formula='d~z1+z2',data=df).fit()
	r2dx=group_partial_r2(model=model_d,covariates=['x1','x2'])
	r2dz=group_partial_r2(model=model_dz,covariates=['z1','z2'])
	kd=r2dz/r2dx
	np.testing.assert_allclose(kd,1,atol=1e-7)
	out    = main.Sensemakr(model = model, treatment = "d", benchmark_covariates = [['x1','x2']], kd =kd, ky= ky)
	out2   = main.Sensemakr(model = model, treatment = "d", benchmark_covariates = {'['+"'x1', 'x2'"+']':['x1','x2']}, kd =kd, ky= ky)
	out3   = main.Sensemakr(model = model, treatment = "d", benchmark_covariates = {'['+"'x1', 'x2'"+']':['x1','x2']}, kd =[1,2,3])
	bound  = ovb_partial_r2_bound(model=model,treatment="d")
	bound2 = ovb_partial_r2_bound(model=model,treatment="d",benchmark_covariates='x1')
	assert(out.bounds.equals(out2.bounds))


def test_bound_errors():
	with pytest.raises(SystemExit):
		ovb_bounds(model=model,treatment='directlyharmed',benchmark_covariates='female',alpha=0.2,bound='partial f2')
	with pytest.raises(SystemExit):
		ovb_partial_r2_bound(model=model,benchmark_covariates='female')
	with pytest.raises(SystemExit):
		ovb_partial_r2_bound(model=model,treatment=2,benchmark_covariates='female')
	with pytest.raises(SystemExit):
		ovb_partial_r2_bound(model=model,treatment='d',benchmark_covariates=np.array([1,2,3]))
	with pytest.raises(SystemExit):
		ovb_partial_r2_bound(model=model,treatment='d',benchmark_covariates=[1,'x1'])
	with pytest.raises(SystemExit):
		ovb_partial_r2_bound(model=model,treatment='d',benchmark_covariates={"X":[1,'x1']})
