import sys
from scipy.stats import t
import numpy as np
import pandas as pd
from sensemakr.bias_functions import *
from sensemakr.sensitivity_statistics import *
import statsmodels.formula.api as smf
from sensemakr import main

# Numerical Tests
def test_bias():
    assert (bias(se = 3.0, dof = 100, r2dz_x = 0.3, r2yz_dx = 0.4) == 3*np.sqrt(100)*np.sqrt(0.4*0.3/(1 - 0.3)))
def test_adjusted_estimate():
    assert(adjusted_estimate(estimate=2,se=3,dof=100,r2dz_x=0.3,r2yz_dx=0.4)==2-3*np.sqrt(100)*np.sqrt(0.4*0.3/(1-0.3)))
    assert(adjusted_estimate(estimate=2,se=3,dof=100,r2dz_x=0.3,r2yz_dx=0.4,reduce=False)==2+3*np.sqrt(100)*np.sqrt(0.4*0.3/(1-0.3)))

def test_adjusted_se():
    assert(adjusted_se(se = 3, dof = 100, r2dz_x = 0.3, r2yz_dx = 0.4)==3*np.sqrt(100/99)*np.sqrt((1 - 0.4)/(1 - 0.3)))

def test_adjusted_t():
    assert(adjusted_t(estimate = 2, se = 3, dof = 100, r2dz_x = 0.3, r2yz_dx = 0.4)==(2 - 3*np.sqrt(100)*np.sqrt(0.4*0.3/(1 - 0.3)))/(3*np.sqrt(100/99)*np.sqrt((1 - 0.4)/(1 - 0.3))))
    assert(adjusted_t(estimate = 2, se = 3, dof = 100, r2dz_x = 0.3, r2yz_dx = 0.4,reduce=False)==(2 + 3*np.sqrt(100)*np.sqrt(0.4*0.3/(1 - 0.3)))/(3*np.sqrt(100/99)*np.sqrt((1 - 0.4)/(1 - 0.3))))
    np.testing.assert_allclose(adjusted_t(estimate = 0.097, se = 0.0233, dof = 783, r2dz_x = 0, r2yz_dx = 0,h0=2*0.097),-4.160431,atol=1e-4)
def test_bf():
    assert(bf(0.1,0.3)==np.sqrt(0.3*0.1/(1-0.1)))
def test_adjusted_partial_r2():
	np.testing.assert_allclose(adjusted_partial_r2(0.2, 0.2,estimate=2,se=3,dof=100),0.02403814,atol=1e-6)
# Simulated Tests
def rcoef():return(np.random.uniform(-1,1,1))

def test_simulation():
	n=1e2
	z=rcoef()*np.random.normal(0,1,int(n))
	d=rcoef()*z + rcoef()*np.random.normal(0,1,int(n))
	y=rcoef()*d + rcoef()*z + rcoef()*np.random.normal(0,1,int(n))
	df=pd.DataFrame({'y':y,'d':d,'z':z})
	r_model = smf.ols(formula='y~d',data=df).fit()
	model = smf.ols(formula='y~d+z',data=df).fit()
	treat_model=smf.ols(formula='d~z',data=df).fit()
	r_coef=r_model.params['d']
	coef=model.params['d']
	true_bias=r_coef-coef
	true_rbias=rel_bias(r_coef,coef)


	# compute partial r2
	r2yz_dx = partial_r2(model, covariates = "z")
	r2dz_x  = partial_r2(treat_model, covariates = "z")
	trueBF  = np.sqrt(r2yz_dx * r2dz_x/(1 - r2dz_x))

	# compute implied biases
	BF = bf(r2dz_x = r2dz_x, r2yz_dx = r2yz_dx)
	assert(BF==trueBF)
	bias_m = bias(model = r_model, treatment = "d", r2dz_x = r2dz_x, r2yz_dx = r2yz_dx)
	np.testing.assert_allclose(bias_m,np.abs(true_bias),atol=1e-6)
	q = relative_bias(model = r_model, treatment = "d", r2dz_x = r2dz_x, r2yz_dx = r2yz_dx)
	np.testing.assert_allclose(q,np.abs(true_rbias))

def test_manual_input():
    out=main.Sensemakr(estimate=2,se=2,dof=100,r2dz_x=0.2,r2dxj_x=0.2,r2yxj_dx=0.2,kd=2,ky=1)
    out.summary()
    est= out.bounds['adjusted_estimate'][0]
    est_check = adjusted_estimate(estimate = 2, se=2, dof=100, r2dz_x=0.2, r2yz_dx=0.2)
    assert(est==est_check)
