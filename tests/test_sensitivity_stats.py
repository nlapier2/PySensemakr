import sys
from scipy.stats import t
import numpy as np
import pandas as pd
from sensemakr.sensitivity_statistics import *
from sensemakr.bias_functions import *
import statsmodels.formula.api as smf
from sensemakr import main
import pytest
# Numerical Tests
def test_partial_r2():
    assert (partial_r2(t_statistic=2,dof=10)==2**2/(2**2+10))
    np.testing.assert_allclose(partial_r2(t_statistic=1.89,dof=1121),0.0032,atol=1e-4)
    np.testing.assert_allclose(partial_r2(t_statistic=2.11,dof=1114),0.004,atol=1e-4)
    np.testing.assert_allclose(partial_r2(t_statistic=37.5,dof=983),0.59,atol=1e-2)
def test_partial_f():
    assert (partial_f(t_statistic=2,dof=10)==2/np.sqrt(10))
def test_partial_f2():
    assert (partial_f2(t_statistic=2,dof=10)==2**2/10)
def test_group_partial_r2():
    assert(group_partial_r2(f_statistic=10,dof=100,p=4)==10*4/(10*4+100))
def test_robustness_value():
	assert(robustness_value(t_statistic=2,dof=10).values==0.5*(np.sqrt((2/np.sqrt(10))**4+4*((2/np.sqrt(10))**2))-(2/np.sqrt(10))**2))
	np.testing.assert_allclose(robustness_value(t_statistic=1.89,dof=1121),0.055,atol=1e-2)
	np.testing.assert_allclose(robustness_value(t_statistic=2.11,dof=1115),0.061,atol=1e-2)
	np.testing.assert_allclose(robustness_value(t_statistic=37.5,dof=983),0.68,atol=1e-2)
	np.testing.assert_allclose(robustness_value(t_statistic=17,dof=983),0.415,atol=1e-2)
def test_small_sample_rv():
	n=4
	x=np.random.normal(0,1,n)
	y=np.random.normal(0,1,n)
	df=pd.DataFrame({'y':y,'x':x})
	model = smf.ols(formula='y~x',data=df).fit()
	rv=robustness_value(model,alpha=0.05)['x']
	assert(rv>=0)

# Error tests
def test_errors():
	with pytest.raises(Exception):
		partial_f('text')
	with pytest.raises(Exception):
		partial_r2('text')
	with pytest.raises(Exception):
		robustness_value('text')
	with pytest.raises(Exception):
		bias('text')
	with pytest.raises(Exception):
		adjusted_estimate('text')
	with pytest.raises(Exception):
		adjusted_t('text')
	with pytest.raises(SystemExit):
		sensitivity_statistics.sensitivity_stats(estimate=2,se=-2)
	with pytest.raises(SystemExit):
		sensitivity_statistics.sensitivity_stats(estimate='hey',se=-2)
	with pytest.raises(SystemExit):
		sensitivity_statistics.sensitivity_stats(estimate=2,se='hey')
	with pytest.raises(SystemExit):
		sensitivity_statistics.sensitivity_stats(estimate=2,se=100,dof=-2)
	with pytest.raises(SystemExit):
		sensitivity_statistics.robustness_value(covariates='female',dof=10)
	with pytest.raises(SystemExit):
		sensitivity_statistics.partial_r2(covariates='female',dof=10)
	with pytest.raises(SystemExit):
		sensitivity_statistics.partial_f2(covariates='female',dof=10)
	with pytest.raises(SystemExit):
		sensitivity_statistics.group_partial_r2(covariates='female',dof=10)
	with pytest.raises(TypeError):
		adjusted_estimate(estimate=2,se=3,dof=100,reduce='nope')
	with pytest.raises(TypeError):
		adjusted_estimate(estimate=[2,3],se=3,dof=100)
	with pytest.raises(SystemExit):
		adjusted_t(estimate=[2,3],se=3,dof=100,r2dz_x=0.1,r2yz_dx=0.2)
	with pytest.raises(SystemExit):
		sensitivity_statistics.check_q('nope')
	with pytest.raises(SystemExit):
		sensitivity_statistics.check_alpha('nope')
	with pytest.raises(SystemExit):
		sensitivity_statistics.check_se(-1)
	with pytest.raises(SystemExit):
		sensitivity_statistics.check_covariates(['female','directlyharmed'],[1,2])
	with pytest.raises(SystemExit):
		sensitivity_statistics.check_covariates(['female','directlyharmed'],'village')
	with pytest.raises(TypeError):
		sensitivity_statistics.check_r2([0.1,0.2,0.3],['nope'])
	with pytest.raises(SystemExit):
		sensitivity_statistics.check_r2([0.1,0.2,0.3],[1,2,3])
