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

path=os.path.join(os.path.dirname(__file__), '../data/darfur.csv')
darfur = pd.read_csv(path)

model = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar +\
                pastvoted + hhsize_darfur + female + village', data=darfur).fit()
darfur['peacefactor']=darfur['peacefactor']*-1
model2 = smf.ols(formula='peacefactor ~ directlyharmed + age + farmer_dar + herder_dar +\
                pastvoted + hhsize_darfur + female + village', data=darfur).fit()

def test_shape():
    assert(darfur.shape==(1276,14))
def test_columns():
    col_vec=(darfur.columns==["wouldvote",
                 "peacefactor",
                 "peace_formerenemies",
                 "peace_jjindiv",
                 "peace_jjtribes",
                 "gos_soldier_execute",
                 "directlyharmed",
                 "age",
                 "farmer_dar",
                 "herder_dar",
                 "pastvoted",
                 "hhsize_darfur",
                 "village",
                 "female"])
    assert(np.sum(col_vec)==14)
def test_darfur_Sensemakr():
	darfur_out=main.Sensemakr(model=model,treatment='directlyharmed',benchmark_covariates='female',kd=[1,2,3])
	darfur_out.summary()
	darfur_out.print()
	darfur_out.ovb_minimal_reporting(format='html',display=False)
	darfur_out.ovb_minimal_reporting(display=False)
	ovb_contour_plot(sense_obj=darfur_out)
	ovb_extreme_plot(sense_obj=darfur_out)
	# info
	assert(darfur_out.treatment=='directlyharmed')
	assert(darfur_out.q==1)
	assert(darfur_out.alpha==0.05)
	assert(darfur_out.reduce==True)
	assert(darfur_out.sensitivity_stats['dof']==783)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['r2yd_x'],0.02187,atol=1e-5)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['rv_q'],0.13878,atol=1e-5)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['rv_qa'],0.07626,atol=1e-5)
	data=[["1x female", "2x female", "3x female"],[0.00916428667504862, 0.0183285733500972, 0.0274928600251459],
	      [0.12464092303637, 0.249324064199975, 0.374050471038094],["directlyharmed","directlyharmed","directlyharmed"],
	      [0.0752202712144491, 0.0529151723844518, 0.0303960234641548],
	      [0.0218733277437572, 0.0203500620779637, 0.0186700648170924],
	      [3.43890386024675, 2.60024623913809, 1.62806202131271],[0.032282966, 0.012968035,-0.006253282],
	      [0.11815758, 0.09286231, 0.06704533]]
	data=np.array(data).T.tolist()
	df = pd.DataFrame(data, columns = ["bound_label", "r2dz_x", "r2yz_dx", "treatment",
	                                                 "adjusted_estimate", "adjusted_se", "adjusted_t",
	                                                 "adjusted_lower_CI", "adjusted_upper_CI"])

	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']] =\
	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']].astype(float)
	assert(darfur_out.bounds.round(6).equals(df.round(6)))

	darfur_out2=main.Sensemakr(model=model,treatment='directlyharmed')
	darfur_out3=main.Sensemakr(model=model, treatment='directlyharmed', q=1.0, alpha=0.05, reduce=True)
	darfur_out3.ovb_minimal_reporting(format='html',display=False)
	darfur_out3.ovb_minimal_reporting(display=False)
	ovb_contour_plot(sense_obj=darfur_out2)
	ovb_extreme_plot(sense_obj=darfur_out2)

def test_darfur_Sensemakr_negative():
	darfur_out=main.Sensemakr(model=model2,treatment='directlyharmed',benchmark_covariates='female',kd=[1,2,3])
	darfur_out.summary()
	ovb_contour_plot(sense_obj=darfur_out)
	ovb_extreme_plot(sense_obj=darfur_out)
	# info
	assert(darfur_out.treatment=='directlyharmed')
	assert(darfur_out.q==1)
	assert(darfur_out.alpha==0.05)
	assert(darfur_out.reduce==True)
	assert(darfur_out.sensitivity_stats['dof']==783)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['r2yd_x'],0.02187,atol=1e-5)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['rv_q'],0.13878,atol=1e-5)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['rv_qa'],0.07626,atol=1e-5)
	data=[["1x female", "2x female", "3x female"],[0.00916428667504862, 0.0183285733500972, 0.0274928600251459],
	      [0.12464092303637, 0.249324064199975, 0.374050471038094],["directlyharmed","directlyharmed","directlyharmed"],
	      [-0.0752202712144491, -0.0529151723844518, -0.0303960234641548],
	      [0.0218733277437572, 0.0203500620779637, 0.0186700648170924],
	      [-3.43890386024675, -2.60024623913809, -1.62806202131271],[-0.11815758, -0.09286231, -0.06704533],[-0.032282966, -0.012968035,0.006253282]
	      ]
	data=np.array(data).T.tolist()
	df = pd.DataFrame(data, columns = ["bound_label", "r2dz_x", "r2yz_dx", "treatment",
	                                                 "adjusted_estimate", "adjusted_se", "adjusted_t",
	                                                 "adjusted_lower_CI", "adjusted_upper_CI"])

	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']] =\
	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']].astype(float)
	assert(darfur_out.bounds.round(6).equals(df.round(6)))

	darfur_out2=main.Sensemakr(model=model2,treatment='directlyharmed')
	ovb_contour_plot(sense_obj=darfur_out2)
	ovb_extreme_plot(sense_obj=darfur_out2)

def test_darfur_manual_bounds():
	sense_out=main.Sensemakr(model=model,treatment='directlyharmed',benchmark_covariates='female',r2dz_x=0.1)
	sense_out.summary()
	bounds_check=sense_out.bounds
	to_check=bounds_check.adjusted_se
	true_check=adjusted_se(model=model,treatment='directlyharmed',r2dz_x=0.1,r2yz_dx=0.1)
	assert(to_check.values[0]==true_check)

def test_darfur_sensemakr_manually():
	model_treat=smf.ols(formula='directlyharmed ~  age + farmer_dar + herder_dar +\
                pastvoted + hhsize_darfur + female + village', data=darfur).fit()
	darfur_out=main.Sensemakr(estimate = 0.09731582,
                                    se = 0.02325654,
                                    dof = 783,
                                    treatment = "directlyharmed",
                                    benchmark_covariates = "female",
                                    r2dxj_x = partial_r2(model_treat, covariates = "female"),
                                    r2yxj_dx = partial_r2(model, covariates = "female"),
                                    kd = [1,2,3])
	assert(darfur_out.q==1)
	assert(darfur_out.alpha==0.05)
	assert(darfur_out.treatment=="D")
	assert(darfur_out.reduce==True)
	assert(darfur_out.sensitivity_stats['dof']==783)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['r2yd_x'],0.02187,atol=1e-4)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['rv_q'],0.13878,atol=1e-5)
	np.testing.assert_allclose(darfur_out.sensitivity_stats['rv_qa'],0.07626,atol=1e-5)
	data=[["1x female", "2x female", "3x female"],
                  [0.00916428667504862, 0.0183285733500972, 0.0274928600251459],
                  [0.12464092303637, 0.249324064199975, 0.374050471038094],
                  [0.0752202698486415, 0.0529151689180575,0.0303960178770157],
                  [0.0218733298036818, 0.0203500639944344,0.0186700665753491],
                  [3.43890347394571, 2.60024582392121,1.62806156873318],
                  [0.0322829603180086,0.0129680276030601, -0.00625329133645187],
                  [0.118157579379274,0.092862310233055, 0.0670453270904833]]
	data=np.array(data).T.tolist()
	df = pd.DataFrame(data, columns = ["bound_label", "r2dz_x", "r2yz_dx",
	                                                 "adjusted_estimate", "adjusted_se", "adjusted_t",
	                                                 "adjusted_lower_CI", "adjusted_upper_CI"])

	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']] =\
	df[['r2dz_x','r2yz_dx','adjusted_estimate','adjusted_se','adjusted_t','adjusted_lower_CI','adjusted_upper_CI']].astype(float)
	assert(darfur_out.bounds.round(6).equals(df.round(6)))
def test_darfur_sensitivity_stats():
	rv=robustness_value(model=model,covariates='directlyharmed')
	np.testing.assert_allclose(rv.values,0.138776,atol=1e-5)
	assert(rv.index.values=='directlyharmed')
	rv=robustness_value(model=model,covariates='directlyharmed',q=1,alpha=0.05)
	np.testing.assert_allclose(rv.values,0.07625797,atol=1e-5)
	assert(rv.index.values=='directlyharmed')

	r2=partial_r2(model=model,covariates='directlyharmed')
	np.testing.assert_allclose(r2,0.02187309,atol=1e-5)
	f2=partial_f2(model=model,covariates='directlyharmed')
	np.testing.assert_allclose(f2,0.02236222,atol=1e-5)
	sens_stats=sensitivity_statistics.sensitivity_stats(model=model,treatment='directlyharmed')
	np.testing.assert_allclose(sens_stats['dof'],783,atol=1e-5)
	np.testing.assert_allclose(sens_stats['estimate'],0.09731582,atol=1e-5)
	np.testing.assert_allclose(sens_stats['se'],0.02325654,atol=1e-5)
	np.testing.assert_allclose(sens_stats['t_statistic'],4.18445,atol=1e-5)
	np.testing.assert_allclose(sens_stats['r2yd_x'],0.02187309,atol=1e-5)
	np.testing.assert_allclose(sens_stats['rv_q'],0.1387764,atol=1e-5)
	np.testing.assert_allclose(sens_stats['rv_qa'],0.07625797,atol=1e-5)
	np.testing.assert_allclose(sens_stats['f2yd_x'],0.02236222,atol=1e-5)
	assert(group_partial_r2(model=model,covariates='directlyharmed')==partial_r2(model=model,covariates='directlyharmed'))
	np.testing.assert_allclose(group_partial_r2(model=model,covariates=['directlyharmed','female']),0.1350435,atol=1e-5)


def test_darfur_adjusted_estimates():
	should_be_zero = adjusted_estimate(model=model, treatment = "directlyharmed", r2yz_dx = 1, r2dz_x = partial_r2(model=model, covariates = "directlyharmed"))
	np.testing.assert_allclose(should_be_zero,0,atol=1e-8)
	rv = robustness_value(model=model, covariates = "directlyharmed")
	should_be_zero = adjusted_estimate(model=model, treatment = "directlyharmed", r2yz_dx = rv, r2dz_x = rv)
	np.testing.assert_allclose(should_be_zero,0,atol=1e-8)

	rv = robustness_value(model=model, covariates = "directlyharmed",alpha=0.05)
	should_be_1_96 = adjusted_t(model=model, treatment = "directlyharmed", r2yz_dx = rv, r2dz_x = rv)
	np.testing.assert_allclose(should_be_1_96,1.96,atol=1e-2)
	should_be_estimate=bias(model=model,treatment='directlyharmed',r2yz_dx=1,r2dz_x=partial_r2(model=model,covariates='directlyharmed'))
	np.testing.assert_allclose(should_be_estimate,model.params['directlyharmed'],atol=1e-8)
	rv=robustness_value(model=model,covariates='directlyharmed')
	should_be_estimate=bias(model=model,treatment='directlyharmed',r2yz_dx=rv,r2dz_x=rv)
	np.testing.assert_allclose(should_be_estimate,model.params['directlyharmed'],atol=1e-8)
	rv=robustness_value(model=model,covariates='directlyharmed',q=0.5)
	should_be_half_estimate=bias(model=model,treatment='directlyharmed',r2yz_dx=rv,r2dz_x=rv)
	np.testing.assert_allclose(should_be_half_estimate,0.5*model.params['directlyharmed'],atol=1e-8)

def test_darfur_plots():
	contour_out=ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',kd=[1,2,3])
	add_bound_to_contour(model=model,treatment='directlyharmed',benchmark_covariates='age',kd=10)
	add_bound_to_contour(model=model, treatment = "directlyharmed",benchmark_covariates = "age", kd = 200, ky = 20)

	## t-value
	contour_out=ovb_contour_plot(model=model,treatment='directlyharmed',benchmark_covariates='female',kd=[1,2,3],sensitivity_of='t-value')
	add_bound_to_contour(model=model,treatment='directlyharmed',benchmark_covariates='age',kd=200,ky=10,sensitivity_of='t-value')

	# test extreme scenario plot
	extreme_out =ovb_extreme_plot(model=model, treatment = "directlyharmed", kd = [1,2,3])

	assert(True)

def test_darfur_different_q():
	darfur_out=main.Sensemakr(model=model,treatment='directlyharmed',benchmark_covariates='female',q=2,kd=[1,2,3])
	rvq=darfur_out.sensitivity_stats['rv_q']
	rvqa=darfur_out.sensitivity_stats['rv_qa']
	assert(rvq==robustness_value(model=model,covariates='directlyharmed',q=2).values)
	assert(rvqa==robustness_value(model=model,covariates='directlyharmed',q=2,alpha=0.05).values)

def test_darfur_group_benchmarks():
	village=[k for k in model.params.index.values if 'village' in k]
	sensitivity=main.Sensemakr(model=model,treatment='directlyharmed',benchmark_covariates=[village],kd=0.3)
	r2y=group_partial_r2(model,covariates=village)
	treat_model=smf.ols(formula='directlyharmed ~  age + farmer_dar + herder_dar +\
                pastvoted + hhsize_darfur + female + village', data=darfur).fit()
	r2d=group_partial_r2(treat_model,covariates=village)
	bounds_check = ovb_partial_r2_bound(r2dxj_x = r2d, r2yxj_dx = r2y, kd = 0.3,benchmark_covariates=['manual'])
	bounds=sensitivity.bounds
	np.testing.assert_allclose(bounds_check['r2dz_x'].values,bounds['r2dz_x'].values,atol=1e-5)
	np.testing.assert_allclose(bounds_check['r2yz_dx'].values,bounds['r2yz_dx'].values,atol=1e-5)
