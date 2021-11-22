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
def test_plots():
	ovb_contour_plot(model=model,treatment='directlyharmed',r2dz_x=0.1)
	ovb_contour_plot(model=model,treatment='directlyharmed',list_par=None)
	
	assert(True)