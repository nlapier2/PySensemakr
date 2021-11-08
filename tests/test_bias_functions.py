import sys
from scipy.stats import t
import numpy as np
import pandas as pd
from sensemakr.bias_functions import *

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
