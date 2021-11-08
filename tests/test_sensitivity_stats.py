import sys
from scipy.stats import t
import numpy as np
import pandas as pd
from sensemakr.sensitivity_stats import *

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
# expect_equal(c(robustness_value(t = 2, dof = 10)), 0.5*(sqrt((2/sqrt(10))^4 + 4*((2/sqrt(10))^2)) - (2/sqrt(10))^2))
#
#   expect_equal(group_partial_r2(F.stats = 10, dof = 100, p = 4), 10*4/(10*4 + 100))
#
#   expect_equivalent(robustness_value(t_statistic = 1.89, dof = 1121), 0.055, tolerance = 1e-2)
#
#   expect_equivalent(robustness_value(t_statistic = 2.11, dof = 1115), 0.061, tolerance = 1e-2)
#
#   expect_equivalent(robustness_value(t_statistic = 37.5, dof = 983), 0.68, tolerance = 1e-2)
#
#   expect_equivalent(robustness_value(t_statistic = 17, dof = 983), 0.415, tolerance = 1e-2)
