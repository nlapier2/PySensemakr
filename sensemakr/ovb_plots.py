# Code for producing sensitivity contour plots and other plots
import matplotlib.pyplot as plt
from . import sensitivity_stats
from . import ovb_bounds
from . import bias_functions
import sys
import numpy as np
from scipy.stats import t
import pandas as pd


plot_env = {'lim': 0.4, 'lim_y': 0.4, 'reduce': None, 'sensitivity_of': None, 'treatment': None}


def plot(sense_obj, plot_type):
    if plot_type == 'contour':
        ovb_contour_plot(sense_obj=sense_obj)
    elif plot_type == 'extreme':
        ovb_extreme_plot(sense_obj=sense_obj)
    else:
        sys.exit('Error: "plot_type" argument must be "contour" or "extreme"')


def ovb_contour_plot(sense_obj=None, sensitivity_of=None, model=None, treatment=None, estimate=None, se=None, dof=None,
                     benchmark_covariates=None, kd=1, ky=None, r2dz_x=None, r2yz_dx=None, bound_label=None,
                     reduce=True, estimate_threshold=0, t_threshold=2, lim=None, lim_y=None,
                     col_contour="black", col_thr_line="red", label_text=True, label_bump_x=None, label_bump_y=None,
                     xlab=None, ylab=None, asp=None, list_par=None, round_dig=3):
    if sensitivity_of not in ["estimate", "t-value"]:
        sys.exit('Error: "sensitivity_of" argument is required and must be "estimate" or "t-value".')
    if sense_obj is not None:
        # treatment, estimate, se, dof, r2dz_x, r2yz_dx, bound_label, reduce, thr, t_thr
        treatment, estimate, se, dof, r2dz_x, r2yz_dx, bound_label, reduce, estimate_threshold, t_threshold = \
            extract_from_sense_obj(sense_obj)
    elif model is not None and treatment is not None:
        estimate, se, dof, r2dz_x, r2yz_dx = extract_from_model(
            model, treatment, benchmark_covariates, kd, ky, r2dz_x, r2yz_dx)
    elif estimate is None or se is None or dof is None:
        sys.exit('Error: must provide a Sensemakr object, a statsmodels OLSResults object and treatment, or'
                 'an estimate, standard error, and degrees of freedom.')
    estimate, r2dz_x, r2yz_dx, lim, lim_y, label_bump_x, label_bump_y, asp, list_par = check_params(
        estimate, r2dz_x, r2yz_dx, lim, lim_y, label_bump_x, label_bump_y, asp, list_par)
    plot_env['lim'] = lim
    plot_env['lim_y'] = lim_y
    plot_env['reduce'] = reduce
    plot_env['sensitivity_of'] = sensitivity_of
    plot_env['treatment'] = treatment

    grid_values_x = np.arange(0, lim, lim / 400)
    grid_values_y = np.arange(0, lim_y, lim_y / 400)
    bound_value = None
    if sensitivity_of == 'estimate':
        # call adjusted_estimate using grid values as r2dz_x and r2yz_dx as well as passed-in estimate, se, and dof
        z_axis = [[bias_functions.adjusted_estimate(grid_values_x[j], grid_values_y[i],
                   estimate=estimate, se=se, dof=dof)
                   for j in range(len(grid_values_x))] for i in range(len(grid_values_y))]
        threshold = estimate_threshold
        plot_estimate = estimate
        if r2dz_x is not None:
            bound_value = bias_functions.adjusted_estimate(r2dz_x, r2yz_dx, estimate=estimate, se=se, dof=dof,
                                                           reduce=reduce)
    else:  # sensitivity_of is 't-value'
        # call adjusted_estimate using grid values as r2dz_x and r2yz_dx as well as passed-in estimate, se, and dof
        z_axis = [[bias_functions.adjusted_t(grid_values_x[j], grid_values_y[i],
                   estimate=estimate, se=se, dof=dof, reduce=reduce, h0=estimate_threshold)
                   for j in range(len(grid_values_x))] for i in range(len(grid_values_y))]
        threshold = t_threshold
        plot_estimate = (estimate - estimate_threshold) / se
        if r2dz_x is not None:
            bound_value = bias_functions.adjusted_t(r2dz_x, r2yz_dx, estimate=estimate, se=se, dof=dof,
                                                    reduce=reduce, h0=estimate_threshold)

    # TODO: see which of these params we want to include in function args list
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    # draw all contours
    CS = ax.contour(grid_values_x, grid_values_y, z_axis,
                    colors=col_contour, linewidths=1.0, linestyles="solid")

    # remove contour line at threshold level
    threshold_index = CS.levels.tolist().index(threshold)
    CS.collections[threshold_index].remove()
    ax.clabel(CS, inline=1, fontsize=8, fmt="%1.3g", colors="gray", levels=np.delete(CS.levels, threshold_index))

    # draw red critical contour line
    CS = ax.contour(grid_values_x, grid_values_y, z_axis,
                    colors=col_thr_line, linewidths=1.0, linestyles=[(0, (7, 3))], levels=[threshold])
    ax.clabel(CS, inline=1, fontsize=8, fmt="%1.3g", colors="gray")

    # Plot point for unadjusted estimate / t_statistic
    ax.scatter([0], [0], c='k', marker='^')
    ax.annotate("Unadjusted\n({:1.3f})".format(plot_estimate), (0.0 + label_bump_x, 0.0 + label_bump_y))

    # Plot labeling and limit-setting
    if xlab is None:
        xlab = r"Partial $R^2$ of confounder(s) with the treatment"
    if ylab is None:
        ylab = r"Partial $R^2$ of confounder(s) with the outcome"
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.xlim(-(lim / 15.0), lim)
    plt.ylim(-(lim_y / 15.0), lim_y)

    # add bounds
    if r2dz_x is not None:
        r2dz_x, r2yz_dx = sensitivity_stats.check_r2(r2dz_x, r2yz_dx)
        add_bound_to_contour(r2dz_x=r2dz_x, r2yz_dx=r2yz_dx, bound_value=bound_value, bound_label=bound_label,
                             sensitivity_of=sensitivity_of, label_text=label_text, label_bump_x=label_bump_x,
                             label_bump_y=label_bump_y, round_dig=round_dig)


def add_bound_to_contour(model=None, benchmark_covariates=None, kd=1, ky=None, reduce=None,
                         treatment=None, bounds=None, r2dz_x=None, r2yz_dx=None, bound_value=None, bound_label=None,
                         sensitivity_of=None, label_text=True, label_bump_x=None, label_bump_y=None, round_dig=3):
    if (model is None or benchmark_covariates is None) and bounds is None and (r2dz_x is None or r2yz_dx is None):
        sys.exit('Error: add_bound_to_contour requires either a statsmodels OLSResults object and names of benchmark '
                 'covariates, or a Pandas DataFrame with bounding information, '
                 'or partial R^2 parameters r2dz_x and r2yz_dx.')
    if plot_env['reduce'] is None:
        sys.exit('Error: must have a current contour plot before adding bounds.')
    if treatment is None:
        treatment = plot_env['treatment']
    if sensitivity_of is None:
        sensitivity_of = plot_env['sensitivity_of']
    if label_bump_x is None:
        label_bump_x = plot_env['lim'] / 15.0
    if label_bump_y is None:
        label_bump_y = plot_env['lim_y'] / 15.0
    if reduce is None:
        reduce = plot_env['reduce']
    if ky is None:
        ky = kd

    if model is not None:
        if treatment != plot_env['treatment']:
            print('Warning: treatment variable provided does not equal treatment of previous contour plot.')

        bounds = ovb_bounds.ovb_bounds(model=model, treatment=treatment, benchmark_covariates=benchmark_covariates,
                                       kd=kd, ky=ky, adjusted_estimates=True, reduce=reduce)
        if sensitivity_of == 'estimate':
            bound_value = bounds['adjusted_estimate']
        else:
            bound_value = bounds['adjusted_t']
        if bound_label is not None:
            bound_label = bounds['bound_label']
    if bounds is not None:
        r2dz_x = bounds['r2dz_x']
        r2yz_dx = bounds['r2yz_dx']

    if type(r2dz_x) is int or type(r2dz_x) is float:
        r2dz_x = [r2dz_x]
    if type(r2yz_dx) is int or type(r2yz_dx) is float:
        r2yz_dx = [r2yz_dx]
    for i in range(len(r2dz_x)):
        plt.scatter(r2dz_x[i], r2yz_dx[i], c='red', marker='D', edgecolors='black')
        if label_text:
            if bound_value is not None and bound_value[i] is not None:
                bound_value[i] = round(bound_value[i], round_dig)
                label = str(bound_label[i]) + '\n(' + str(bound_value[i]) + ')'
            else:
                label = bound_label[i]
            plt.annotate(label, (r2dz_x[i] + label_bump_x, r2yz_dx[i] + label_bump_y))


def ovb_extreme_plot(sense_obj):  # not yet implemented
    return sense_obj


# Extracts sensitivity and bounding parameters from a given Sensemakr object
def extract_from_sense_obj(sense_obj):
    treatment = sense_obj.treatment
    estimate = sense_obj.estimate
    q = sense_obj.q
    reduce = sense_obj.reduce
    alpha = sense_obj.alpha
    se = sense_obj.se
    dof = sense_obj.dof

    if reduce:
        thr = estimate * (1 - q)
    else:
        thr = estimate * (1 + q)
    t_thr = abs(t.ppf(alpha / 2, dof - 1)) * np.sign(sense_obj.sensitivity_stats['t_statistic'])

    if sense_obj.bounds is None:
        r2dz_x = None
        r2yz_dx = None
        bound_label = ""
    else:
        r2dz_x = sense_obj.bounds['r2dz_x']
        r2yz_dx = sense_obj.bounds['r2yz_dx']
        bound_label = sense_obj.bounds['bound_label']
    return treatment, estimate, se, dof, r2dz_x, r2yz_dx, bound_label, reduce, thr, t_thr


# Extracts estimate, standard error, degrees of freedom, and parial R^2 values from a specified model+treatment pair
def extract_from_model(model, treatment, benchmark_covariates, kd, ky, r2dz_x, r2yz_dx):
    if ky is None:
        ky = kd
    check_multipliers(ky, kd)
    if type(treatment) is not str:
        sys.exit('Error: treatment must be a single string.')

    model_data = sensitivity_stats.model_helper(model, covariates=treatment)
    estimate = model_data['estimate']
    se = model_data['se']
    dof = model_data['dof']

    if benchmark_covariates is None:  # no benchmark bounding to do
        return estimate, se, dof, r2dz_x, r2yz_dx
    else:
        bench_bounds = ovb_bounds.ovb_bounds(
            model, treatment, benchmark_covariates=benchmark_covariates, kd=kd, ky=ky, adjusted_estimates=False)
        if r2dz_x is None:
            bounds = bench_bounds
        else:
            bounds = pd.DataFrame(data={'r2dz_x': r2dz_x, 'r2yz_dx': r2yz_dx})
            bounds.append(bench_bounds)
    return estimate, se, dof, bounds['r2dz_x'], bounds['r2yz_dx']


# Checks to make sure given parameters are valid and sets some default parameter values if not specified by the user
def check_params(estimate, r2dz_x, r2yz_dx, lim, lim_y, label_bump_x, label_bump_y, asp, list_par):
    check_estimate(estimate)
    if r2yz_dx is None:
        r2yz_dx = r2dz_x
    r2dz_x, r2yz_dx = sensitivity_stats.check_r2(r2dz_x, r2yz_dx)

    if lim is None:
        lim = min(np.max(list(r2dz_x * 1.2) + [0.4]), 1 - 10 ** -12)
    if lim_y is None:
        lim_y = min(np.max(list(r2yz_dx * 1.2) + [0.4]), 1 - 10 ** -12)
    if asp is None:
        asp = lim / lim_y
    if label_bump_x is None:
        label_bump_x = lim / 15.0
    if label_bump_y is None:
        label_bump_y = lim_y / 15.0
    if lim > 1.0:
        lim = 1 - 10 ** -12
        print('Warning: Contour limit larger than 1 was set to 1.')
    elif lim < 0:
        lim = 0.4
        print('Warning: Contour limit less than 0 was set to 0.4.')
    if lim_y > 1.0:
        lim_y = 1 - 10 ** -12
        print('Warning: Contour limit larger than 1 was set to 1.')
    elif lim_y < 0:
        lim_y = 0.4
        print('Warning: Contour limit less than 0 was set to 0.4.')
    if list_par is None:
        list_par = {'mar': [4, 4, 1, 1]}
    return estimate, r2dz_x, r2yz_dx, lim, lim_y, label_bump_x, label_bump_y, asp, list_par


# Parameter validators
def check_estimate(estimate):
    """ Make sure that the estimate is a single floating point number. """
    if estimate is None:
        sys.exit('Error: You must supply either a `model` and `treatment_covariate` to '
                 'extract an estimate or a directly supplied `estimate` argument')
    if type(estimate) is not float:
        sys.exit('Error: The estimated effect must be numeric.')


def check_multipliers(ky, kd):
    """ Make sure ky and kd are both numbers or equal-length lists of numbers. """
    if type(ky) not in [int, float, list] or type(kd) not in [int, float, list]:
        sys.exit('Error: ky and kd must be numeric.')
    if ((type(ky) is int or type(ky) is float) and type(kd) is list) or (
            (type(kd) is int or type(kd) is float) and type(ky) is list):
        sys.exit('Error: ky and kd must be both floats or ints or lists of the same length.')
    if type(ky) is list and type(kd) is list:
        if len(ky) != len(kd):
            sys.exit('Error: ky and kd must be the same length.')
        if not all(type(i) in [int, float] for i in ky) or not all(type(i) in [int, float] for i in kd):
            sys.exit('Error: ky and kd must contain only ints or floats.')
