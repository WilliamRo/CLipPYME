# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 6, 2016
#   Author: William Ro
#
########################################################################

from .. import cl

import numpy as np
import scipy.optimize as optimize

from PYME.Analysis.FitFactories import FFBase
from PYME.Analysis.FitFactories.LatGaussFitFR import GaussianFitResultR

# region : Format of Results

result_data_type = [('tIndex', '<i4'),
                    ('fitResults', [('A', '<f4'),
                                    ('x0', '<f4'), ('y0', '<f4'),
                                    ('sigma', '<f4'),
                                    ('background', '<f4'),
                                    ('bx', '<f4'),
                                    ('by', '<f4')]),
                    ('fitError', [('A', '<f4'),
                                  ('x0', '<f4'),
                                  ('y0', '<f4'),
                                  ('sigma', '<f4'),
                                  ('background', '<f4'),
                                  ('bx', '<f4'),
                                  ('by', '<f4')]),
                    ('resultCode', '<i4'),
                    ('slicesUsed', [('x', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                                    ('y', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')]),
                                    ('z', [('start', '<i4'), ('stop', '<i4'), ('step', '<i4')])]),
                    ('subtractedBackground', '<f4')
                    ]


# endregion : Format of Results

class GaussianFitFactory(FFBase.FitFactory):
    """GaussianFitFactory inherits from FFBase.FitFactory to use its
    getROIAtPoint method"""

    # region : Constructor

    def __init__(self, data, metadata,
                 background=None, noise_sigma=None):
        # call to constructor of super class
        super(GaussianFitFactory, self).__init__(
            data, metadata, f_gauss2d, background, noise_sigma)

    # endregion : Constructor

    # region : Core Method

    def FromPoint(self, x, y, z=None,
                  roi_half_size=5, axial_half_size=15):
        # > get ROI [3.0%]
        # --------------------------------------------------------------
        X, Y, data, background, sigma, xslice, yslice, zslice = \
            self.getROIAtPoint(x, y, z, roi_half_size, axial_half_size)

        data_mean = data - background

        # > estimate some start parameters
        # --------------------------------------------------------------
        A = data.max() - data.min()  # amplitude

        x0 = 1e3 * self.metadata.voxelsize.x * x
        y0 = 1e3 * self.metadata.voxelsize.y * y

        bg_mean = np.mean(background)

        start_parameters = [A, x0, y0, 250 / 2.35,
                            data_mean.min(), .001, .001]

        # > do the fit
        # --------------------------------------------------------------
        (res, cov_x, info_dict, msg, res_code) = fit_model_weighted(
            self.fitfcn, start_parameters, data_mean, sigma, X, Y)

        # > try to estimate errors based on the covariance matrix
        # --------------------------------------------------------------
        fit_errors = None
        try:
            fit_errors = np.sqrt(
                np.diag(cov_x) * (info_dict['fvec'] * info_dict['fvec']).sum() /
                (len(data_mean.ravel()) - len(res)))
        except Exception:
            print('!!! Failed to estimate errors based on the covariance matrix')

        # > package results
        # --------------------------------------------------------------
        return GaussianFitResultR(res, self.metadata, (xslice, yslice, zslice),
                                  res_code, fit_errors, bg_mean)

    # endregion : Core Method

    pass


# region : Model Functions


def f_gauss2d(p, X, Y):
    """2D Gaussian model function with linear background
     - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    return A * np.exp(-((X - x0) ** 2 + (Y - y0) ** 2) /
                      (2 * s ** 2)) + b + b_x * X + b_y * Y


# endregion : Model Functions

# region : Solver

def fit_model_weighted(model_fcn, start_parameters,
                       data, sigmas, *args):
    return optimize.leastsq(weighted_miss_fit, start_parameters,
                            (model_fcn, data.ravel(), (1.0 / sigmas).
                             astype('f').ravel()) + args, full_output=1)


def weighted_miss_fit(p, fcn, data, weights, *args):
    """Helper function which evaluates a model function (fcn) with
    parameters (p) and additional arguments(*args) and compares
    this with measured data (data), scaling with precomputed weights
    corresponding to the errors in the measured data (weights)"""
    mod = fcn(p, *args)
    mod = mod.ravel()
    return (data - mod) * weights

# endregion : Solver

# region : Uniform Interface

FitFactory = GaussianFitFactory

# endregion : Uniform Interface
