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

from PYME.Analysis.FitFactories.LatGaussFitFR import GaussianFitResultR
from PYME.Analysis.FitFactories.LatGaussFitFR import f_gauss2d as fast_gauss

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

class GaussianFitFactory:
    """GaussianFitFactory
        Current version only support 2D fitting
    """

    # region : Constructor

    def __init__(self, data, metadata,
                 background=None, noise_sigma=None):
        # set fields
        self.data = data
        self.background = background
        self.metadata = metadata
        self.noise_sigma = noise_sigma
        self.fit_fcn = fast_gauss
        # self.fit_fcn = f_gauss2d

        # prepare noise_sigma and background
        if self.noise_sigma is None:
            sigma = np.sqrt(self.metadata.Camera.ReadNoise ** 2 +
                            (self.metadata.Camera.NoiseFactor ** 2) *
                            self.metadata.Camera.ElectronsPerCount *
                            self.metadata.Camera.TrueEMGain *
                            (np.maximum(data, 1) + 1)) / \
                    self.metadata.Camera.ElectronsPerCount

        if self.background is not None and len(np.shape(self.background)) > 1 \
                and not ('Analysis.subtractBackground' in self.metadata.getEntryNames()
                         and self.metadata.Analysis.subtractBackground is False):
            self.background = self.background.squeeze() - \
                         self.metadata.Camera.ADOffset
        else:
            self.background = 0

    # endregion : Constructor

    # region : Core Method

    def FromPoint(self, x, y, roi_half_size=5):
        # > get ROI [3.0%]
        # --------------------------------------------------------------
        x_r = round(x)
        y_r = round(y)

        xslice = slice(max((x_r - roi_half_size), 0),
                       min((x_r + roi_half_size + 1), self.data.shape[0]))
        yslice = slice(max((y_r - roi_half_size), 0),
                       min((y_r + roi_half_size + 1), self.data.shape[1]))

        data = self.data[xslice, yslice].squeeze() \
               - self.metadata.Camera.ADOffset

        X = 1e3 * self.metadata.voxelsize.x * np.mgrid[xslice]
        Y = 1e3 * self.metadata.voxelsize.y * np.mgrid[yslice]

        # estimate errors in data
        sigma = self.noise_sigma
        if not isinstance(sigma, float):
            sigma = self.noise_sigma[xslice, yslice]

        background = self.background
        if not isinstance(background, int):
            background = self.background[xslice, yslice]

        data_mean = data - background

        # > estimate some start parameters
        # --------------------------------------------------------------
        A = data.max() - data.min()  # amplitude

        x0 = 1e3 * self.metadata.voxelsize.x * x
        y0 = 1e3 * self.metadata.voxelsize.y * y

        start_parameters = [A, x0, y0, 250 / 2.35,
                            data_mean.min(), .001, .001]

        # > do the fit
        # --------------------------------------------------------------
        (res, cov_x, info_dict, msg, res_code) = fit_model_weighted(
            self.fit_fcn, start_parameters, data_mean, sigma, X, Y)

        # > try to estimate errors based on the covariance matrix
        # --------------------------------------------------------------
        fit_errors = None
        try:
            fit_errors = np.sqrt(
                np.diag(cov_x) * (info_dict['fvec'] * info_dict['fvec']).sum() /
                (len(data_mean.ravel()) - len(res)))
        except Exception:
            pass
            # print('!!! Failed to estimate errors based on the covariance matrix')

        # > package results
        # --------------------------------------------------------------
        bg_mean = 0
        return GaussianFitResultR(res, self.metadata,
                                  (xslice, yslice, slice(0, 1, None)),
                                  res_code, fit_errors, bg_mean)

    # endregion : Core Method

    pass


# region : Model Functions


def f_gauss2d(p, X, Y):
    """2D Gaussian model function with linear background
     - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    # delta_x = X[1] - X[0]
    # delta_y = Y[1] - Y[0]
    # for i in range(len(X)): X[i] += delta_x
    # for i in range(len(Y)): Y[i] += delta_y
    XV, YV = np.meshgrid(X, Y, indexing='ij')
    r = A * np.exp(-((XV - x0) ** 2 + (YV - y0) ** 2) /
                   (2 * s ** 2)) + b + \
        b_x * (XV - x0) + b_y * (YV - y0)
    return r


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
    res = (data - mod) * weights

    # mod_fast = fast_gauss(p, *args)
    # mod_fast = mod_fast.ravel()
    # res_fast = (data - mod_fast) * weights
    # delta = max(abs(res_fast - res))
    # if delta > 1e-12:
    #     pass

    return res


# endregion : Solver

# region : Uniform Interface

FitFactory = GaussianFitFactory
FitResultsDType = result_data_type

# endregion : Uniform Interface
