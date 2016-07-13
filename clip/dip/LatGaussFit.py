# -*- coding: utf-8 -*-

########################################################################
#
#   Created: June 6, 2016
#   Author: William Ro
#
########################################################################

import clip.cl as cl
from clip.op.leastsq import lmdif

import numpy as np
import scipy.optimize as optimize

from PYME.Analysis.FitFactories.LatGaussFitFR import GaussianFitResultR
from PYME.Analysis.FitFactories.LatGaussFitFR import \
    f_gauss2d as fast_gauss

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
                    ('slicesUsed', [('x',
                                     [('start', '<i4'), ('stop', '<i4'),
                                      ('step', '<i4')]),
                                    ('y',
                                     [('start', '<i4'), ('stop', '<i4'),
                                      ('step', '<i4')]),
                                    ('z',
                                     [('start', '<i4'), ('stop', '<i4'),
                                      ('step', '<i4')])]),
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

        self.data = data.squeeze()
        self.background = background
        self.noise_sigma = noise_sigma
        self.sigma_is_scalar = not len(np.shape(noise_sigma)) > 1
        self.metadata = metadata
        self.fit_fcn = fast_gauss

        ado = np.float32(self.metadata.Camera.ADOffset)
        am = cl.mem_access_mode
        hm = cl.mem_host_ptr_mode

        # region : Data pre-processing

        if ado != 0:
            self.data -= ado

        # endregion : Data pre-processing

        # region : Background pre-processing

        pp1 = self.background is not None
        pp2 = len(np.shape(background)) > 1
        pp3 = 'Analysis.subtractBackground' in \
              self.metadata.getEntryNames() and \
              self.metadata.Analysis.subtractBackground is False

        if pp1 and pp2 and not pp3:
            self.background = self.background.squeeze() - ado
        else:
            self.background = 0

        # endregion : Background pre-processing

        # region : Noise sigma pre-processing

        if self.noise_sigma is None:
            self.noise_sigma = \
                np.sqrt(self.metadata.Camera.ReadNoise ** 2 +
                        (self.metadata.Camera.NoiseFactor ** 2) *
                        self.metadata.Camera.ElectronsPerCount *
                        self.metadata.Camera.TrueEMGain *
                        (np.maximum(data, 1) + 1)) / \
                self.metadata.Camera.ElectronsPerCount
        if not self.sigma_is_scalar:
            self.noise_sigma = self.noise_sigma.squeeze()

        # > initialize sigma in device
        cl.sigma = cl.create_buffer(am.READ_ONLY,
                                    hostbuf=self.noise_sigma,
                                    host_ptr_mode=hm.COPY_HOST_PTR)

        # endregion : Noise sigma pre-processing

        # region : Initialize data in device

        # TODO
        '''Use 'data = data - bkg' to save space for data_mean,
            A (= data.max() - data.min()) in start parameter will be
            affected'''
        self.data_mean = self.data - self.background
        cl.data = cl.create_buffer(am.READ_ONLY,
                                   hostbuf=self.data_mean,
                                   host_ptr_mode=hm.COPY_HOST_PTR)

        # endregion : Initialize data in device

        pass

    # endregion : Constructor

    # region : Core Method

    def FromPoint(self, x, y, roi_half_size=5):
        # > get ROI [3.0%]
        # --------------------------------------------------------------
        # region : STD
        x_r = round(x)
        y_r = round(y)

        xslice = slice(max((x_r - roi_half_size), 0),
                       min((x_r + roi_half_size + 1),
                           self.data.shape[0]))
        yslice = slice(max((y_r - roi_half_size), 0),
                       min((y_r + roi_half_size + 1),
                           self.data.shape[1]))

        data = self.data[xslice, yslice]

        X = 1e3 * self.metadata.voxelsize.x * np.mgrid[xslice]
        Y = 1e3 * self.metadata.voxelsize.y * np.mgrid[yslice]

        # estimate errors in data
        sigma = self.noise_sigma
        if not self.sigma_is_scalar:
            sigma = self.noise_sigma[xslice, yslice]

        data_mean = self.data_mean[xslice, yslice]
        # endregion : STD

        # > estimate some start parameters
        # --------------------------------------------------------------
        # TODO
        '''Changing A will lead to large delta of final A and sigma'''
        # A = data_mean.max() - data_mean.min()  # amplitude
        A = data.max() - data.min()  # amplitude

        x0 = 1e3 * self.metadata.voxelsize.x * x
        y0 = 1e3 * self.metadata.voxelsize.y * y

        start_parameters = np.asarray([A, x0, y0, 250 / 2.35,
                                       data_mean.min(), .001, .001],
                                      np.float64)

        # > send data to device
        # TODO: set to CL scope temporarily
        # >> X
        cl.X = cl.create_buffer(cl.am.READ_ONLY, X.nbytes)
        cl.X.enqueue_write(X)
        # >> Y
        cl.Y = cl.create_buffer(cl.am.READ_ONLY, Y.nbytes)
        cl.Y.enqueue_write(Y)
        # >> x0
        cl.x0 = cl.create_buffer(cl.am.READ_WRITE,
                                 start_parameters.nbytes)
        cl.x0.enqueue_write(start_parameters)
        # >> wa
        wa = np.array([self.data_mean.shape[1], 0], np.int32)
        cl.wa = cl.create_buffer(cl.am.READ_WRITE,
                                 wa.nbytes)
        cl.wa.enqueue_write(wa)

        # > do the fit
        # --------------------------------------------------------------
        if False:
            L = X.size
            m = L * L
            n = 7
            filename = r'latgauss_export.data'
            f = open(filename, 'w')
            # > write X
            for i in range(L):
                f.write('%f\n' % X[i])
            # > write Y
            for i in range(L):
                f.write('%f\n' % Y[i])
            # > write y
            for i in range(L):
                for j in range(L):
                    f.write('%f\n' % data_mean[i, j])
            # > write sigma
            for i in range(L):
                for j in range(L):
                    f.write('%.16f\n' % sigma[i, j])
            # > write x0
            for i in range(n):
                f.write('%.16f\n' % start_parameters[i])
            # > write img
            img = np.asarray(self.data_mean, np.float64).flatten()
            for i in range(img.size):
                f.write('%.16f\n' % img[i])
            # > write sigma
            img_sig = np.asarray(self.noise_sigma, np.float64).flatten()
            for i in range(img_sig.size):
                f.write('%s.16f\n' % img_sig[i])

            # > close
            f.close()

        (res, cov_x, info_dict, msg, res_code) = fit_model_weighted(
            self.fit_fcn, start_parameters, data_mean, sigma, X, Y)

        # > try to estimate errors based on the covariance matrix
        # --------------------------------------------------------------
        fit_errors = None
        bg_mean = 0
        try:
            bg_mean = np.linalg.norm(info_dict['fvec'])
            fit_errors = np.sqrt(
                np.diag(cov_x) * (
                    info_dict['fvec'] * info_dict['fvec']).sum() /
                (len(data_mean.ravel()) - len(res)))
        except Exception:
            pass
            # print('!!! Failed to estimate errors based on the
            # covariance matrix')

        # > package results
        # --------------------------------------------------------------
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
    std_res = optimize.leastsq(weighted_miss_fit, start_parameters,
                               (model_fcn, data.ravel(), (1.0 /
    sigmas).
                                astype('f').ravel()) + args,
                               full_output=1)

    # res = lmdif(weighted_miss_fit, start_parameters,
    #             (model_fcn, data.ravel(), (1.0 / sigmas).
    #              astype('float64').ravel()) + args,
    #             full_output=1)

    # res = cl_leastsq(11)
    #
    # if True:  # DEBUG
    #     res[2]['fvec'] = weighted_miss_fit(
    #         res[0],
    #         model_fcn,
    #         data.ravel(),
    #         (1.0 / sigmas).astype('float64').ravel(),
    #         *args
    #     )

    return std_res


def cl_leastsq(L):
    cl.program.fit((L, L),
                   (cl.data, cl.sigma, cl.X, cl.Y, cl.x0, cl.wa),
                   (L, L))
    x = np.zeros(7, np.float64)
    wa = np.zeros(2, np.int32)

    cl.finish_default_queue()

    cl.x0.enqueue_read(x)
    cl.wa.enqueue_read(wa)

    # > wrap result
    dct = {'fjac': None, 'fvec': None, 'ipvt': None,
           'nfev': wa[1], 'qtf': None}
    mesg = 'info = %d' % wa[0]
    ier = wa[0]

    return x, None, dct, mesg, ier


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
