#!/usr/bin/python

##################
# LatGaussFitFR.py
#
# Copyright David Baddeley, 2009
# d.baddeley@auckland.ac.nz
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
##################

import numpy as np
from .fitCommon import fmtSlicesUsed
from . import FFBase

from PYME.Analysis.cModels.gauss_app import genGauss,genGaussJac, genGaussJacW
from PYME.Analysis._fithelpers import FitModelWeighted, FitModelWeightedJac


##################
# Model functions
def f_gauss2dSlow(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    return A*np.exp(-((X-x0)**2 + (Y - y0)**2)/(2*s**2)) + b + b_x*X + b_y*Y

def f_gauss2d(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y]"""
    A, x0, y0, s, b, b_x, b_y = p
    r = genGauss(X,Y,A,x0,y0,s,b,b_x,b_y) #this is coded in c and defined in gauss_app
    return r

def f_gauss2dF(p, X, Y):
    """2D Gaussian model function with linear background - parameter vector [A, x0, y0, sigma, background, lin_x, lin_y] - uses fast exponential approx"""
    A, x0, y0, s, b, b_x, b_y = p
    r = genGaussF(X,Y,A,x0,y0,s,b,b_x,b_y)
    return r

def f_j_gauss2d(p,func, d, w, X,Y):
    '''generate the jacobian for a 2d Gaussian'''
    A, x0, y0, s, b, b_x, b_y = p
    r = genGaussJacW(X,Y,w,A,x0,y0,s,b,b_x,b_y)
    r = -r.ravel().reshape((-1,7))
    return r.T

def f_J_gauss2d(p,X,Y):
    '''generate the jacobian for a 2d Gaussian - for use with _fithelpers.weightedJacF'''
    A, x0, y0, s, b, b_x, b_y = p
    r = genGaussJac(X,Y,A,x0,y0,s,b,b_x,b_y)
    r = r.reshape((-1, 7))
    return r

f_gauss2d.D = f_J_gauss2d

#####################

#define the data type we're going to return
fresultdtype=[('tIndex', '<i4'),
              ('fitResults', [('A', '<f4'),
                              ('x0', '<f4'),('y0', '<f4'),
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
              ('slicesUsed', [('x', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('y', [('start', '<i4'),('stop', '<i4'),('step', '<i4')]),
                              ('z', [('start', '<i4'),('stop', '<i4'),('step', '<i4')])]),
              ('subtractedBackground', '<f4')
              ]

def GaussianFitResultR(fitResults, metadata, slicesUsed=None, resultCode=-1, fitErr=None, background=0):
    slicesUsed = fmtSlicesUsed(slicesUsed)
    #print slicesUsed

    if fitErr is None:
        fitErr = -5e3*np.ones(fitResults.shape, 'f')

    res =  np.array([(metadata.tIndex, fitResults.astype('f'), fitErr.astype('f'), resultCode, slicesUsed, background)], dtype=fresultdtype)
    #print res
    return res


class GaussianFitFactory(FFBase.FitFactory):
    def __init__(self, data, metadata, fitfcn=f_gauss2d, background=None, noiseSigma=None):
        '''Create a fit factory which will operate on image data (data), potentially using voxel sizes etc contained in
        metadata. '''
        FFBase.FitFactory.__init__(self, data, metadata, fitfcn, background, noiseSigma)

        if False:#'D' in dir(fitfcn): #function has jacobian
            self.solver = FitModelWeightedJac
        else:
            self.solver = FitModelWeighted

    def FromPoint(self, x, y, z=None, roiHalfSize=5, axialHalfSize=15):
        X, Y, data, background, sigma, xslice, yslice, zslice = self.getROIAtPoint(x, y, z, roiHalfSize, axialHalfSize)

        dataMean = data - background

        #estimate some start parameters...
        A = data.max() - data.min() #amplitude

        x0 =  1e3*self.metadata.voxelsize.x*x
        y0 =  1e3*self.metadata.voxelsize.y*y

        bgm = np.mean(background)

        startParameters = [A, x0, y0, 250/2.35, dataMean.min(), .001, .001]

        #do the fit
        (res, cov_x, infodict, mesg, resCode) = self.solver(self.fitfcn, startParameters, dataMean, sigma, X, Y)

        #try to estimate errors based on the covariance matrix
        fitErrors=None
        try:
            bgm = np.linalg.norm(infodict['fvec'])
            fitErrors = np.sqrt(np.diag(cov_x)*(infodict['fvec']*infodict['fvec']).sum()/(len(dataMean.ravel())- len(res)))
        except Exception:
            pass

        #package results
        return GaussianFitResultR(res, self.metadata, (xslice, yslice, zslice), resCode, fitErrors, bgm)

    @classmethod
    def evalModel(cls, params, md, x=0, y=0, roiHalfSize=5):
        '''Evaluate the model that this factory fits - given metadata and fitted parameters.

        Used for fit visualisation'''
        #generate grid to evaluate function on
        X = 1e3*md.voxelsize.x*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]
        Y = 1e3*md.voxelsize.y*np.mgrid[(x - roiHalfSize):(x + roiHalfSize + 1)]

        return (f_gauss2d(params, X, Y), X[0], Y[0], 0)


#so that fit tasks know which class to use
FitFactory = GaussianFitFactory
FitResult = GaussianFitResultR
FitResultsDType = fresultdtype #only defined if returning data as numarray

DESCRIPTION = 'Vanilla 2D Gaussian fit.'
LONG_DESCRIPTION = 'Single colour 2D Gaussian fit. This should be the first stop for simple analyisis.'
USE_FOR = '2D single-colour'