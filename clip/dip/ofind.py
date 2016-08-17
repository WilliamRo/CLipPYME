
##################
# clOfind.py
#
##################

import ofindPara as ofdP
#import pylab


class ObjectIdentifier(list):
    def __init__(self, data):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data), and a filtering mode (filterMode, one of ["fast", "good"])
        where "fast" performs a z-projection and then filters, wheras "good" filters in 3D before
        projecting. The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. filterRadiusZ is the radius used for the axial smoothing filter"""

        if 'md' not in ofdP.__dict__:
            ofdP.md = ofdP.clMetadata(data.shape[0], data.shape[1], 1, 0.08, 0.08, 0.10, True,\
                                      4, 12, 9, 25,ofdP.wl, ofdP.wh, \
                                      0.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5, True, 0, 0, 100, 1, 5)
            ofdP.ofindInit()
        self.clData = data.astype('f').reshape([ofdP.pixelCount])

    def FindObjects(self, index):
        """."""
        # run kernel in device
        self.ofdLen = ofdP.RunKernel(self.clData, index)