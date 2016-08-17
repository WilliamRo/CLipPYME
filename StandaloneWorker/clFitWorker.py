########################################################################
#
#   Created: August 14, 2016
#   Author: William Ro
#
########################################################################

"""clFitWorker

A opencl version of remFitBuf ...

"""

import numpy as np

from PYME.Analysis.remFitBuf import CameraInfoManager

# region : Global Fields

cameraMaps = CameraInfoManager()


# endregion : Global Fields

# region : Class backgroundBuffer

class backgroundBuffer:
    def __init__(self, dataBuffer):
        self.dataBuffer = dataBuffer
        self.curFrames = set()
        self.curBG = np.zeros(dataBuffer.shape[1:3], 'f4')

    def getBackground(self, bgindices):
        bgi = set(bgindices)

        # subtract frames we're currently holding but don't need
        for fi in self.curFrames.difference(bgi):
            self.curBG[:] = (self.curBG - self.dataBuffer[fi])[:]

        # add frames we don't already have
        for fi in bgi.difference(self.curFrames):
            self.curBG[:] = (self.curBG + self.dataBuffer[fi])[:]

        self.curFrames = bgi

        return self.curBG / len(bgi)


# endregion : Class backgroundBuffer

# region : Class Worker

class Worker:
    """Class Worker
    ----------------
    Demo:
    worker = Worker(...)
    results = worker.fit()"""

    # region : Constructor

    def __init__(self,
                 buffer,
                 threshold,
                 metadata,
                 fitModule,
                 SNThreshold=False):
        """Create a new fitting worker
        ----------------
        Parameters:
             buffer - static or dynamic image buffer
                      with a shape of (slices, width, height)
          threshold - threshold to be used to detect points n.b.
                      this applies to the filtered, potentially
                      bg subtracted data
           metadata - image metadata
          fitModule - name of module
        SNThreshold - ???"""

        # initialize fields
        self.dBuffer = buffer
        self.threshold = threshold
        self.md = metadata
        self.fitModule = fitModule
        self.findMethod = 'CLOfind'
        self.SNThreshold = SNThreshold

        self.bBuffer = backgroundBuffer(self.dBuffer)

        # initialize parameters
        self.minBgIndicesLen = self.md.getOrDefault( \
            'Analysis.MinimumBackgroudLength', 1)

        self.fitMod = self._getFitMod()
        self.ofdMod = self._getOfind()

    # endregion : Constructor

    # region : Core Method

    def fit(self, index, bgindices):
        """Do fitting on frame buffer[index]
        ----------------
        Parameters:
              index - slice index of frame to fit
          bgindices - indices of frames used to calculate
                      background = average(buffer[bgindices])"""

        # region : Preparation

        data = self.dBuffer[index]
        fitMod = self.fitMod

        # data = data.reshape(data.shape + (1,)) * 1.0

        # endregion : Preparation

        # region : ! Special cases - defer object finding to fit module
        # endregion

        # region : Find candidate molecule positions
        # > splitter mapping function is omitted

        # routines for splitter
        pass

        # perform objects finding

        ofd = self.ofdMod.ObjectIdentifier(data.astype('f'))
        ofd.FindObjects(index)

        # endregion : Find candidate molecule positions

        # region : ! Find fiducials and routines for splitter
        # endregion

        # region : Perform fit for each point that we detected
        # > drift estimation is omitted
        # without this line fitFac.FromPoint will fail
        self.md.tIndex = index
        res_len = ofd.ofdLen  # TODO

        if 'FitResultsDType' in dir(fitMod):
            if 'Analysis.ROISize' in self.md.getEntryNames():
                rs = self.md.getEntry('Analysis.ROISize')
                res = fitMod.from_points(self.md, res_len,
                                         data.shape[1],
                                         2 * rs + 1)
            else:
                res = fitMod.from_points(self.md, res_len,
                                         data.shape[1])

        # endregion : Perform fit for each point that we detected

        return res

    # endregion : Core Method

    # region : Private Methods

    def _getFitMod(self):
        # import CL based modules
        return __import__('clip.dip.LatGaussFit',
                              fromlist=['clip', 'dip'])


    def _getOfind(self):
        # import CL based modules
        return __import__('clip.dip.clOfind',
                            fromlist=['clip', 'dip'])

    # endregion : Private Methods

    pass  # Do not delete this line

# endregion : Class Worker
