########################################################################
#
#   Created: June 1, 2016
#   Author: William Ro
#
########################################################################

"""fitWorker

A standalone version of remFitBuf ...

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
        self.SNThreshold = SNThreshold

        self.bBuffer = backgroundBuffer(self.dBuffer)

        # initialize parameters
        self.minBgIndicesLen = self.md.getOrDefault( \
            'Analysis.MinimumBackgroudLength', 1)

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
        fitMod = self._getFitMod()

        # endregion : Preparation

        # region : Calculate background and noise
        # ! without cameraMaps.correctImage(...)

        data = data.reshape(data.shape + (1,)) * 1.0

        if len(bgindices) >= self.minBgIndicesLen:
            bg = self.bBuffer.getBackground(bgindices).reshape(data.shape)
        else:
            bg = self.md['Camera.ADOffset']

        sigma = self.calcSigma(data - self.md['Camera.ADOffset'])

        # endregion : Calculate background and noise

        # region : ! Special cases - defer object finding to fit module
        # endregion

        # region : Find candidate molecule positions
        # > splitter mapping function is omitted

        bgd = data.astype('f') - bg  # shape = (w, h, 1)

        # routines for splitter
        pass

        # get ofind
        import PYME.Analysis.ofind as ofind
        ofd = ofind.ObjectIdentifier(bgd * (bgd > 0))

        # perform objects finding
        debounce = self.md.getOrDefault('Analysis.DebounceRadius', 5)
        discardClumpRadius = self.md.getOrDefault('Analysis.ClumpRejectRadius', 0)

        ofd.FindObjects(self.calcThreshold(sigma), 0,
                        debounceRadius=debounce,
                        discardClumpRadius=discardClumpRadius)

        # endregion : Find candidate molecule positions

        # region : ! Find fiducials and routines for splitter
        # endregion

        # region : Perform fit for each point that we detected
        # > drift estimation is omitted

        # create a fit 'factory'
        if 'cl' in dir(fitMod):
            # CLip class has no fit_fcn in constructor
            fitFac = fitMod.FitFactory(data, self.md, bg, sigma)
        else:
            fitFac = fitMod.FitFactory(data, self.md,
                                       background=bg,
                                       noiseSigma=sigma)

        # without this line fitFac.FromPoint will fail
        self.md.tIndex = index

        if 'FitResultsDType' in dir(fitMod):
            res = np.empty(len(ofd), fitMod.FitResultsDType)
            if 'Analysis.ROISize' in self.md.getEntryNames():
                rs = self.md.getEntry('Analysis.ROISize')
                for i in range(len(ofd)):
                    p = ofd[i]
                    res[i] = fitFac.FromPoint(p.x, p.y, roiHalfSize=rs)
            else:
                for i in range(len(ofd)):
                    p = ofd[i]
                    res[i] = fitFac.FromPoint(p.x, p.y)
        else:
            res = [fitFac.FromPoint(p.x, p.y) for p in ofd]

        # endregion : Perform fit for each point that we detected

        return res

    # endregion : Core Method

    # region : Private Methods

    def _getFitMod(self):
        # import CL based modules
        if self.fitModule == 'LatGaussFitFR':
            return __import__('clip.dip.LatGaussFit', fromlist=['clip', 'dip'])
        # import PYME modules
        else:
            return __import__('PYME.Analysis.FitFactories.' + self.fitModule,
                              fromlist=['PYME', 'Analysis', 'FitFactories'])

    def _getOfind(self):
        # !
        pass

    def calcSigma(self, data):
        var = cameraMaps.getVarianceMap(self.md)

        n = self.md.Camera.NoiseFactor
        e = self.md.Camera.ElectronsPerCount
        t = self.md.Camera.TrueEMGain

        return np.sqrt(var + (n ** 2) * (e * t * np.maximum(data, 1) + t * t)) / e

    def calcThreshold(self, sigma):
        if self.SNThreshold:
            # to account for the fact that the blurring etc...
            #   in ofind doesn't preserve intensities - at the
            #   moment completely arbitrary so a threshold setting
            #   of 1 results in reasonable detection.
            fudgeFactor = 1
            return (sigma * fudgeFactor * self.threshold).squeeze()
        else:
            return self.threshold

    # endregion : Private Methods

    pass  # Do not delete this line

# endregion : Class Worker - 8th Test
