#!/usr/bin/python

##################
# clofind.py
#
##################

from scipy import ndimage
from scipy.ndimage import _nd_image, _ni_support
from scipy.spatial import ckdtree
import  numpy as np
import time
import ofindPara as ofdP
import  clip.cl as cl
from PYME.Analysis.remFitBuf import CameraInfoManager
#import pylab

cameraMaps = CameraInfoManager()

class OfindPoint:
    def __init__(self, x, y, z=None, detectionThreshold=None):
        """Creates a point object, potentially with an undefined z-value."""
        self.x = x
        self.y = y
        self.z = z
        self.detectionThreshold = detectionThreshold

#a hack so we'll be able to do something like ObjectIdentifier.x[i]
class PseudoPointList:
    def __init__(self,parent, varName):
        self.parent = parent
        self.varName = varName

    def __len__(self):
        return len(self.parent)

    def __getitem__(self, key):
        tmp = self.parent[key]
        if not '__len__' in dir(tmp):
            return tmp.__dict__[self.varName]
        else:
            tm2 = []
            for it in tmp:
                tm2.append(it.__dict__[self.varName])
            return tm2

    #def __iter__(self):
    #    self.curpos = -1
    #    return self

    #def next(self):
    #    curpos += 1
    #    if (curpos >= len(self)):
    #        raise StopIteration
    #    return self[self.curpos]

class ObjectIdentifier(list):
    def __init__(self, data, bg, metadata, SNThreshold, threshold):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data), and a filtering mode (filterMode, one of ["fast", "good"])
        where "fast" performs a z-projection and then filters, wheras "good" filters in 3D before
        projecting. The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. filterRadiusZ is the radius used for the axial smoothing filter"""
        global filtRadLowpass, filtRadHighpass, weightsLowpass, weightsHighpass

        if 'md' not in ofdP.__dict__:
            ofdP.md = ofdP.clMetadata(data.shape[1], data.shape[0], data.shape[2], 0.08, 0.08, 0.10, True, 4, 12, 9, 25, ofdP.wl, ofdP.wh, \
                                      0.0, 1.0, 1.0, 1.0, 5.0, 1.0, 1.0, 5, True, 0, 0, 100, 1, 5)
            ofdP.ofindInit()
        self.clData = data.astype('f').reshape([ofdP.pixelCount])
        self.data = data.astype('f')
        self.rawData = self.data
        self.bg = bg
        self.metadata = metadata
        self.SNThreshold = SNThreshold
        self.threshold = threshold
        # define cl data structrue
        self.clFilteredData = np.zeros(ofdP.pixelCount, 'float32')
        self.candiPosi = np.zeros([ofdP.maxCount * 2], 'float32')
        self.candiCount = np.zeros(2, 'int32')
        self.clBinaryImage = np.zeros(ofdP.pixelCount, 'uint16')
        self.clLabel = np.zeros(ofdP.pixelCount, 'int32')
        self.clThreshold = np.zeros(ofdP.pixelCount, 'float32')
        self.clDebounceCandi = np.zeros(ofdP.maxCount * 2, 'float32')
        self.clStartPara = np.zeros(ofdP.maxCount * 7, 'float32')
        self.clSigma = np.zeros(ofdP.pixelCount, 'float32')



    def __FilterDataFast(self):
        data = self.data[:,:,0]
        mode = _ni_support._extend_mode_to_code("reflect")
        # lowpass filter to suppress noise
        output, a = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, ofdP.weightsLowpass, 0, output, mode, 0, 0)
        _nd_image.correlate1d(output, ofdP.weightsLowpass, 1, output, mode, 0, 0)

        # lowpass filter again to find background
        output, b = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, ofdP.weightsHighpass, 0, output, mode, 0, 0)
        _nd_image.correlate1d(output, ofdP.weightsHighpass, 1, output, mode, 0, 0)

        return a - b

    def __Debounce(self, xs, ys, radius=4):
        if len(xs) < 2:
            return xs, ys

        kdt = ckdtree.cKDTree(np.array([xs,ys]).T)

        xsd = []
        ysd = []
        visited = 0*xs

        for i in xrange(len(xs)):
            if not visited[i]:
                xi = xs[i]
                yi = ys[i]
                #neigh = kdt.query_ball_point([xi,yi], radius)

                dn, neigh = kdt.query(np.array([xi,yi]), 5)

                neigh = neigh[dn < radius]

                # if len(neigh) > 5:
                #     neigh = neigh[0:5]

                if len(neigh) > 1:
                    In = self.filteredData[xs[neigh].astype('i'),ys[neigh].astype('i')]
                    mi = In.argmax()

                    xsd.append(xs[neigh[mi]])
                    ysd.append(ys[neigh[mi]])
                    visited[neigh] = 1


                else:
                    xsd.append(xi)
                    ysd.append(yi)
        # xsdm = []
        # ysdm = []
        # for i in xrange(len(xsd)):
        #     if (not xsd[i] in xsdm) or (not ysd[i] in ysdm):
        #         xsdm.append(xsd[i])
        #         ysdm.append(ysd[i])

        return xsd, ysd

        # return xsd, ysd

    def __discardClumped(self, xs, ys, radius=4):
        if len(xs) < 2:
            return xs, ys

        kdt = ckdtree.cKDTree(np.array([xs,ys]).T)

        xsd = []
        ysd = []

        for i in xrange(len(xs)):
            xi = xs[i]
            yi = ys[i]
            #neigh = kdt.query_ball_point([xi,yi], radius)

            dn, neigh = kdt.query(np.array([xi,yi]), 2)
            print dn

            if (dn[1] > radius):
                xsd.append(xi)
                ysd.append(yi)

        print len(xsd)

        return np.array(xsd), np.array(ysd)

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

    def calcSigma(self, data):
        var = cameraMaps.getVarianceMap(self.metadata)

        n = self.metadata.Camera.NoiseFactor
        e = self.metadata.Camera.ElectronsPerCount
        t = self.metadata.Camera.TrueEMGain

        return np.sqrt(
            var + (n ** 2) * (e * t * np.maximum(data, 1) + t * t)) / e

    def FindObjects(self,
                    index,
                    debounceRadius = 4,
                    maskEdgeWidth = 5,
                    discardClumpRadius = 0):
        """."""
        #clear the list of previously found points
        del self[:]

        # run kernel in device

        for i in xrange(1):
            if i != 0:
                ofdP.cpuStartTime.append(time.time())
            # copy data into device

            ofdP.RunKernel(self.clData, index)

            if i != 0:
                ofdP.cpuEndTime.append(time.time())

        if ofdP.viewKernelInfo:
            ofdP.PrintTotalInformation()

        if ofdP.verify:

            sigma = self.calcSigma(self.rawData - self.metadata['Camera.ADOffset'])
            self.lowerThreshold = self.calcThreshold(sigma)

            # copy data into host
            cl.enqueue_copy(ofdP.commandQueue, self.clData, cl.memImage)
            cl.enqueue_copy(ofdP.commandQueue, self.clLabel,cl.memLabeledImage)
            cl.enqueue_copy(ofdP.commandQueue, self.candiCount,cl.memCandiCount)
            cl.enqueue_copy(ofdP.commandQueue, self.clDebounceCandi,cl.memCandiPosi)
            cl.enqueue_copy(ofdP.commandQueue, self.clFilteredData, cl.memFilteredImage)
            cl.enqueue_copy(ofdP.commandQueue, self.clBinaryImage, cl.memBinaryImage)
            cl.enqueue_copy(ofdP.commandQueue, self.clThreshold, cl.memThresholdMap)
            cl.enqueue_copy(ofdP.commandQueue, self.clStartPara, cl.memStartPara)
            cl.enqueue_copy(ofdP.commandQueue, self.candiPosi, cl.memTempCandiPosi).wait()

            self.clFilteredData = self.clFilteredData.reshape([ofdP.md.imageHeight, ofdP.md.imageWidth])

            sBg = self.data - self.bg
            sBg = (sBg) * (sBg > 0)
            ofdP.compareMatrix(sBg,
                          self.clData.reshape([ofdP.md.imageHeight, ofdP.md.imageWidth]),
                          '>>> 0. Subtracted image')

            ofdP.compareMatrix(self.lowerThreshold,
                           self.clThreshold.reshape([ofdP.md.imageHeight, ofdP.md.imageWidth]),
                           '>>> 0.1. Threshold')

            # >>>> 1. do filtering
            self.data = self.data - self.bg
            self.data = self.data * (self.data > 0)
            self.filteredData = self.__FilterDataFast()
            self.filteredData *= (self.filteredData > 0)

            # apply mask
            maskedFilteredData = self.filteredData
            # maskedFilteredData = self.clFilteredData

            # manually mask the edge pixels
            if maskEdgeWidth and self.filteredData.shape[1] > maskEdgeWidth:
                maskedFilteredData[:, :maskEdgeWidth] = 0
                maskedFilteredData[:, -maskEdgeWidth:] = 0
                maskedFilteredData[-maskEdgeWidth:, :] = 0
                maskedFilteredData[:maskEdgeWidth, :] = 0

                ofdP.compareMatrix(self.clFilteredData, maskedFilteredData,
                               '>>> 1. Filter result', 0.01)

            X,Y = np.mgrid[0:maskedFilteredData.shape[0], 0:maskedFilteredData.shape[1]]

            #store x, y, and thresholds
            xs = []
            ys = []
            ts = []

            # >>>> 2. applying threshold
            im = maskedFilteredData
            imt = im > self.lowerThreshold
            climt = self.clBinaryImage.reshape([ofdP.md.imageHeight, ofdP.md.imageWidth])
            ofdP.compareMatrix(imt, climt, '>>> 2. Applying threshold', 0)

            # >>>> 3. labeling image
            (labeledPoints, nLabeled) = ndimage.label(imt)

            labelSet = set(self.clLabel)
            labelList = []
            for i in labelSet:
                labelList.append(i)
            labelList.sort()
            self.clLabel = self.clLabel.reshape([ofdP.md.imageHeight, ofdP.md.imageWidth])
            for i in xrange(nLabeled):
                self.clLabel[self.clLabel == labelList[i+1]] = i+1;
            # compareMatrix(labeledPoints, self.clLabel, '>>> 3. Label', 0)

            objSlices = ndimage.find_objects(labeledPoints)

            # >>>> 4. calculateing candidate position
            for i in range(nLabeled):
                #measure position
                imO = im[objSlices[i]]
                x = (X[objSlices[i]]*imO).sum()/imO.sum()
                y = (Y[objSlices[i]]*imO).sum()/imO.sum()

                #and add to list
                xs.append(x)
                ys.append(y)
                ts.append(self.lowerThreshold)

            candiRes = np.zeros([nLabeled,2],dtype='f')
            clCandiRes = np.zeros([nLabeled,2],dtype='f')
            for i in xrange(nLabeled):
                candiRes[i,0] = xs[i]
                candiRes[i,1] = ys[i]
                clCandiRes[i,0] = self.candiPosi[2*i]
                clCandiRes[i,1] = self.candiPosi[2*i+1]
            ofdP.compareMatrix(candiRes,clCandiRes,
                               '>>> 4. Calculating candidate position', tol=0.01)

            xs = np.array(xs)
            ys = np.array(ys)

            if discardClumpRadius > 0:
                print 'ditching clumps'
                xs, ys = self.__discardClumped(xs, ys, discardClumpRadius)

            xs, ys = self.__Debounce(xs, ys, debounceRadius)
             # xs, ys = self.Debounce(xs, ys, debounceRadius)

            for x, y, t in zip(xs, ys, ts):
                self.append(OfindPoint(x,y,t))

            # >>> 5. fit initialization
            startPara = np.zeros([self.candiCount[1], 7], 'float32')
            roiHalfSize = 5
            for i in xrange(self.candiCount[1]):
                px = self.clDebounceCandi[2 * i]
                py = self.clDebounceCandi[2 * i + 1]
                x = round(px)
                y = round(py)
                xslice = slice(max((x - roiHalfSize), 0), min((x + roiHalfSize + 1), self.data.shape[0]))
                yslice = slice(max((y - roiHalfSize), 0), min((y + roiHalfSize + 1), self.data.shape[1]))
                dataROI = self.rawData[xslice, yslice] - self.metadata.Camera.ADOffset

                if not self.bg is None and len(np.shape(self.bg)) > 1 and not (
                                'Analysis.subtractBackground' in self.metadata.getEntryNames() and self.metadata.Analysis.subtractBackground == False):
                    bgROI = self.bg[xslice, yslice]- self.metadata.Camera.ADOffset
                else:
                    bgROI = 0
                dataMean = dataROI - bgROI
                A = dataROI.max() - dataROI.min()
                x0 = 1e3 * self.metadata.voxelsize.x * px
                y0 = 1e3 * self.metadata.voxelsize.y * py
                startPara[i, :] = [A, x0, y0, 250 / 2.35, dataMean.min(), .001, .001]
            clStartPara = self.clStartPara.reshape([ofdP.maxCount, 7])
            clStartPara = clStartPara[:self.candiCount[1], :]
            ofdP.compareMatrix(clStartPara, startPara, '>>> 5. Fit initialization',0.01)

        else:
            if False:
                cl.enqueue_copy(ofdP.commandQueue, self.clFilteredData, cl.memFilteredImage)
                cl.enqueue_copy(ofdP.commandQueue, self.clDebounceCandi, cl.memCandiPosi)
                cl.enqueue_copy(ofdP.commandQueue, self.candiPosi, cl.memTempCandiPosi).wait()
                xs = []
                ys = []
                ts = []
                for i in xrange(self.candiCount[0]):
                    xs.append(self.candiPosi[2*i])
                    ys.append(self.candiPosi[2*i+1])
                    ts.append(self.lowerThreshold)
                xs = np.array(xs)
                ys = np.array(ys)

                self.filteredData = self.clFilteredData.reshape([ofdP.md.imageHeight, md.imageWidth])
                xs, ys = self.__Debounce(xs, ys, debounceRadius)
                for i in xrange(len(xs)):
                    self.clDebounceCandi[2*i] = xs[i]
                    self.clDebounceCandi[2*i+1] = ys[i]
                cl.enqueue_copy(ofdP.commandQueue, cl.memCandiCount, np.array([0, len(xs)]))
                cl.enqueue_copy(ofdP.commandQueue, cl.memCandiPosi, self.clDebounceCandi).wait()
                cl.fitInit.enqueue_nd_range(cl.fitInitGlobalDim,
                                            ofdP.commandQueue,
                                            cl.fitInitLocalDim)
                ofdP.commandQueue.finish()

                for x, y, t in zip(xs, ys, ts):
                    self.append(OfindPoint(x, y, t))

        cl.enqueue_copy(ofdP.commandQueue, self.candiCount, cl.memCandiCount,
                        is_blocking=False).wait()
        self.ofdLen = self.candiCount[1]
        self.sigma = self.clSigma.reshape([ofdP.md.imageHeight, ofdP.md.imageWidth])

        #create pseudo lists to allow indexing along the lines of self.x[i]
        # self.x = PseudoPointList(self, 'x')
        # self.y = PseudoPointList(self, 'y')




