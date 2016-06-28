#!/usr/bin/python

##################
# ofind.py
#
##################

#import scipy
import numpy
import math
from scipy import ndimage
from scipy.ndimage import _nd_image, _ni_support
from scipy.spatial import ckdtree
import  numpy as np
import  clip.cl as cl
import ctypes as ct
import sys
#import pylab


def calc_gauss_weights(sigma):
    '''calculate a gaussian filter kernel (adapted from scipy.ndimage.filters.gaussian_filter1d)'''
    sd = float(sigma)
    # make the length of the filter equal to 4 times the standard
    # deviations:
    lw = int(4.0 * sd + 0.5)
    weights = numpy.zeros(2 * lw + 1, 'float64')
    weights[lw] = 1.0
    sum = 1.0
    sd = sd * sd
    # calculate the kernel:
    for ii in range(1, lw + 1):
        tmp = math.exp(-0.5 * float(ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum += 2.0 * tmp

    return weights/sum

#default filter sizes
filtRadLowpass = 1
filtRadHighpass = 3

#precompute our filter weights
weightsLowpass = calc_gauss_weights(filtRadLowpass)
weightsHighpass = calc_gauss_weights(filtRadHighpass)

# region : CLOFIND

# region : create metadata
lowFilterLength = 9
highFilterLength = 25
class clMetadata(ct.Structure):
    _fields_ = [("imageWidth", ct.c_int32),
                ("imageHeight", ct.c_int32),
                ("imageDepth", ct.c_int32),
                ("voxelSizeX", ct.c_double),
                ("voxelSizeY", ct.c_double),
                ("voxelSizeZ", ct.c_double),
                ("is2DImage", ct.c_bool),
                ("filterRadiusLowpass", ct.c_int32),
                ("filterRadiusHighpass", ct.c_int32),
                ("lowFilterLength", ct.c_int32),
                ("highFilterLength", ct.c_int32),
                ("weightsLow", ct.c_double*lowFilterLength),
                ("weightsHigh", ct.c_double*highFilterLength),
                ("cameraOffset", ct.c_double),
                ("cameraNoiseFactor", ct.c_double),
                ("cameraElectronsPerCount", ct.c_double),
                ("cameraTrueEMGain", ct.c_double),
                ("debounceRadius", ct.c_double),
                ("threshold", ct.c_double),
                ("fudgeFactor", ct.c_double),
                ("maskEdgeWidth", ct.c_int32),
                ("SNThreshold", ct.c_bool)]
wl = (ct.c_double * lowFilterLength)()
for i in xrange(lowFilterLength):
    wl[i] = ct.c_double(weightsLowpass[i])
wh = (ct.c_double * highFilterLength)()
for i in xrange(highFilterLength):
    wh[i] = ct.c_double(weightsHighpass[i])
md = clMetadata(170,140,1,0.08,0.08,0.10,True,4,12,9,25,wl,wh,\
                0.0,1.0,1.0,1.0,5.0,0.6,1.0,5,True)


# endregion : create metadata

# other parameters
pixelCount = md.imageWidth * md.imageHeight
maxCount = 5000
maxRegionPointNum = 2000
maxpass = 10
candiCount = 0
typeIntSize = ct.sizeof(ct.c_int32)
typeShortSize = ct.sizeof(ct.c_int16)
typeDoubleSize = ct.sizeof(ct.c_double)
typeBoolSize = ct.sizeof(ct.c_bool)

# get device and default commandQueue
context = cl.context
device = context.default_device
commandQueue = context.default_queue
program = cl.program
ma = cl.mem_access_mode

# region : create kernel

cl.colFilterKernel = program.colFilterImage
cl.rowFilterKernel = program.rowFilterImage
cl.labelInitKernel = program.labelInit
cl.labelPropagateKernel = program.labelMain
cl.candiInitKernel = program.calcCandiPosiInit
cl.candiSumKernel = program.calcCandiPosiSum
cl.candiMainKernel = program.caclCandiPosiMain
cl.debounceCandiKernel = program.debounceCandiPosi

# endregion : create kernel

# region : create memory buffer

cl.memMetadata = context.create_buffer(ma.READ_ONLY, ct.sizeof(md))
cl.memImage = context.create_buffer(ma.READ_ONLY, pixelCount * typeShortSize)
cl.memFilteredImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeDoubleSize)
cl.memHighFilteredImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeDoubleSize)
cl.memLowFilteredImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeDoubleSize)
cl.memSyncIndex = context.create_buffer(ma.READ_WRITE, 2 * typeIntSize)
cl.memBinaryImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeShortSize)
cl.memLabeledImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeIntSize)
cl.memCandiPosi = context.create_buffer(ma.READ_WRITE, 2 * maxCount * typeDoubleSize)
cl.memTempCandiPosi = context.create_buffer(ma.READ_WRITE, 2 * maxCount * typeDoubleSize)
cl.memRegionPointPosi = context.create_buffer(ma.READ_WRITE, maxCount * maxRegionPointNum * typeShortSize)
cl.memCandiCount = context.create_buffer(ma.READ_WRITE, typeIntSize)
cl.memPointCount = context.create_buffer(ma.READ_WRITE, maxCount * maxCount)

# endregion : create memory buffer

# region : set kernel arguments and wotk item dimension

cl.colFilterKernel.set_arg(0, cl.memImage)
cl.colFilterKernel.set_arg(1, cl.memLowFilteredImage)
cl.colFilterKernel.set_arg(2, cl.memHighFilteredImage)
cl.colFilterKernel.set_arg(3, cl.memMetadata)

cl.rowFilterKernel.set_arg(0, cl.memLowFilteredImage)
cl.rowFilterKernel.set_arg(1, cl.memHighFilteredImage)
cl.rowFilterKernel.set_arg(2, cl.memFilteredImage)
cl.rowFilterKernel.set_arg(3, cl.memBinaryImage)
cl.rowFilterKernel.set_arg(4, cl.memMetadata)
cl.filterKernelGlobalSize = [md.imageHeight, md.imageWidth, 1]
cl.filterKernelLocalSize = [16, 16, 1]

cl.labelInitKernel.set_arg(0, cl.memLabeledImage)
cl.labelInitKernel.set_arg(1, cl.memBinaryImage)
cl.labelInitKernel.set_arg(2, cl.memMetadata)

cl.labelPropagateKernel.set_arg(0, cl.memLabeledImage)
cl.labelPropagateKernel.set_arg(1, cl.memMetadata)
cl.labelPropagateKernel.set_arg(2, cl.memSyncIndex)
cl.labelKernelGlobalSize = [(md.imageHeight + 31)&~31, (md.imageWidth + 31)&~31, 1]
cl.labelKernelLocalSize = [1, 1, 1]

cl.candiInitKernel.set_arg(0, cl.memLabeledImage)
cl.candiInitKernel.set_arg(1, cl.memRegionPointPosi)
cl.candiInitKernel.set_arg(2, cl.memCandiCount)
cl.candiInitKernel.set_arg(3, cl.memPointCount)
cl.candiInitKernel.set_arg(4, cl.memMetadata)

cl.candiSumKernel.set_arg(0, cl.memLabeledImage)
cl.candiSumKernel.set_arg(1, cl.memRegionPointPosi)
cl.candiSumKernel.set_arg(2, cl.memCandiCount)
cl.candiSumKernel.set_arg(3, cl.memPointCount)
cl.candiSumKernel.set_arg(4, cl.memMetadata)
cl.candiInitKernelGlobalSize = [md.imageHeight, md.imageWidth, 1]
cl.candiInitKernelLocalSize = [1, 1, 1]

cl.candiMainKernel.set_arg(0, cl.memFilteredImage)
cl.candiMainKernel.set_arg(1, cl.memTempCandiPosi)
cl.candiMainKernel.set_arg(2, cl.memFilteredImage)
cl.candiMainKernel.set_arg(3, cl.memCandiCount)
cl.candiMainKernel.set_arg(4, cl.memPointCount)
cl.candiMainKernel.set_arg(5, cl.memMetadata)

cl.debounceCandiKernel.set_arg(0, cl.memFilteredImage)
cl.debounceCandiKernel.set_arg(1, cl.memCandiPosi)
cl.debounceCandiKernel.set_arg(2, cl.memTempCandiPosi)
cl.debounceCandiKernel.set_arg(3, cl.memCandiCount)
cl.debounceCandiKernel.set_arg(4, cl.memMetadata)
cl.candiMainKernelGlobalSize = [maxCount, 1, 1]
cl.candiMainKernelLocalSize = [1, 1, 1]

# endregion : set kernel arguments and wotk item dimension

# write metadata into device
# cl.enqueue_copy(commandQueue, cl.memMetadata, clMetadata)
cl.enqueue_copy(commandQueue, cl.memMetadata, md)

# endregion : CLOFIND

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
    def __init__(self, data):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data), and a filtering mode (filterMode, one of ["fast", "good"])
        where "fast" performs a z-projection and then filters, wheras "good" filters in 3D before
        projecting. The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. filterRadiusZ is the radius used for the axial smoothing filter"""
        global filtRadLowpass, filtRadHighpass, weightsLowpass, weightsHighpass

        self.clData = data.astype('uint16').reshape([pixelCount])
        self.data = data.astype('f')


    def __FilterData2D(self,data):
        mode = _ni_support._extend_mode_to_code("reflect")
        #lowpass filter to suppress noise
        output, a = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, weightsLowpass, 0, output, mode, 0,0)
        _nd_image.correlate1d(output, weightsLowpass, 1, output, mode, 0,0)

        #lowpass filter again to find background
        output, b = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, weightsHighpass, 0, output, mode, 0,0)
        _nd_image.correlate1d(output, weightsHighpass, 1, output, mode, 0,0)

        return a - b

    def __FilterDataFast(self):
        #project data
        if len(self.data.shape) == 2: #if already 2D, do nothing
            return self.__FilterData2D(self.data)
        else:
            return sum([self.__FilterData2D(self.data[:,:,i]) for i in range(self.data.shape[2])])

    def __Debounce(self, xs, ys, radius=4):
        if len(xs) < 2:
            return xs, ys

        kdt = ckdtree.cKDTree(numpy.array([xs,ys]).T)

        xsd = []
        ysd = []
        visited = 0*xs

        for i in xrange(len(xs)):
            if not visited[i]:
                xi = xs[i]
                yi = ys[i]
                #neigh = kdt.query_ball_point([xi,yi], radius)

                dn, neigh = kdt.query(numpy.array([xi,yi]), 5)

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

    def Debounce(self, xs, ys, radius=4):
        if len(xs) < 2:
            return xs, ys

        xsd =[]
        ysd = []
        visited = 0*xs

        for i in xrange(len(xs)):
            if not visited[i]:
                xi = xs[i]
                yi = ys[i]
                neighDistance = []
                neigh = []

                # for j in xrange(len(xs)):
                #     dis = self._distance([xs[j],ys[j]],[xi, yi],mode='Euclid')
                #     neighDistance.append(dis)
                #     neigh.append(j)
                #
                # neighSorted = []
                # neighDistanceSorted = sorted(neighDistance)
                # for j in xrange(len(xs)):
                #     neighSorted.append(neigh[neighDistance.index(neighDistanceSorted[j])])
                # neigh = np.ndarray(shape=(5),dtype=int, buffer=np.array(neighSorted[0:5]))
                # dn = np.ndarray(shape=(5),dtype=float, buffer=np.array(neighDistanceSorted[0:5]))
                count = 0
                for j in xrange(len(xs)):
                    dis = self._distance([xs[j], ys[j]], [xi, yi], mode='Euclid')
                    if (dis < radius):
                        neigh.append(j)
                        count = count + 1

                if count > 5:
                    neigh = np.ndarray(shape=(5), dtype=int, buffer=np.array(neigh[0:5]))
                else:
                    neigh = np.ndarray(shape=(count), dtype=int, buffer=np.array(neigh[0:count]))

                #neigh = neigh[dn < radius]


                if len(neigh) > 1:
                    In = self.filteredData[xs[neigh].astype('i'), ys[neigh].astype('i')]
                    mi = In.argmax()

                    xj = neigh[mi]
                    flag = True
                    for j in xrange(len(neigh)):
                        flag = flag and (neigh[j] >= neigh[0])
                    # if xj >= i and flag:
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

    def _distance(self, lis1, lis2, mode='cityblock'):
        dis = 0
        lis1 = np.array(lis1)
        lis2 = np.array(lis2)
        if len(lis1) != len(lis2):
            print 'Wrong input!'
        else:
            if mode == 'cityblock':
                dis = sum(abs(lis1 - lis2))
            elif mode == 'Euclid':
                for i in xrange(len(lis1)):
                    dis += math.pow(lis1[i] - lis2[i],2)
                dis = math.sqrt(dis)
        return dis

    def __discardClumped(self, xs, ys, radius=4):
        if len(xs) < 2:
            return xs, ys

        kdt = ckdtree.cKDTree(numpy.array([xs,ys]).T)

        xsd = []
        ysd = []

        for i in xrange(len(xs)):
            xi = xs[i]
            yi = ys[i]
            #neigh = kdt.query_ball_point([xi,yi], radius)

            dn, neigh = kdt.query(numpy.array([xi,yi]), 2)
            print dn

            if (dn[1] > radius):
                xsd.append(xi)
                ysd.append(yi)

        print len(xsd)

        return numpy.array(xsd), numpy.array(ysd)

    def FindObjects(self,
                    thresholdFactor,
                    debounceRadius = 4,
                    maskEdgeWidth = 5,
                    discardClumpRadius = 0):
        """Finds point-like objects by subjecting the data to a band-pass filtering (as defined when
        creating the identifier) followed by z-projection and a thresholding procedure where the
        threshold is progressively decreased from a maximum value (half the maximum intensity in the image) to a
        minimum defined as [thresholdFactor]*the mode (most frequently occuring value,
        should correspond to the background) of the image. The number of steps can be given as
        [numThresholdSteps], with defualt being 5 when filterMode="fast" and 10 for filterMode="good".
        At each step the thresholded image is blurred with a Gaussian of radius [blurRadius] to
        approximate the image of the points found in that step, and subtracted from the original, thus
        removing the objects from the image such that they are not detected at the lower thresholds.
        This allows the detection of objects which are relatively close together and spread over a
        large range of intenstities. A binary mask [mask] may be applied to the image to specify a region
        (e.g. a cell) in which objects are to be detected.

        A copy of the filtered image is saved such that subsequent calls to FindObjects with, e.g., a
        different thresholdFactor are faster."""

        #save a copy of the parameters.
        self.lowerThreshold = thresholdFactor

        #clear the list of previously found points
        del self[:]

        # run kernel in device
        self.clFilteredData = numpy.zeros(pixelCount, 'float64')
        cl.enqueue_copy(commandQueue, cl.memImage, self.clData)
        cl.enqueue_copy(commandQueue, cl.memCandiCount, np.array(candiCount)).wait()
        cl.colFilterKernel.enqueue_nd_range(cl.filterKernelGlobalSize, commandQueue, cl.filterKernelLocalSize)
        cl.rowFilterKernel.enqueue_nd_range(cl.filterKernelGlobalSize, commandQueue, cl.filterKernelLocalSize)
        cl.labelInitKernel.enqueue_nd_range(cl.labelKernelGlobalSize, commandQueue)
        for i in xrange(maxpass):
            cl.labelPropagateKernel.enqueue_nd_range(cl.labelKernelGlobalSize, commandQueue)
        cl.candiInitKernel.enqueue_nd_range(cl.candiInitKernelGlobalSize, commandQueue)
        cl.candiSumKernel.enqueue_nd_range(cl.candiInitKernelGlobalSize, commandQueue)
        cl.candiMainKernel.enqueue_nd_range(cl.candiMainKernelGlobalSize, commandQueue)
        cl.debounceCandiKernel.enqueue_nd_range(cl.candiMainKernelGlobalSize, commandQueue)

        cl.enqueue_copy(commandQueue, self.clFilteredData, cl.memFilteredImage).wait()

        #do filtering
        self.filteredData = self.__FilterDataFast()
        self.filteredData *= (self.filteredData > 0)

        #apply mask
        self.clFilteredData = self.clFilteredData.reshape([md.imageHeight, md.imageWidth])
        maskedFilteredData = self.filteredData

        #manually mask the edge pixels
        if maskEdgeWidth and self.filteredData.shape[1] > maskEdgeWidth:
            maskedFilteredData[:, :maskEdgeWidth] = 0
            maskedFilteredData[:, -maskEdgeWidth:] = 0
            maskedFilteredData[-maskEdgeWidth:, :] = 0
            maskedFilteredData[:maskEdgeWidth, :] = 0

        print np.linalg.norm(self.clFilteredData-maskedFilteredData)
        X,Y = numpy.mgrid[0:maskedFilteredData.shape[0], 0:maskedFilteredData.shape[1]]

        #store x, y, and thresholds
        xs = []
        ys = []
        ts = []

        im = maskedFilteredData
        imt = im > self.lowerThreshold
        (labeledPoints, nLabeled) = ndimage.label(imt)

        objSlices = ndimage.find_objects(labeledPoints)

        #loop over objects
        for i in range(nLabeled):
            #measure position
            imO = im[objSlices[i]]
            x = (X[objSlices[i]]*imO).sum()/imO.sum()
            y = (Y[objSlices[i]]*imO).sum()/imO.sum()

            #and add to list
            #self.append(OfindPoint(x,y,detectionThreshold=self.lowerThreshold))
            xs.append(x)
            ys.append(y)
            ts.append(self.lowerThreshold)


        xs = numpy.array(xs)
        ys = numpy.array(ys)

        if discardClumpRadius > 0:
            print 'ditching clumps'
            xs, ys = self.__discardClumped(xs, ys, discardClumpRadius)

        import time
        tStart = time.time()
        xs, ys = self.__Debounce(xs, ys, debounceRadius)
        tEnd = time.time()
        # print('>>> Elapsed time for analyzing frames is %.2f ms' \
        #       % ((tEnd - tStart) * 1000))
        # xs, ys = self.Debounce(xs, ys, debounceRadius)

        # for i in xrange(len(xs1)):
        #     if xs1[i] != xs2[i]:
        #         print i,':', xs1[i], xs2[i]

        for x, y, t in zip(xs, ys, ts):
            self.append(OfindPoint(x,y,t))


        #create pseudo lists to allow indexing along the lines of self.x[i]
        self.x = PseudoPointList(self, 'x')
        self.y = PseudoPointList(self, 'y')
