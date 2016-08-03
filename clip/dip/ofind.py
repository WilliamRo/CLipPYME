#!/usr/bin/python

##################
# ofind.py
#
##################

from scipy import ndimage
from scipy.ndimage import _nd_image, _ni_support
from scipy.spatial import ckdtree
import  numpy as np
import time
from ofindPara import *
import pyopencl as pycl
#import pylab

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
    def __init__(self, data, bg):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data), and a filtering mode (filterMode, one of ["fast", "good"])
        where "fast" performs a z-projection and then filters, wheras "good" filters in 3D before
        projecting. The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. filterRadiusZ is the radius used for the axial smoothing filter"""
        global filtRadLowpass, filtRadHighpass, weightsLowpass, weightsHighpass

        self.clData = data.astype('f').reshape([pixelCount])
        self.data = data.astype('f')
        self.bg = bg

    def __FilterData2D(self,data):
        mode = _ni_support._extend_mode_to_code("reflect")
        #lowpass filter to suppress noise
        output, a = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, weightsLowpass, 0, output, mode, 0,0)
        # print '1st low filter result in py is %.10f. ' % a[13, 112]
        # tmpSum = 0.0
        # for i in range(-4,5,1):
        #     tmpSum = tmpSum + a[13, 112+i]*weightsLowpass[i+4]
        # print 'tmpSum:', tmpSum
        _nd_image.correlate1d(output, weightsLowpass, 1, output, mode, 0,0)
        # print '2nd low filter result in py is %f. ' % a[13, 112]

        #lowpass filter again to find background
        output, b = _ni_support._get_output(None, data)
        _nd_image.correlate1d(data, weightsHighpass, 0, output, mode, 0,0)
        # print '1st high filter result in py is %.10f. ' % b[13, 112]
        _nd_image.correlate1d(output, weightsHighpass, 1, output, mode, 0,0)
        # print '2nd high filter result in py is %f. ' % b[13, 112]

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
                    index,
                    thresholdFactor,
                    debounceRadius = 4,
                    maskEdgeWidth = 5,
                    discardClumpRadius = 0):
        """."""

        #save a copy of the parameters.
        self.lowerThreshold = thresholdFactor

        #clear the list of previously found points
        del self[:]

        # run kernel in device
        self.clFilteredData = numpy.zeros(pixelCount, 'float32')
        self.candiPosi = numpy.zeros([maxCount*2], 'float32')
        self.candiCount = numpy.array(0)
        self.clBinaryImage = numpy.zeros(pixelCount, 'uint16')
        self.clLabel = numpy.zeros(pixelCount, 'int32')
        self.clThreshold = numpy.zeros(pixelCount, 'float32')
        self.clDebounceCandi = numpy.zeros(maxCount*2, 'float32')

        for i in xrange(10):
            if i != 0:
                cpuStartTime.append(time.time())
            # copy data into device

            if i is 0:
                copyIntoDeviceEvent.append(
                    cl.enqueue_copy(commandQueue,
                                    cl.memImageStack,
                                    self.clData,
                                    device_offset=index * pixelCount * typeFloatSize,
                                    is_blocking=False)
                )
                cl.enqueue_copy(commandQueue,
                                cl.memBufferIndex,
                                numpy.array(index,'int32'),
                                is_blocking = False)
                cl.enqueue_copy(commandQueue,
                                cl.memCandiCount,
                                np.array(candiCount),
                                is_blocking = False)
            else:
                initBufferEvent.append(
                    cl.initBuffer.enqueue_nd_range(cl.initBufferGlobalDim,
                                                   commandQueue)
                )

            # subtract bg and get sigma map threshold map
            subBgEvent.append(
                cl.subBg.enqueue_nd_range(cl.filterGlobalDim,
                                          commandQueue,
                                          cl.filterLocalDim)
            )

            # do filtering
            colFilterEvent.append(
                cl.colFilter.enqueue_nd_range(cl.filterGlobalDim,
                                              commandQueue,
                                              cl.filterLocalDim)
            )
            rowFilterEvent.append(
                cl.rowFilter.enqueue_nd_range(cl.filterGlobalDim,
                                              commandQueue,
                                              cl.filterLocalDim)
            )

            # label image
            labelInitEvent.append(
                cl.labelInit.enqueue_nd_range(cl.labelGlobalDim,
                                              commandQueue,
                                              cl.labelLocalDim)
            )
            for j in xrange(maxpass):
                labelMainEvent.append(
                    cl.labelMain.enqueue_nd_range(cl.labelGlobalDim,
                                                  commandQueue)
                )
                cl.labelSync.enqueue_nd_range(labelSyncGlobalDim,
                                              commandQueue)


            # calculate candidate positions
            candiInitEvent.append(
                cl.candiInit.enqueue_nd_range(cl.candiInitGlobalDim,
                                              commandQueue)
            )
            candiObjEvent.append(
                cl.getObject.enqueue_nd_range(cl.candiInitGlobalDim,
                                              commandQueue)
            )
            candiMainEvent.append(
                cl.candiMain.enqueue_nd_range(cl.candiMainGlobalDim,
                                              commandQueue)
            )

            # debounce candidate position
            debounceCandiEvent.append(
                cl.debCandi.enqueue_nd_range(cl.candiMainGlobalDim,
                                             commandQueue,
                                             cl.candiMainLocalDim)
            )

            commandQueue.flush()
            commandQueue.finish()

            if i != 0:
                cpuEndTime.append(time.time())

        PrintTotalInformation()

        # copy data into host
        cl.enqueue_copy(commandQueue, self.clLabel,cl.memLabeledImage)
        cl.enqueue_copy(commandQueue, self.candiCount,cl.memCandiCount)
        cl.enqueue_copy(commandQueue, self.clDebounceCandi,cl.memCandiPosi)
        cl.enqueue_copy(commandQueue, self.clFilteredData, cl.memFilteredImage)
        cl.enqueue_copy(commandQueue, self.clBinaryImage, cl.memBinaryImage)
        cl.enqueue_copy(commandQueue, self.clThreshold, cl.memThresholdMap)
        cl.enqueue_copy(commandQueue, self.candiPosi, cl.memTempCandiPosi).wait()

        self.clFilteredData = self.clFilteredData.reshape([md.imageHeight, md.imageWidth])

        # compareMatrix(self.data - self.bg,
        #                    self.clData.reshape([md.imageHeight, md.imageWidth]),
        #                    '>>> 0. Subtracted image')
        #
        # compareMatrix(self.lowerThreshold,
        #                self.clThreshold.reshape([md.imageHeight, md.imageWidth]),
        #                '>>> 0.1. Threshold')

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

        compareMatrix(self.clFilteredData, maskedFilteredData,
                           '>>> 1. Filter result', 0.01)

        X,Y = numpy.mgrid[0:maskedFilteredData.shape[0], 0:maskedFilteredData.shape[1]]

        #store x, y, and thresholds
        xs = []
        ys = []
        ts = []

        # >>>> 2. applying threshold
        im = maskedFilteredData
        imt = im > self.lowerThreshold
        climt = self.clBinaryImage.reshape([md.imageHeight, md.imageWidth])
        compareMatrix(imt, climt, '>>> 2. Applying threshold', 0)

        # >>>> 3. labeling image
        (labeledPoints, nLabeled) = ndimage.label(imt)

        labelSet = set(self.clLabel)
        labelList = []
        for i in labelSet:
            labelList.append(i)
        labelList.sort()
        self.clLabel = self.clLabel.reshape([md.imageHeight, md.imageWidth])
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

        candiRes = numpy.zeros([nLabeled,2],dtype='f')
        clCandiRes = numpy.zeros([nLabeled,2],dtype='f')
        for i in xrange(nLabeled):
            candiRes[i,0] = xs[i]
            candiRes[i,1] = ys[i]
            clCandiRes[i,0] = self.candiPosi[2*i]
            clCandiRes[i,1] = self.candiPosi[2*i+1]
        candiRes[:,0].sort()
        candiRes[:,1].sort()
        clCandiRes[:,0].sort()
        clCandiRes[:,1].sort()
        compareMatrix(candiRes,clCandiRes,
                           '>>> 4. Calculating candidate position', tol=0.01)

        xs = numpy.array(xs)
        ys = numpy.array(ys)

        if discardClumpRadius > 0:
            print 'ditching clumps'
            xs, ys = self.__discardClumped(xs, ys, discardClumpRadius)

        xs, ys = self.__Debounce(xs, ys, debounceRadius)
        clxs = []
        clys = []
        for i in xrange(maxCount):
            if (self.clDebounceCandi[2*i] != 0) and \
                (self.clDebounceCandi[2*i+1] != 0):
                clxs.append(self.clDebounceCandi[2*i])
                clys.append(self.clDebounceCandi[2*i+1])
         # xs, ys = self.Debounce(xs, ys, debounceRadius)

        for x, y, t in zip(xs, ys, ts):
            self.append(OfindPoint(x,y,t))


        #create pseudo lists to allow indexing along the lines of self.x[i]
        self.x = PseudoPointList(self, 'x')
        self.y = PseudoPointList(self, 'y')
