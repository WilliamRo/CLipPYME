#!/usr/bin/python

##################
# clOfind.py
#
##################

from ofindPara import *
#import pylab


class ObjectIdentifier(list):
    def __init__(self, data):
        """Creates an Identifier object to be used for object finding, takes a 2D or 3D slice
        into a data stack (data), and a filtering mode (filterMode, one of ["fast", "good"])
        where "fast" performs a z-projection and then filters, wheras "good" filters in 3D before
        projecting. The parameters filterRadiusLowpass and filterRadiusHighpass control the bandpass filter
        used to identify 'point-like' features. filterRadiusZ is the radius used for the axial smoothing filter"""

        self.clData = data.astype('f').reshape([pixelCount])
        self.candiCount = numpy.zeros(2, 'int32')

    def RunKernel(self, index):
        copyIntoDeviceEvent.append(
            cl.enqueue_copy(commandQueue,
                            cl.memImageStack,
                            self.clData,
                            device_offset=index * pixelCount * typeFloatSize,
                            is_blocking=False)
        )
        cl.enqueue_copy(commandQueue,
                        cl.memBufferIndex,
                        numpy.array(index, 'int32'),
                        is_blocking=False)

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
                                              commandQueue,
                                              cl.labelLocalDim)
            )
            cl.labelSync.enqueue_nd_range(labelSyncGlobalDim,
                                          commandQueue)

        # calculate candidate positions
        cl.sortInit.enqueue_nd_range(cl.candiInitGlobalDim,
                                     commandQueue)
        candiInitEvent.append(
            cl.candiInit.enqueue_nd_range(cl.candiInitGlobalDim,
                                          commandQueue,
                                          cl.candiInitKernelLocalSize)
        )
        candiObjEvent.append(
            cl.getObject.enqueue_nd_range(cl.candiInitGlobalDim,
                                          commandQueue,
                                          cl.candiInitKernelLocalSize)
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

        # fit initialization
        fitInitEvent.append(
            cl.fitInit.enqueue_nd_range(cl.fitInitGlobalDim,
                                        commandQueue,
                                        cl.fitInitLocalDim)
        )

        # read data from device
        # cl.enqueue_copy(commandQueue, self.candiCount, cl.memCandiCount,
        #                 is_blocking=False)
        # cl.enqueue_copy(commandQueue, self.clSigma, cl.memSigmaMap,
        #                 is_blocking=False)

        commandQueue.flush()
        commandQueue.finish()

    def FindObjects(self, index):
        """."""
        # run kernel in device
        self.RunKernel(index)

        self.ofdLen = self.candiCount[1]

