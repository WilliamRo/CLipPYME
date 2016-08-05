import  clip.cl as cl
import numpy
import math
import ctypes as ct

# region : Compare method

def compareMatrix(mat1, mat2, head, tol=0.001, view=True):
    if not view:
        return True
    if (mat1.shape[0] != mat2.shape[0]) or (mat1.shape[1] != mat2.shape[1]):
        print head, 'has diffrent shape!'
        return False
    errorCount = 0
    for i in xrange(mat1.shape[0]):
        for j in xrange(mat1.shape[1]):
            if abs(mat1[i, j] - mat2[i, j]) > tol \
                    or numpy.isnan(mat1[i, j]) \
                    or numpy.isnan(mat2[i, j]):
                errorCount += 1
    if errorCount is 0:
        print  head, 'is right!'
    else:
        print head, 'is wrong!(error count is %d)' % errorCount

# endregion

# region : Define metadata

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
lowFilterLength = 9
highFilterLength = 25

#precompute our filter weights
weightsLowpass = calc_gauss_weights(filtRadLowpass)
weightsHighpass = calc_gauss_weights(filtRadHighpass)
weightsLow = numpy.zeros(highFilterLength,'f')
weightsLow[8:17] = weightsLowpass
# weights = numpy.zeros([highFilterLength , highFilterLength], dtype='f')
# for i in xrange(highFilterLength):
#     for j in xrange(highFilterLength):
#         weights[i,j] = weightsLow[i]*weightsLow[j] - \
#                        weightsHighpass[i]*weightsHighpass[j]
# weights = weights.reshape([highFilterLength*highFilterLength])
weights = weightsLow

Ftype = ct.c_float
class clMetadata(ct.Structure):
    _fields_ = [("imageWidth", ct.c_int32),
                ("imageHeight", ct.c_int32),
                ("imageDepth", ct.c_int32),
                ("voxelSizeX", Ftype),
                ("voxelSizeY", Ftype),
                ("voxelSizeZ", Ftype),
                ("is2DImage", ct.c_bool),
                ("filterRadiusLowpass", ct.c_int32),
                ("filterRadiusHighpass", ct.c_int32),
                ("lowFilterLength", ct.c_int32),
                ("highFilterLength", ct.c_int32),
                ("weightsLow", Ftype*lowFilterLength),
                ("weightsHigh", Ftype*highFilterLength),
                ("weights", Ftype*(highFilterLength)),
                ("cameraOffset", Ftype),
                ("cameraNoiseFactor", Ftype),
                ("cameraElectronsPerCount", Ftype),
                ("cameraTrueEMGain", Ftype),
                ("debounceRadius", Ftype),
                ("threshold", Ftype),
                ("fudgeFactor", Ftype),
                ("maskEdgeWidth", ct.c_int32),
                ("SNThreshold", ct.c_bool),
                ("bgStartInd", ct.c_int),
                ("bgEndInd", ct.c_int),
                ("maxFrameNum", ct.c_int),
                ("minBgIndicesLen", ct.c_int),
                ("roiHalfSize", ct.c_int)]

# endregion

# region : Profile run time

def ProfileKernelRunTime(evt):
    if isinstance(evt, list):
        time = []
        if len(evt) is 0:
            return 0
        for e in evt:
            time.append(e.get_profiling_info(CL_PROFILING_COMMAND_END) \
                        - e.get_profiling_info(CL_PROFILING_COMMAND_START))
        return  time
    else:
        return evt.get_profiling_info(CL_PROFILING_COMMAND_END) - \
        evt.get_profiling_info(CL_PROFILING_COMMAND_START)

def ProfileKernelTotalTime(evt):
    return evt.get_profiling_info(CL_PROFILING_COMMAND_END) - \
        evt.get_profiling_info(CL_PROFILING_COMMAND_QUEUED)

def PrintTime(time, head, timeFormat = 'NanoSecond'):
    if timeFormat == 'NanoSecond':
        print '>>> ', head, 'run time is %.3f us' % time/1e3
    elif timeFormat == 'Second':
        print '>>> ', head, 'run time is %.3f s' % time * 1e3

def PrintKernelRunTime(evt, head):
    PrintTime(ProfileKernelRunTime(evt), head)

def PrintHead(headList = []):
    headLen = 0
    for i in headList:
        headLen += (len(i)+5)
    headLen += 1
    partitionLine = []
    for i in xrange(headLen):
        partitionLine.append('-')
    print ''.join(partitionLine)
    ret2 = ['|']
    for i in headList:
        ret2.append('  ')
        ret2.append(i)
        ret2.append('  |')
    print ''.join(ret2)
    print ''.join(partitionLine)

def PrintTable(content, headList):
    if len(content) != len(headList):
        ValueError('Content do not match the head list!')
    else:
        lineString = []
        for i in xrange(len(content)):
            item = ''.join(str(content[i]))
            itemLen = len(item)
            headLen = len(headList[i])+4
            spaceLen = int((headLen - itemLen) / 2)
            lineString.append('|')
            for j in xrange(spaceLen):
                lineString.append(' ')
            lineString.append(str(content[i]))
            for j in xrange(spaceLen):
                lineString.append(' ')
            if spaceLen*2 + itemLen != headLen:
                lineString.append(' ')
        lineString.append('|')
        print  ''.join(lineString)
        headLen = 0
        for i in headList:
            headLen += (len(i)+5)
        headLen += 1
        partitionLine = []
        for i in xrange(headLen):
            partitionLine.append('-')
        print ''.join(partitionLine)


def PrintKernelInformation(head,
                           timeList,
                           headList,
                           listtype = 'event',
                           timeFormat = 'NanoSecond'):
    content = []
    if listtype is 'event':
        time = ProfileKernelRunTime(timeList)
    elif listtype is 'time':
        time = timeList
    length = len(time)
    if timeFormat is 'NanoSecond':
        for i in xrange(length):
            time[i] = time[i] / 1e6;
    elif timeFormat is 'Second':
        for i in xrange(length):
            time[i] = time[i] * 1e3;
    totalTime = round(sum(time), 5)
    avgTime = round(totalTime / length, 5)
    maxTime = round(max(time), 5)
    minTime = round(min(time), 5)
    content.append(head)
    content.append(length)
    content.append(totalTime)
    content.append(avgTime)
    content.append(maxTime)
    content.append(minTime)
    PrintTable(content, headList)

def PrintTotalInformation():
    global copyIntoDeviceEvent, subBgEvent, colFilterEvent, rowFilterEvent,\
    labelInitEvent, labelMainEvent, candiInitEvent, candiObjEvent, \
    candiMainEvent, debounceCandiEvent, copyIntoHostEvent, cpuStartTime,\
    cpuStartTime, gpuTotalTime

    for i in xrange(len(subBgEvent)):
        gpuTotalTime.append(debounceCandiEvent[i].get_profiling_info(CL_PROFILING_COMMAND_END)\
                            - copyIntoDeviceEvent[i].get_profiling_info(CL_PROFILING_COMMAND_START))

    headList = ['Kernel Name', '# of Calls', 'Total Time(ms)', 'Avg Time(ms)',\
                'Max Time(ms)', 'Min Time(ms)']
    PrintHead(headList)

    PrintKernelInformation('copyIntoDevice', copyIntoDeviceEvent, headList)
    PrintKernelInformation('initBuffer', initBufferEvent, headList)
    PrintKernelInformation('subBg', subBgEvent, headList)
    PrintKernelInformation('colFilter', colFilterEvent, headList)
    PrintKernelInformation('rowFilter', rowFilterEvent, headList)
    PrintKernelInformation('labelInit', labelInitEvent, headList)
    PrintKernelInformation('labelMain', labelMainEvent, headList)
    PrintKernelInformation('candiInit', candiInitEvent, headList)
    PrintKernelInformation('candiObj', candiObjEvent, headList)
    PrintKernelInformation('candiMain', candiMainEvent, headList)
    PrintKernelInformation('debounceCandi', debounceCandiEvent, headList)
    PrintKernelInformation('fitInit', fitInitEvent, headList)
    # PrintKernelInformation('copyIntoHost', copyIntoHostEvent, headList)
    PrintKernelInformation('gpuTotal', gpuTotalTime, headList, 'time')
    PrintKernelInformation('cpuTotal',
                           numpy.array(cpuEndTime)-numpy.array(cpuStartTime),
                           headList, 'time', timeFormat='Second')

# endregion

# region : Create ofind opencl enviroment

# region : define event list

copyIntoDeviceEvent = []
initBufferEvent = []
subBgEvent = []
colFilterEvent = []
rowFilterEvent = []
labelInitEvent = []
labelMainEvent = []
candiInitEvent = []
candiObjEvent = []
candiMainEvent = []
debounceCandiEvent = []
fitInitEvent = []
copyIntoHostEvent = []

# record cpu time
cpuStartTime = []
cpuEndTime = []

# record gpu time
gpuTotalTime = []

# endregion

# region : define some opencl macro

CL_PROFILING_COMMAND_QUEUED = 0x1280
CL_PROFILING_COMMAND_SUBMIT = 0x1281
CL_PROFILING_COMMAND_START = 0x1282
CL_PROFILING_COMMAND_END = 0x1283
CL_PROFILING_COMMAND_COMPLETE = 0x1284

# endregion

# region : create metadata

wl = (Ftype * lowFilterLength)()
for i in xrange(lowFilterLength):
    wl[i] = Ftype(weightsLowpass[i])
wh = (Ftype * highFilterLength)()
for i in xrange(highFilterLength):
    wh[i] = Ftype(weightsHighpass[i])
w = (Ftype * (highFilterLength))()
for i in xrange(highFilterLength):
    w[i] = Ftype(weights[i])
md = clMetadata(170,140,1,0.08,0.08,0.10,True,4,12,9,25,wl,wh,w,\
                0.0,1.0,1.0,1.0,5.0,1.0,1.0,5,True,-20,0,100,1,5)


# endregion : create metadata

# region : define other parameters

pixelCount = md.imageWidth * md.imageHeight
maxCount = 5000
maxRegionPointNum = 2000
maxpass = 10
candiCount = 0
typeIntSize = ct.sizeof(ct.c_int32)
typeShortSize = ct.sizeof(ct.c_int16)
typeFloatSize = ct.sizeof(Ftype)
typeDoubleSize = ct.sizeof(ct.c_double)
typeBoolSize = ct.sizeof(ct.c_bool)

# endregion

# region : get device and default commandQueue
context = cl.context
device = context.default_device
commandQueue = context.default_queue
program = cl.program
ma = cl.mem_access_mode

# endregion

# region : create kernel

cl.initBuffer = program.initBuffer
cl.colFilter = program.colFilterImage
cl.rowFilter = program.rowFilterImage
cl.subBg = program.subBgAndCalcSigmaThres
cl.labelInit = program.labelInit
cl.labelMain = program.labelMain
cl.labelSync = program.labelSync
cl.candiInit = program.calcCandiPosiInit
cl.getObject = program.getCandiPosiObj
cl.candiMain = program.caclCandiPosiMain
cl.debCandi = program.debounceCandiPosi
cl.fitInit = program.fitInit

# endregion : create kernel

# region : create memory buffer

cl.memMetadata = context.create_buffer(ma.READ_ONLY, ct.sizeof(md))
cl.memImageStack = context.create_buffer(ma.READ_WRITE, md.maxFrameNum * pixelCount * typeFloatSize)
cl.memBufferIndex = context.create_buffer(ma.READ_WRITE, typeIntSize)
cl.memImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeFloatSize)
cl.memPaddedImage = context.create_buffer(ma.READ_WRITE, \
            (md.imageWidth+highFilterLength-1)*(md.imageHeight+highFilterLength-1)*typeFloatSize)
cl.memFilteredImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeFloatSize)
cl.memHighFilteredImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeFloatSize)
cl.memLowFilteredImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeFloatSize)
cl.memSigmaMap = context.create_buffer(ma.READ_WRITE, pixelCount * typeFloatSize)
cl.memThresholdMap = context.create_buffer(ma.READ_WRITE, pixelCount * typeFloatSize)
cl.memVarianceMap = context.create_buffer(ma.READ_WRITE, pixelCount * typeFloatSize)
cl.memSyncIndex = context.create_buffer(ma.READ_WRITE, 2 * typeIntSize)
cl.memPass = context.create_buffer(ma.READ_WRITE, maxpass * typeIntSize)
cl.memBinaryImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeShortSize)
cl.memLabeledImage = context.create_buffer(ma.READ_WRITE, pixelCount * typeIntSize)
cl.memCandiPosi = context.create_buffer(ma.READ_WRITE, 2 * maxCount * typeFloatSize)
cl.memTempCandiPosi = context.create_buffer(ma.READ_WRITE, 2 * maxCount * typeFloatSize)
cl.memCandiRegion = context.create_buffer(ma.READ_WRITE, 4 * maxCount* typeIntSize)
cl.memCandiCount = context.create_buffer(ma.READ_WRITE, 2 * typeIntSize)
cl.memXGrid = context.create_buffer(ma.READ_WRITE, maxCount * (2*md.roiHalfSize+1) * typeFloatSize)
cl.memYGrid = context.create_buffer(ma.READ_WRITE, maxCount * (2*md.roiHalfSize+1) * typeFloatSize)
cl.memStartPara = context.create_buffer(ma.READ_WRITE, 7 * maxCount * typeFloatSize)

# endregion : create memory buffer

# region : set kernel arguments and wotk item dimension

cl.initBuffer.set_arg(0, cl.memBufferIndex)
cl.initBuffer.set_arg(1, cl.memSyncIndex)
cl.initBuffer.set_arg(2, cl.memCandiCount)
cl.initBuffer.set_arg(3, cl.memPass)
cl.initBufferGlobalDim = [2,1,1]
cl.initBufferLocalDim = [1,2,1]

cl.colFilter.set_arg(0, cl.memImage)
cl.colFilter.set_arg(1, cl.memLowFilteredImage)
cl.colFilter.set_arg(2, cl.memHighFilteredImage)
cl.colFilter.set_arg(3, cl.memMetadata)

cl.rowFilter.set_arg(0, cl.memLowFilteredImage)
cl.rowFilter.set_arg(1, cl.memHighFilteredImage)
cl.rowFilter.set_arg(2, cl.memFilteredImage)
cl.rowFilter.set_arg(3, cl.memBinaryImage)
cl.rowFilter.set_arg(4, cl.memThresholdMap)
cl.rowFilter.set_arg(5, cl.memMetadata)

cl.subBg.set_arg(0, cl.memImageStack)
cl.subBg.set_arg(1, cl.memImage)
cl.subBg.set_arg(2, cl.memSigmaMap)
cl.subBg.set_arg(3, cl.memThresholdMap)
cl.subBg.set_arg(4, cl.memVarianceMap)
cl.subBg.set_arg(5, cl.memBufferIndex)
cl.subBg.set_arg(6, cl.memMetadata)
cl.filterGlobalDim = [(md.imageHeight + 31)&~31, (md.imageWidth + 31)&~31, 1]
cl.filterGlobalDimVec = [((md.imageHeight + 31)&~31),
                             ((md.imageWidth + 31)&~31)/2, 1]
# cl.filterGlobalDim = [md.imageHeight, md.imageWidth, 1]
# cl.filterLocalDim = [1, 256, 1]
cl.filterLocalDim = None

cl.labelInit.set_arg(0, cl.memLabeledImage)
cl.labelInit.set_arg(1, cl.memBinaryImage)
cl.labelInit.set_arg(2, cl.memMetadata)

cl.labelMain.set_arg(0, cl.memLabeledImage)
cl.labelMain.set_arg(1, cl.memMetadata)
cl.labelMain.set_arg(2, cl.memSyncIndex)
cl.labelMain.set_arg(3, cl.memPass)
cl.labelGlobalDim = [(md.imageHeight + 31)&~31, (md.imageWidth + 31)&~31, 1]
# cl.labelLocalDim = [1, 256, 1]
cl.labelLocalDim = None

cl.labelSync.set_arg(0, cl.memSyncIndex)
labelSyncGlobalDim = [1,1,1]
# labelSyncLocalDim = [1,1,1]
labelSyncLocalDim = None

cl.candiInit.set_arg(0, cl.memLabeledImage)
cl.candiInit.set_arg(1, cl.memCandiRegion)
cl.candiInit.set_arg(2, cl.memCandiCount)
cl.candiInit.set_arg(3, cl.memMetadata)

cl.getObject.set_arg(0, cl.memLabeledImage)
cl.getObject.set_arg(1, cl.memCandiRegion)
cl.getObject.set_arg(2, cl.memCandiCount)
cl.getObject.set_arg(3, cl.memMetadata)
cl.candiInitGlobalDim = [(md.imageHeight + 31)&~31, (md.imageWidth + 31)&~31, 1]
# cl.candiInitKernelLocalSize = [1, 1, 1]
cl.candiInitKernelLocalSize  = None

cl.candiMain.set_arg(0, cl.memFilteredImage)
cl.candiMain.set_arg(1, cl.memCandiRegion)
cl.candiMain.set_arg(2, cl.memTempCandiPosi)
cl.candiMain.set_arg(3, cl.memCandiCount)
cl.candiMain.set_arg(4, cl.memMetadata)
cl.candiMainGlobalDim = [(maxCount+31)&~31, 1, 1]
# cl.candiMainLocalDim = [1, 256, 1]
cl.candiMainLocalDim = None

cl.debCandi.set_arg(0, cl.memFilteredImage)
cl.debCandi.set_arg(1, cl.memCandiPosi)
cl.debCandi.set_arg(2, cl.memTempCandiPosi)
cl.debCandi.set_arg(3, cl.memCandiCount)
cl.debCandi.set_arg(4, cl.memMetadata)

cl.fitInit.set_arg(0, cl.memImageStack)
cl.fitInit.set_arg(1, cl.memImage)
cl.fitInit.set_arg(2, cl.memBufferIndex)
cl.fitInit.set_arg(3, cl.memCandiPosi)
cl.fitInit.set_arg(4, cl.memCandiCount)
cl.fitInit.set_arg(5, cl.memXGrid)
cl.fitInit.set_arg(6, cl.memYGrid)
cl.fitInit.set_arg(7, cl.memStartPara)
cl.fitInit.set_arg(8, cl.memMetadata)
roiSize = (2 * md.roiHalfSize + 1 + 15)&~15
cl.fitInitGlobalDim = [roiSize, (maxCount+15)&~15]
cl.fitInitLocalDim = [roiSize, 1]


# endregion : set kernel arguments and wotk item dimension

# write metadata into device
cl.enqueue_copy(commandQueue, cl.memMetadata, md)

# endregion : CLOFIND