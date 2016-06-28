"""
    testWorker.py
    Script for test fitWorker.py
"""
import os
import tables
import time

import numpy as np
from tkFileDialog import askopenfilename

from PYME.gohlke.tifffile import TIFFfile
from PYME.Acquire.MetaDataHandler import NestedClassMDHandler

from StandaloneWorker.fitWorker import Worker
from StandaloneWorker.remFitBufPro import fitTask

print('>>> Modules imported')


# region : Utilities

def getErrorText(index, res, std, res_f, std_f):
    global colnames, tol
    text = 'row[%d]\n' % index
    txtRes = '   res: '
    txtStd = 'stdres: '
    txtDel = ' delta: '
    for i in range(len(colnames)):
        d = abs((std[i] - res[i]) / std[i])
        if d < tol: continue
        txtRes += '%s(%f)  ' % (colnames[i], res[i])
        txtStd += '%s(%f)  ' % (colnames[i], std[i])
        txtDel += '%s(%.1f%%)  ' % (colnames[i], d * 100)
    txtRes += '-> ||r|| = %.10f' % res_f
    txtStd += '-> ||r|| = %.10f' % std_f
    return text + txtRes + '\n' + txtStd + '\n' + txtDel + '\n\n'


# endregion : Utilities

# region : Initialize parameters

# > Input parameter
numFramesToAnalyze = 1

fitMod = 'LatGaussFitFR'
threshold = 1
bIndiceRange = [-20, 0]

# > File names
filename = r'data\100x140x170.tif'

# > Metadata
metadata = NestedClassMDHandler()

metadata.setEntry('voxelsize.x', 0.08)
metadata.setEntry('voxelsize.y', 0.08)
metadata.setEntry('voxelsize.z', 0.2)
metadata.setEntry('voxelsize.units', 'um')

metadata.setEntry('Camera.ADOffset', 0)
metadata.setEntry('Camera.ReadNoise', 0)
metadata.setEntry('Camera.NoiseFactor', 1)
metadata.setEntry('Camera.ElectronsPerCount', 1)
metadata.setEntry('Camera.TrueEMGain', 1)

metadata.setEntry('Analysis.FitModule', fitMod)

# > Runtime arguments
results = []

print('>>> Parameters initialized')

# endregion : Initialize parameters

# region : Prepare for fitting

# > check file
if not os.path.exists(filename) or not 'filename' in dir():
    filename = askopenfilename()
    if not os.path.exists(filename) or not 'filename' in dir():
        exit()

# > file names
fn = filename.split('\\')[-1].split('.')[0]
resFilename = r'DH5Files\results_%s_%d.h5r' % (fn, numFramesToAnalyze)
stdresFilename = r'DH5Files\stdResults_%s_%d.h5r' % (
fn, numFramesToAnalyze)
veriFilename = r'DH5Files\veri_%s_%d.txt' % (fn, numFramesToAnalyze)

# > read frames with shape (slices, width, height) from file
frames = TIFFfile(filename).asarray()
numSlices = frames.shape[0]
print('>>> Loaded frames from %s' % filename)

# > generate indices
indices = range(0, numFramesToAnalyze)
indices = [i % numSlices for i in indices]

print('>>> Preparation done')

# endregion : Prepare for fitting

# region : Fitting

# > create worker
worker = Worker(frames, threshold, metadata, fitMod, True)
print('>>> Fitting...')

tStart = time.time()
for i in indices:
    # >> generate background indices
    bgi = range(max(0, i + bIndiceRange[0]),
                min(numSlices - 1, i + bIndiceRange[1]))

    # >> do fitting
    res = worker.fit(i, bgi)

    results.append(res)

tEnd = time.time()
print('>>> Elapsed time for analyzing %d frames is %.2f ms' \
      % (numFramesToAnalyze, (tEnd - tStart) * 1000))

# > prepare to save results
resFile = tables.openFile(resFilename, 'w')

resFile.createTable(resFile.root, 'FitResults', np.hstack(results),
                    filters=tables.Filters(complevel=5, shuffle=True))

resFile.close()
print('>>> Saved results to file %s' % resFilename)

# endregion : Fitting

# region : Verify

# region : Verify : Check standard results

if not os.path.exists(stdresFilename):
    stdResults = []
    print('>>> Creating standard results...')
    for i in indices:
        # > generate background indices
        bgi = range(max(0, i + bIndiceRange[0]),
                    min(numSlices - 1, i + bIndiceRange[1]))

        moduleName = 'TiffDataSource'
        ft = fitTask(filename, i, threshold, metadata,
                     fitMod, moduleName, bgi, True)

        res = ft()
        stdResults.append(res.results)

    # > save results
    stdresFile = tables.openFile(stdresFilename, 'w')

    stdresFile.createTable(stdresFile.root, 'FitResults',
                           np.hstack(stdResults),
                           filters=tables.Filters(complevel=5,
                                                  shuffle=True))

    stdresFile.close()
    print('>>> Standard results created')

# endregion : Verify : Check standard results

# > generate verification file
print('>>> Verifying...')
stdresFile = tables.openFile(stdresFilename, 'r')
resFile = tables.openFile(resFilename, 'r')
veriFile = open(veriFilename, 'w')

# > get result tables and row numbers
stdres = stdresFile.root.FitResults
stdresNRows = stdres.nrows

res = resFile.root.FitResults
resNRows = res.nrows

if stdresNRows != resNRows:
    print('!!! nrows(%d) != standard nrows(%d)' % \
          (resNRows, stdresNRows))

nrows = min(resNRows, stdresNRows)

# > get column names and length
colnames = res.description.fitResults._v_names
ncols = len(colnames)

# > write verification file
errCount = 0
maxCount = 100
tol = 1e-3

line = 'Parameters: [%s]\ntol = %.1f%%\n' % (
', '.join(colnames), tol * 100)
line += 'res row number: %d, stdres row number: %d' % (
resNRows, stdresNRows)
veriFile.writelines(line + '\n\n')

# > scan each row
for i in range(0, nrows):
    for j in range(0, ncols):
        d = abs((stdres[i][1][j] - res[i][1][j]) / stdres[i][1][j])
        if d > tol:
            errCount += 1
            content = getErrorText(i, res[i][1], stdres[i][1],
                                   res[i][5], stdres[i][5])
            veriFile.writelines(content)
            break

    if errCount >= maxCount:
        break

# > close files
stdresFile.close()
resFile.close()
veriFile.close()

if errCount == 0:
    print('>>> Perfect!')
else:
    print('>>> Created verification file %s, errCount: %d (max: %d)' \
          % (veriFilename, errCount, maxCount))

if True and errCount: os.startfile(veriFilename)

# endregion : Verify

# Wm: 11th Test Ha
