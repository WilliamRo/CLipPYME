/***************************************
This file contain all kernel used in PYME OpenCL version.
This is a version 0.03
From HPEC Lab, ZheJiang University, 2016/05/26.
***************************************/

#define LOWFILTERLENGTH 9
#define HIGHFILTERLENGTH 25

#define MIN(x, y) (x < y)? x : y
#define MAX(x, y) (x > y)? x : y

#define BGCOLOUR 0
#define MAXLABELPASS 10

#define MAXCOUNT 5000
#define MAXREGIONPOINT 2000

#define IMAGEBOUNDAY(x, y) x = (x < 0)? -x - 1 : x; \
						   x = (x >= y)? 2 * y - x - 1 : x;

#define KERNELBEGIN const int globalIdx = get_global_id(0);\
					const int globalIdy = get_global_id(1);\
					const int groupIdx = get_group_id(0);\
					const int groupIdy = get_group_id(1);\
					const int globalSizeX = get_global_size(0);\
					const int globalSizeY = get_global_size(1);\
					const int localIdx = get_local_id(0);\
					const int localIdy = get_local_id(1);\
					const int groupNumx = get_num_groups(0);\
					const int groupNumy = get_num_groups(1);\

#define GETREALPOSI(t, idx, idy) t = ((idx >= 0 && idx < globalSizeX)? idx : ((idx < 0)? 0 : globalSizeX - 1)) * globalSizeY + ((idy >= 0 && idy < globalSizeY)? idy : ((idy < 0)? 0 : globalSizeY - 1))

#define SYNCGROUP 	if (localIdx == 0 && localIdy == 0 && atom_inc(&syncIndex[1]) == groupNumx * groupNumy - 1)\
						{\
							syncIndex[0] += 1;\
							syncIndex[1] = 0;\
						}

struct Metadata
{
	// Image information
	int imageWidth;
	int imageHeight;
	int imageDepth;
	double voxelSizeX; // um
	double voxelSizeY; // um
	double voxelSizeZ; // um
	bool is2DImage; // 0: 2DImage 1: 3DImage

					// Image filter setting
	int filterRadiusLowpass;
	int filterRadiusHighpass;
	int lowFilterLength;
	int highFilterLength;
	double weightsLow[LOWFILTERLENGTH];
	double weightsHigh[HIGHFILTERLENGTH];

	// Camera information
	double cameraOffset;
	double cameraNoiseFactor;
	double cameraElectronsPerCount;
	double cameraTrueEMGain;

	// Finding setting
	double debounceRadius;
	double threshold;
	double fudgeFactor;
	int maskEdgeWidth;
	bool SNThreshold;
};


double vectorMulti(double * vectorA, constant double * vectorB, int length)
{
	double sum = 0;
	for (int i = 0; i < length; i++)
	{
		sum += vectorA[i] * vectorB[i];
	}
	return sum;
}

//#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_amd_printf : enable
kernel void colFilterImage(global ushort * image,
						   global double * lowFilteredImage,
						   global double * highFilteredImage,
						   constant struct Metadata  * md)
{
	// begin
	KERNELBEGIN
	// get real position in array
	if (globalIdx == 0 && globalIdy == 0)
	{
	    //printf("the width is %d, the height is %d.\n", md->imageWidth, md->imageHeight);
        //for (int i = 0; i < md->highFilterLength; i++)
         //       printf("weightsHigh[%d]: %f\n", i, md->weightsHigh[i]);
	}
	int offset = globalIdx * md->imageWidth + globalIdy;
	if (globalIdy >= md->imageWidth || globalIdx >= md->imageHeight) return;

	// define some parameters and get parameters from metadata
	double lowData[LOWFILTERLENGTH];
	double highData[HIGHFILTERLENGTH];
	int temp;

	// low : col
	for (int i = 0; i < LOWFILTERLENGTH; i++)
	{
		temp = globalIdx + (i - md->filterRadiusLowpass);
		IMAGEBOUNDAY(temp, md->imageHeight)
		lowData[i] = image[temp * md->imageWidth + globalIdy];
	}
	lowFilteredImage[offset] = vectorMulti(lowData, md->weightsLow, LOWFILTERLENGTH);
	// high : col
	for (int i = 0; i < HIGHFILTERLENGTH; i++)
	{
		temp = globalIdx + (i - md->filterRadiusHighpass);
		IMAGEBOUNDAY(temp, md->imageHeight)
		highData[i] = image[temp * md->imageWidth + globalIdy];
	}
	highFilteredImage[offset] = vectorMulti(highData, md->weightsHigh, HIGHFILTERLENGTH);
}

kernel void rowFilterImage(global double * lowFilteredImage,
                           global   ouble * highFilteredImage,
                           global double * filteredImage,
                           global ushort * binaryImage,
                           constant struct Metadata  * md)
{
	// begin
	KERNELBEGIN
	// get real position in array
	int offset = globalIdx * md->imageWidth + globalIdy;
	if (globalIdy >= md->imageWidth || globalIdx >= md->imageHeight) return;

	// define some parameters and get parameters from metadata
	double lowData[LOWFILTERLENGTH];
	double highData[HIGHFILTERLENGTH];
	int temp;

	// low : row
	for (int i = 0; i < LOWFILTERLENGTH; i++)
	{
		temp = globalIdy + (i - md->filterRadiusLowpass);
		IMAGEBOUNDAY(temp, md->imageWidth)
		lowData[i] = lowFilteredImage[globalIdx * md->imageWidth + temp];
	}
	// high : row
	for (int i = 0; i < HIGHFILTERLENGTH; i++)
	{
		temp = globalIdy + (i - md->filterRadiusHighpass);
		IMAGEBOUNDAY(temp, md->imageWidth)
		highData[i] = highFilteredImage[globalIdx * md->imageWidth + temp];
	}
	barrier(CLK_GLOBAL_MEM_FENCE);
	// get result
	double filterResult = 0;
	filterResult = vectorMulti(lowData, md->weightsLow, LOWFILTERLENGTH) - vectorMulti(highData, md->weightsHigh, HIGHFILTERLENGTH);
	filterResult = filterResult < 0 ? 0 : filterResult;
	int mskWid = md->maskEdgeWidth;
	if (md->imageWidth > mskWid && (globalIdx < mskWid || (md->imageHeight - 1 - globalIdx) < mskWid || globalIdy < mskWid || (md->imageWidth - 1 - globalIdy) < mskWid))
		filterResult = 0;
	filteredImage[offset] = filterResult;
	// calculate sigma and apply it
	double threshold, sigma, var = 0;
	double n = md->cameraNoiseFactor;
	double t = md->cameraTrueEMGain;
	double e = md->cameraElectronsPerCount;
	sigma = sqrt(var + (n * n) * (e * t * max(filterResult - md->cameraOffset, 1.0) + t * t)) / e;
	if (md->SNThreshold)
		threshold = sigma * md->fudgeFactor * md->threshold;
	else
		threshold = md->threshold;
	binaryImage[offset] = (filterResult > threshold) ? 1 : 0;

}


kernel void labelInit(global int *label,
                      global ushort *binaryImage,
                      constant struct Metadata  * md)
{
  // begin
  const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
  int bgc = BGCOLOUR, iw = md->imageWidth, ih= md->imageHeight;
  int posi = globalIdx * iw + globalIdy;

  if (globalIdy >= iw || globalIdx >= ih) return;

  if (binaryImage[posi] == bgc) { label[posi] = 0; return; }
  if (globalIdx > 0 && binaryImage[posi] == binaryImage[posi-iw]) { label[posi] = posi-iw; return; }
  if (globalIdy > 0 && binaryImage[posi] == binaryImage[posi- 1]) { label[posi] = posi- 1; return; }
  label[posi] = posi;

}

kernel void labelMain(global int * label,
					  constant struct Metadata  * md,
					  global int * syncIndex)
{
  //begin
  KERNELBEGIN
  //const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);

  int iw = md->imageWidth, ih = md->imageHeight, pass = syncIndex[0];
  int posi = globalIdx * iw + globalIdy;
  if (globalIdy >= iw || globalIdx >= ih) return;

  int g = label[posi], og = g;

  if (g == 0) return;

  int rx[4] = {-1, 1, 0, 0};
  int ry[4] = {0, 0, -1, 1};
  for (int i = 0; i < 4; i++)
  {
	  if (0 <=  globalIdy + ry[i] &&  globalIdy + ry[i] < iw && 0 <=  globalIdx + rx[i] &&  globalIdx + rx[i] < ih)
	  {
			const int p1 = (globalIdx + rx[i]) * iw + globalIdy + ry[i], s = label[p1];
			if (s != 0 && s < g) g = s;
	  }
  }

  for(int j=0;j<6;j++) g = label[g];

  if (g != og)
  {
    atomic_min(&label[og], g);
    atomic_min(&label[posi], g);
  }

  if (localIdx == 0 && localIdy == 0 && atom_inc(&syncIndex[1]) == groupNumx * groupNumy - 1)
	{
		syncIndex[0] += 1;
		syncIndex[1] = 0;
	}
}

kernel void calcCandiPosiInit(global int * label,
							  global ushort * regionPointPosi,
							  global int * count
							  global int * pointCount,
							  constant struct Metadata * md)
{

	const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
	int posi = globalIdx * md->imageWidth + globalIdy;
	if (globalIdy >= md->imageWidth || globalIdx >= md->imageHeight) return;
	if (label[posi] == 0) return;

	if (label[posi] == posi)
	{
		// count[0] record the count of the candidate position
		// pointCount[order] record the point count of order(th) candidate position region
		int order = atomic_inc(count);
		if (order >= MAXCOUNT) return;
		pointCount[order] = 1;
		regionPointPosi[order * MAXREGIONPOINT] = globalIdx;
		regionPointPosi[order * MAXREGIONPOINT + 1] = globalIdy;
		label[posi] = -order - 1;
	}

}

kernel void calcCandiPosiSum(global int * label,
							 global ushort * regionPointPosi,
							 global int * count,
							 global int * pointCount,
							 constant struct Metadata * md)
{
	const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
	int posi = globalIdx * md->imageWidth + globalIdy;
	if (globalIdy >= md->imageWidth || globalIdx >= md->imageHeight) return;
	if (label[posi] <= 0) return;

	int order = -label[label[posi]] - 1;
	if (order > MAXCOUNT - 1) return;
	int pointOrder = atomic_inc(&pointCount[order]);
	if (pointOrder > MAXREGIONPOI   / 2 - 1) return;

	regionPointPosi[order * MAXREGIONPOINT + 2 * pointOrder] = globalIdx;
	regionPointPosi[order * MAXREGIONPOINT + 2 * pointOrder + 1] = globalIdy;

}

kernel void caclCandiPosiMain(global double * filteredImage,
							  global ushort * regionPointPosi,
							  global double * tempCandiPosi,
							  global int * count,
							  global int * pointCount,
							  constant struct Metadata * md)
{

	const int idx = get_global_id(0);
	int candiCount = count[0];
	if (idx > candiCount-1 || idx > MAXCOUNT) return;
	int pointNum = (pointCount[idx] > MAXREGIONPOINT)? MAXREGIONPOINT : pointCount[idx], X = 0, Y = 0;
	double posix = 0, posiy = 0, totalIntensity = 0, I = 0
	for (int i = 0; i < pointNum; i++)
	{
		X = regionPointPosi[idx * MAXREGIONPOINT + 2 * i];
		Y = regionPointPosi[idx * MAXREGIONPOINT + 2 * i + 1];
		I = filteredImage[X * md->imageWidth + Y];
		posix += X * I;
		posiy += Y * I;
		totalIntensity += I;
	}

	posix = posix / totalIntensity;
	posiy = posiy / totalIntensity;
	tempCandiPosi[2 * idx + 0] = posix;
	tempCandiPosi[2 * idx + 1] = posiy;

}

kernel void debounceCandiPosi(global double * filteredImage,
							  global double * candiPosi,
							  global double * tempCandiPosi,
							  global int * count,
							  constant struct Metadata * md)
{
	const int idx = get_global_id(0);
	int candiCount = count[0];
	double debounceRadius = md->debounceRadius;
	if (idx > candiCount-1 || idx > MAXCOUNT) return;
	double posix = 0, posiy = 0;

	int neighCount = 0, neighDistance = 0, tempX = (int)posix, tempY = (int)posiy, tempData = 0, maxIntensity = filteredImage[tempX*md->imageWidth+tempY];
	int neigh[100];

	for (int i = 0; i < candiCount; i++)
	{
		neighDistance = sqrt((posix - tempCandiPosi[2 * i]) * (posix - tempCandiPosi[2 * i]) +
		(posiy - tempCandiPosi[2 * i + 1]) * (posiy - tempCandiPosi[2 * i + 1]));
		if (neighDistance < debounceRadius)
			neigh[neighCount++] = i;
	}

	neighCount = (neighCount > 5)? 5 : neighCount;
	for (int i = 0; i < neighCount; i++)
	{
		tempX = (int)tempCandiPosi[2 * neigh[i]];
		tempY = (int)tempCandiPosi[2 * neigh[i] + 1];
		tempData = filteredImage[tempX*md->imageWidth+tempY];
		if (tempData > maxIntensity)
		{
			posix = -1.0;
			posiy = -1.0;
			break;
		}
	}

	candiPosi[2 * idx] = posix;
	candiPosi[2 * idx + 1] = posiy;

	if(idx == 0)
		count[0] = 0;
}
/* Recycle Code
*/

