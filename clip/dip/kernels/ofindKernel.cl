/***************************************
This file contain all kernel used in PYME OpenCL version.
This is a version 0.05
From HPEC Lab, ZheJiang University, 2016/07/12.
***************************************/

#define Ftype float
#define Ftype2 float2
#define Ftype4 float4
#define Ftype16 float16

#define LOWFILTERLENGTH 9
#define HIGHFILTERLENGTH 25

#define MIN(x, y) ((x < y)? (x) : (y))
#define MAX(x, y) ((x > y)? (x) : (y))

#define BGCOLOUR 0
#define MAXLABELPASS 10

#define MAXCOUNT 5000
#define MAXREGIONPOINT 2000

#define IMAGEBOUNDAY(x, y) x = ((x) < 0)? -(x) - 1 : (x); \
						   x = ((x) >= (y))? 2 * (y) - (x) - 1 : (x);

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

struct Metadata
{
	// Image information
	int imageWidth;
	int imageHeight;
	int imageDepth;
	Ftype voxelSizeX; // um
	Ftype voxelSizeY; // um
	Ftype voxelSizeZ; // um
	bool is2DImage; // 0: 2DImage 1: 3DImage

	// Image filter setting
	int filterRadiusLowpass;
	int filterRadiusHighpass;
	int lowFilterLength;
	int highFilterLength;
	Ftype weightsLow[LOWFILTERLENGTH];
	Ftype weightsHigh[HIGHFILTERLENGTH];
	Ftype weights[HIGHFILTERLENGTH];

	// Camera information
	Ftype cameraOffset;
	Ftype cameraNoiseFactor;
	Ftype cameraElectronsPerCount;
	Ftype cameraTrueEMGain;

	// Finding setting
	Ftype debounceRadius;
	Ftype threshold;
	Ftype fudgeFactor;
	int maskEdgeWidth;
	bool SNThreshold;
	
	// Subtracting background setting
	int bgStartInd;
	int bgEndInd;
	int maxFrameNum;
	int minBgIndicesLen;
};

kernel void initBuffer(global int * bufferIndex,
					   global int2 * syncIndex,
					   global int2 * count,
					   global int * pass)
{
	const int id = get_global_id(0);
	if (id == 0)
	{
		bufferIndex[0] = bufferIndex[0] + 0;
		syncIndex[0] = (int2){1, 0};
		count[0] = (int2)0;
		pass[0] = 1;
		for (int i = 1; i < MAXLABELPASS; i++)
			pass[i] = 0;
	}
}

kernel void subBgAndCalcSigmaThres(global Ftype * imageStack,
                                   global Ftype * image,
							       global Ftype * sigmaMap,
							       global Ftype * thresholdMap,
							       global Ftype * varianceMap,
                                   global int * bufferIndex,
                                   constant struct Metadata * md)
{
    // begin
    const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
    int iw = md->imageWidth, ih = md->imageHeight, bInd = bufferIndex[0];
    if (globalIdx >= ih || globalIdy >= iw) return;
    int posi = globalIdx * iw + globalIdy, stackPosi = bInd * iw * ih + posi;

    // get image
    image[posi] = imageStack[stackPosi];
	
	// calculate sigma map
    Ftype n = md->cameraNoiseFactor;
	Ftype t = md->cameraTrueEMGain;
	Ftype e = md->cameraElectronsPerCount;
	varianceMap[posi] = 0; // !!! set variance map to 0 in this version
	sigmaMap[posi] = sqrt(varianceMap[posi] + (n * n) * \
	(e * t * MAX(image[posi] - md->cameraOffset, 1.0) + t * t)) / e;
	// calculate threshold map
	if (md->SNThreshold)
		thresholdMap[posi] = sigmaMap[posi] * md->fudgeFactor * md->threshold;
	else
		thresholdMap[posi] = md->threshold;
	
	// get background index
	int startInd = MAX(0, bInd + md->bgStartInd);
	int endInd = MIN(md->maxFrameNum - 1, bInd + md->bgEndInd);
	Ftype bg = 0;
	if (endInd - startInd >= md->minBgIndicesLen)
	{
		for (int i = startInd; i < endInd; i++)
			bg = bg + imageStack[i*iw*ih+posi];
		bg = bg / (endInd - startInd);
	}
	else
		bg = md->cameraOffset;
	
	// get subtracted image
	image[posi] = image[posi] - bg;
	image[posi] = select(image[posi], (float)0.0, isless(image[posi], (float)0.0));
}

kernel void padImage(global Ftype * image,
							   global Ftype * paddedImage,
							   constant struct Metadata * md)
{
	// begin
	int2 id = {get_global_id(0), get_global_id(1)};
	// get real position
	int ih = md->imageHeight, iw = md->imageWidth, halflen = md->filterRadiusHighpass, \
		padlen = 2 * halflen;
	if (id.x >= ih + padlen || id.y >= iw + padlen)
		return;
	int padposi = id.x * (iw + padlen) + id.y;
	id.x = (id.x < ih + halflen)? abs(id.x - halflen) : (ih - abs(id.x - ih));
	id.y = (id.y < iw + halflen)? abs(id.y - halflen) : (iw - abs(id.y - iw));
	paddedImage[padposi] = image[id.x * iw + id.y];
}

kernel void filterImage(global Ftype * paddedImage,
								global Ftype * filteredImage,
								global Ftype * binaryImage,
								global Ftype * thresholdMap,
								constant struct Metadata * md)
{
	// begin
	const int2 id = {get_global_id(0), get_global_id(1)};
	// get real position in array
	int ih = md->imageHeight, iw = md->imageWidth, halflen = md->filterRadiusHighpass,\
		posi = id.x * iw + id.y, padiw = iw + 2 * halflen;
	if (id.y >= iw || id.x >= ih) return;
	int2 padId = id + (int2)(halflen, halflen);
	
	// convolution
	Ftype filterResult = 0;
	int count = 0;
	for (int i = -halflen; i <= halflen; i++)
		for (int j = -halflen; j<= halflen; j++)
			filterResult += (paddedImage[(padId.x+i)*padiw+padId.y+j] * md->weights[count++]);
	
	// get result
	 filterResult = filterResult < 0 ? 0 : filterResult;
	int mskWid = md->maskEdgeWidth;
	if (iw > mskWid && \
		(id.x < mskWid || (ih - 1 - id.x) < mskWid || id.y < mskWid || (iw - 1 - id.y) < mskWid))
		filterResult = 0;
	filteredImage[posi] = filterResult;

	// applying threshold
	binaryImage[posi] = (filterResult > thresholdMap[posi]) ? 1 : 0;
}

#if 1
kernel void colFilterImage(global Ftype * image,
						   global Ftype2 * lowFilteredImage,
						   global Ftype2 * highFilteredImage,
						   constant struct Metadata * md)
{
	// begin
	const int2 id = {get_global_id(0), get_global_id(1)};
	int iw = md->imageWidth, ih = md->imageHeight;
	if (id.x >= ih || 2*id.y >= iw) return;

	// define parameters
	int temp;
	int2 idy = 2*id.y + (int2){0,1};
	Ftype2 lowFilterResult = (Ftype2)0.0, highFilterResult = (Ftype2)0.0, im;

	// convolution
	for (int i = 0; i < HIGHFILTERLENGTH; i++)
	{
		temp = id.x + (i - md->filterRadiusHighpass);
		temp = select(temp, -temp-1, isless(temp, 0.0));
		temp = select(2*ih-temp-1, temp, isless(temp, (Ftype)ih)) * iw;
		im = (Ftype2){image[temp+idy.S0], image[temp+idy.S1]};
		lowFilterResult += im * (Ftype2)(md->weights[i]);
		highFilterResult += im * (Ftype2)(md->weightsHigh[i]); 
	}
	lowFilteredImage[id.x*iw/2+id.y] = lowFilterResult;
	highFilteredImage[id.x*iw/2+id.y] = highFilterResult;
}

#else
kernel void colFilterImage(global Ftype * image,
						   global Ftype * lowFilteredImage,
						   global Ftype * highFilteredImage,
						   constant struct Metadata  * md)
{
	// begin
	const int2 id = {get_global_id(0), get_global_id(1)};
	int iw = md->imageWidth, ih = md->imageHeight;
	if (id.x >= ih || id.y >= iw) return;

	// define parameters
	int temp;
	Ftype lowFilterResult = 0.0, highFilterResult = 0.0;

	// convolution
	for (int i = 0; i < HIGHFILTERLENGTH; i++)
	{
		temp = id.x + (i - md->filterRadiusHighpass);
		temp = select(temp, -temp-1, isless(temp, 0.0f));
		temp = select(2*ih-temp-1, temp, isless(temp, (Ftype)ih)) * iw + id.y;
		lowFilterResult += image[temp] * md->weights[i];
		highFilterResult += image[temp] * md->weightsHigh[i]; 
	}
	lowFilteredImage[id.x*iw+id.y] = lowFilterResult;
	highFilteredImage[id.x*iw+id.y] = highFilterResult;

}
#endif

#if 0
kernel void rowFilterImage(global Ftype * lowFilteredImage,
                           global Ftype * highFilteredImage,
                           global Ftype2 * filteredImage,
                           global ushort2 * binaryImage,
                           global Ftype2 * thresholdMap,
                           constant struct Metadata  * md)
{
	// begin
	const int2 id = {get_global_id(0), get_global_id(1)};
	int iw = md->imageWidth, ih = md->imageHeight;
	if (id.x >= ih || 2*id.y >= iw) return;

	// convolution
	int2 temp;
	int2 idy = 2*id.y + (int2){0,1};
	Ftype2 filterResult = (Ftype2)0.0;
	for (int i = 0; i < HIGHFILTERLENGTH; i++)
	{
		temp = idy + (i - md->filterRadiusHighpass);
		temp = select(temp, -temp-1, isless(convert_float2(temp), (Ftype2)0.0f));
		temp = select(2*iw-temp-1, temp, isless(convert_float2(temp), (Ftype2)iw)) + id.x*iw;
		filterResult += (Ftype2){lowFilteredImage[temp.S0], lowFilteredImage[temp.S1]} * (Ftype2)md->weights[i];
		filterResult -= (Ftype2){highFilteredImage[temp.S0], highFilteredImage[temp.S1]} * (Ftype2)md->weightsHigh[i];
	}
	// mask image edge
	filterResult = as_float2(as_int2(filterResult) & (filterResult > (Ftype2)0));
	int mskWid = md->maskEdgeWidth;
	int2 isTrue = iw > mskWid && \
		(id.x < mskWid || (ih - 1 - id.x) < mskWid || idy < mskWid || (iw - 1 - idy) < mskWid);
	filterResult = as_float2(as_int2(filterResult) & ~isTrue);
	int offset = id.x*iw/2+id.y;
	filteredImage[offset] = filterResult;

	// applying threshold
	binaryImage[offset] = convert_ushort2(select((Ftype2)0, (Ftype2)1, isless(thresholdMap[offset], filterResult)));
}
#else
kernel void rowFilterImage(global Ftype * lowFilteredImage,
                           global Ftype * highFilteredImage,
                           global Ftype * filteredImage,
                           global ushort * binaryImage,
                           global Ftype * thresholdMap,
                           constant struct Metadata  * md)
{
	// begin
	const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
	// get real position in array
	int iw = md->imageWidth, ih = md->imageHeight, filterRadius = md->filterRadiusHighpass, \
		posi = globalIdx * iw + globalIdy;
	if (globalIdy >= iw || globalIdx >= ih) return;
		
	// define some parameters and get parameters from metadata
	int temp;
	Ftype filterResult = 0;

	// high : row
	for (int i = 0; i < HIGHFILTERLENGTH; i++)
	{
		temp = globalIdy + (i - filterRadius);
		IMAGEBOUNDAY(temp, iw)
		filterResult += lowFilteredImage[globalIdx * iw + temp] * md->weights[i];
		filterResult -= highFilteredImage[globalIdx * iw + temp] * md->weightsHigh[i];
	}

	// get result
	 filterResult = filterResult < 0 ? 0 : filterResult;
	int mskWid = md->maskEdgeWidth;
	if (iw > mskWid && \
		(globalIdx < mskWid || (ih - 1 - globalIdx) < mskWid || globalIdy < mskWid || (iw - 1 - globalIdy) < mskWid))
		filterResult = 0;
	filteredImage[posi] = filterResult;

	// applying threshold
	binaryImage[posi] = (filterResult > thresholdMap[posi]) ? 1 : 0;

}
#endif

kernel void labelInit(global int *label,
                      global ushort *binaryImage,
                      constant struct Metadata  * md)
{
  // begin
  const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
  int bgc = BGCOLOUR, iw = md->imageWidth, ih= md->imageHeight;
  int posi = globalIdx * iw + globalIdy;

  if (globalIdy == 0 || globalIdy >= iw || globalIdx == 0 || globalIdx >= ih) return;

  float3 im = {binaryImage[posi], binaryImage[posi-1], binaryImage[posi-iw]},
  		offset = {posi, posi-1, posi-iw};
  float flag = offset.S0;
  flag = select(flag, offset.S1, isequal(im.S1, 1.0f));
  flag = select(flag, offset.S2, isequal(im.S2, 1.0f));
  flag = select(0.0f, flag, isequal(im.S0, 1.0f));
  label[posi] = convert_int(flag);

}

kernel void labelMain(global int * label,
					  constant struct Metadata  * md,
					  global int * syncIndex,
					  global int * pass)
{
  //begin
  const int2 id = {get_global_id(0), get_global_id(1)};

  int iw = md->imageWidth, ih = md->imageHeight, posi = id.x * iw + id.y;
  if (pass[syncIndex[0]-1] == 0 || id.y >= iw || id.x >= ih) return;

  int g = label[posi], og = g;

  if (g == 0) return;

  int rx[4] = {-1, 1, 0, 0};
  int ry[4] = {0, 0, -1, 1};
  for (int i = 0; i < 4; i++)
  {
	  if (0 <=  id.y + ry[i] &&  id.y + ry[i] < iw && 0 <=  id.x + rx[i] &&  id.x + rx[i] < ih)
	  {
			const int p1 = (id.x + rx[i]) * iw + id.y + ry[i], s = label[p1];
			if (s != 0 && s < g) g = s;
	  }
  }

  for(int i = 0; i < 6; i++) 
  	g = label[g];

  if (g != og)
  {
    atomic_min(&label[og], g);
    atomic_min(&label[posi], g);
	pass[syncIndex[0]] = 1;
  }

}

kernel void labelSync(global int * syncIndex)
{
	const int id = get_global_id(0);
	if (id == 0)
		syncIndex[0] += 1;
}

kernel void calcCandiPosiInit(global int * label,
							  global int * candiRegion,
							  global int * count,
							  constant struct Metadata * md)
{

	const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
	int posi = globalIdx * md->imageWidth + globalIdy;
	if (globalIdy >= md->imageWidth || globalIdx >= md->imageHeight) return;
	if (label[posi] == 0) return;

	// init object 
	if (label[posi] == posi)
	{
		// count[0] record the count of the candidate position
		int order = atomic_inc(count);
		if (order >= MAXCOUNT) return;
        candiRegion[4*order+0] = globalIdx;
        candiRegion[4*order+1] = globalIdy;
        candiRegion[4*order+2] = globalIdx;
        candiRegion[4*order+3] = globalIdy;
		label[posi] = -order - 1;
	}

}

kernel void getCandiPosiObj(global int * label,
                            global int * candiRegion,
                            global int * count,
                            constant struct Metadata * md)
{
    const int globalIdx = get_global_id(0), globalIdy = get_global_id(1);
	int posi = globalIdx * md->imageWidth + globalIdy;
	if (globalIdy >= md->imageWidth || globalIdx >= md->imageHeight) return;
	if (label[posi] <= 0) return;

	int order = -label[label[posi]] - 1;
	if (order > MAXCOUNT - 1) return;

	atomic_min(&candiRegion[4*order+0],globalIdx);
    atomic_min(&candiRegion[4*order+1],globalIdy);
	atomic_max(&candiRegion[4*order+2],globalIdx);
	atomic_max(&candiRegion[4*order+3],globalIdy);

}

kernel void caclCandiPosiMain(global Ftype * filteredImage,
							  global int * candiRegion,
							  global Ftype * tempCandiPosi,
							  global int * count,
							  constant struct Metadata * md)
{

	const int idx = get_global_id(0);
	int candiCount = count[0];
	if (idx > candiCount-1 || idx > MAXCOUNT) return;
	Ftype posix = 0, posiy = 0, totalIntensity = 0, I = 0;

	for (int i = candiRegion[4*idx]; i <= candiRegion[4*idx+2]; i++)
	    for (int j = candiRegion[4*idx+1]; j <= candiRegion[4*idx+3]; j++)
	    {
            I = filteredImage[i * md->imageWidth + j];
            posix += i * I;
            posiy += j * I;
            totalIntensity += I;
	    }

	tempCandiPosi[2*idx+0] = posix / totalIntensity;
	tempCandiPosi[2*idx+1] = posiy / totalIntensity;

}

kernel void debounceCandiPosi(global Ftype * filteredImage,
							  global Ftype * candiPosi,
							  global Ftype * tempCandiPosi,
							  global int * count,
							  constant struct Metadata * md)
{
	const int idx = get_global_id(0);
	int candiCount = count[0];
	Ftype debounceRadius = md->debounceRadius;
	if (idx > candiCount-1 || idx > MAXCOUNT) return;
	Ftype posix = tempCandiPosi[2*idx], posiy = tempCandiPosi[2*idx+1];

	int neighCount = 0, neighDistance = 0, tempX = (int)posix, tempY = (int)posiy, 
		tempData = 0, maxIntensity = filteredImage[tempX*md->imageWidth+tempY];
	int neigh[100];

	for (int i = 0; i < candiCount; i++)
	{
		neighDistance = sqrt((posix - tempCandiPosi[2 * i]) * (posix - tempCandiPosi[2 * i]) +
		(posiy - tempCandiPosi[2 * i + 1]) * (posiy - tempCandiPosi[2 * i + 1]));
		if (neighDistance < debounceRadius)
			neigh[neighCount++] = i;
	}

	bool flag = true;
	neighCount = (neighCount > 5)? 5 : neighCount;
	for (int i = 0; i < neighCount; i++)
	{
		tempX = (int)tempCandiPosi[2 * neigh[i]];
		tempY = (int)tempCandiPosi[2 * neigh[i] + 1];
		tempData = filteredImage[tempX*md->imageWidth+tempY];
		if (tempData > maxIntensity)
		{
			flag = false;
			break;
		}
	}
	if(flag)
	{
		int order = atomic_inc(&count[1]);
		candiPosi[2 * order] = posix;
		candiPosi[2 * order + 1] = posiy;	
	}

}


