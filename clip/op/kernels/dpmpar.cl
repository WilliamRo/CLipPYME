#ifndef DPMPAR_CL_INCLUDED
#define DPMPAR_CL_INCLUDED

#include "clminpack.clh"

real dpmpar(int i)
{
#ifdef USE_FLOAT_64
	switch (i)
	{
	case 1:
		return 2.2204460492503131e-16;
	case 2:
		return 2.2250738585072014e-308;
	default:
		return 1.7976931348623157e+308;
	}
#else
	switch (i)
	{
	case 1:
		return 1.19209290e-07F;
	case 2:
		return 1.17549435e-38F;
	default:
		return 3.40282347e+38F;
	}
#endif
}

#endif