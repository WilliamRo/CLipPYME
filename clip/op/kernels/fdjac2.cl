#ifndef FDJAC2_CL_INCLUDED
#define FDJAC2_CL_INCLUDED

#include "clminpack.h"

// [Verified]
void fdjac2(local fcndata *p, local real *x,
			local real *fvec, local real *fjac,
			int ldfjac, real epsfcn, local real *wa)
	/*
	computes a forward-difference approximation to the 
	 M by N jacobian matrix associated with a specified 
	 problem of M functions in N variables

	Parameters
	----------
	p: pointer of fcndata
		parameters used to evaluate fcn

	x: real array of length N
		point of N-dimension space at which a forward-difference 
		approximation will be calculated
		
	fvec: real array of length M
		contain the functions evaluated at x

	fjac: real array of length M by N [output]
		contains the approximation to the jacobian matrix 
		evaluated at x

	ldfjac: int scalar, positive, not less than M
		specifies the leading dimension of the array fjac

	epsfcn: real scalar
		used in determining a suitable step length for the 
		forward-difference approximation. this approximation 
		assumes that the relative errors in the functions are 
		of the order of epsfcn. if epsfcn is less than the 
		machine precision, it is assumed that the relative 
		errors in the functions are of the order of the machine 
		precision. 

	wa: real array of length M
		a work array										*/
{
	// declare variables
	int index = INDEX;
	
	local real h;
	local real eps, temp, epsmch;

	// determine eps
	if (index == 0)
	{
		epsmch = dpmpar(1);
		eps = sqrt(max(epsfcn, epsmch));
	}

	// calculate fjac
	for (int j = 0; j < N; j++)
	{
		if (index == 0)
		{
			temp = x[j];
			h = eps * fabs(temp);
			if (h < epsmch) h = eps;
			x[j] = temp + h;
		}
		loc_bar;

		fcn_mn(p, x, wa);
		loc_bar;

		if (index == 0) x[j] = temp;
		fjac[index + j * ldfjac] = (wa[index] - fvec[index]) / h;
	}
}

#endif
