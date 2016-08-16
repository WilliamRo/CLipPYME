#ifndef FDJAC2_CL_INCLUDED
#define FDJAC2_CL_INCLUDED

#include "clminpack.clh"

// [Verified]
void fdjac2(local fcndata *p, local real *x, local real *fvec,
			local real *fjac, int ldfjac, real epsfcn,
			local real *wa, local real *reala)
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

	// determine eps
	if (index == 0)
	{
		/// on NVDIA 980M commenting next line costs about 40 us
		reala[EPSMCH] = dpmpar(1);  
		reala[EPS] = sqrt(fmax(epsfcn, reala[EPSMCH]));
	}
	//loc_bar; return;
	// calculate fjac
	for (int j = 0; j < N; j++)
	{
		if (index == 0)
		{
			reala[FD_TEMP] = x[j];
			reala[FDH] = reala[EPS] * fabs(reala[FD_TEMP]);
			if (reala[FDH] < reala[EPSMCH]) reala[FDH] = reala[EPS];
			x[j] = reala[FD_TEMP] + reala[FDH];
		}
		loc_bar;

		fcn_mn(p, x, wa);
		loc_bar;

		if (index == 0) x[j] = reala[FD_TEMP];
		fjac[index + j * ldfjac] = (wa[index] - fvec[index]) / reala[FDH];
	}
}

#endif
