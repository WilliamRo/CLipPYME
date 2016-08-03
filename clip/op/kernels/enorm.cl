#ifndef ENORM_CL_INCLUDED
#define ENORM_CL_INCLUDED

#include "clminpack.h"

#ifdef USE_FLOAT_64

#define rdwarf 1.82691291192569e-153
#define rgiant 1.34078079299426e+153

#else

#define rdwarf 1.327871072777421e-18f
#define rgiant 1.844674297419792e+18f

#endif

real enorm_w(int n, local real *x)
{
	real sum = 0.0;
	for (int i = 0; i < n; i++)
		sum = mad(x[i], x[i], sum);
	return sqrt(sum);
}

// [Verified]
void enorm_p(int n, local real *x, local real *res, local real *wa)
/*
*/
{
	int index = INDEX;

	if (index < n) wa[index] = x[index] * x[index];

	loc_bar;

	int offset = M_NEXT >> 1;
	while (offset > 0)
	{
		if (offset <= index && index < min(n, offset << 1))
			wa[index - offset] += wa[index];

		offset >>= 1;
		loc_bar;
	}

	if (index == 0) {
		*res = sqrt(wa[0]);
	}
}

// [Verified]
real enorm(int n, local real *x)
/*
	exactly the same as enorm in minpack
*/
{
	/* System generated locals */
	real ret_val, d1;

	/* Local variables */
	int i;
	real s1, s2, s3, xabs, x1max, x3max, agiant;

	/*     ********** */

	/*     function enorm */

	/*     given an n-vector x, this function calculates the */
	/*     euclidean norm of x. */

	/*     the euclidean norm is computed by accumulating the sum of */
	/*     squares in three different sums. the sums of squares for the */
	/*     small and large components are scaled so that no overflows */
	/*     occur. non-destructive underflows are permitted. underflows */
	/*     and overflows do not occur in the computation of the unscaled */
	/*     sum of squares for the intermediate components. */
	/*     the definitions of small, intermediate and large components */
	/*     depend on two constants, rdwarf and rgiant. the main */
	/*     restrictions on these constants are that rdwarf**2 not */
	/*     underflow and rgiant**2 not overflow. the constants */
	/*     given here are suitable for every known computer. */

	/*     the function statement is */

	/*       double precision function enorm(n,x) */

	/*     where */

	/*       n is a positive integer input variable. */

	/*       x is an input array of length n. */

	/*     subprograms called */

	/*       fortran-supplied ... dabs,dsqrt */

	/*     argonne national laboratory. minpack project. march 1980. */
	/*     burton s. garbow, kenneth e. hillstrom, jorge j. more */

	/*     ********** */

	s1 = 0.;
	s2 = 0.;
	s3 = 0.;
	x1max = 0.;
	x3max = 0.;
	agiant = rgiant / (real)n;
	for (i = 0; i < n; ++i) {
		xabs = fabs(x[i]);
		if (xabs >= agiant) {
			/*              sum for large components. */
			if (xabs > x1max) {
				/* Computing 2nd power */
				d1 = x1max / xabs;
				s1 = 1. + s1 * (d1 * d1);
				x1max = xabs;
			}
			else {
				/* Computing 2nd power */
				d1 = xabs / x1max;
				s1 += d1 * d1;
			}
		}
		else if (xabs <= rdwarf) {
			/*              sum for small components. */
			if (xabs > x3max) {
				/* Computing 2nd power */
				d1 = x3max / xabs;
				s3 = 1. + s3 * (d1 * d1);
				x3max = xabs;
			}
			else if (xabs != 0.) {
				/* Computing 2nd power */
				d1 = xabs / x3max;
				s3 += d1 * d1;
			}
		}
		else {
			/*           sum for intermediate components. */
			/* Computing 2nd power */
			s2 += xabs * xabs;
		}
	}

	/*     calculation of norm. */

	if (s1 != 0.) {
		ret_val = x1max * sqrt(s1 + (s2 / x1max) / x1max);
	}
	else if (s2 != 0.) {
		if (s2 >= x3max) {
			ret_val = sqrt(s2 * (1. + (x3max / s2) * (x3max * s3)));
		}
		else {
			ret_val = sqrt(x3max * ((s2 / x3max) + (x3max * s3)));
		}
	}
	else {
		ret_val = x3max * sqrt(s3);
	}
	return ret_val;
}

#endif
