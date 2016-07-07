#ifndef LMDIF_CL_INCLUDED
#define LMDIF_CL_INCLUDED

#include "clminpack.h"

void lmdif(local fcndata *p, local real *x, local real *fvec,
		   real ftol, real xtol, real gtol, int maxfev, real epsfcn,
		   local real *diag, int mode, real factor, local int *nfev,
		   local real *fjac, int ldfjac, local int *ipvt,
		   local real *qtf, local real *wa1, local real *wa2,
		   local real *wa3, local real *wa4, local int *info)
	/* LMDIF

	Parameters
	----------
	p: pointer of fcndata
		parameters for evaluating fcn_mn

	x: real array of length N
		on input x must contain an initial estimate of the solution
		vector. on output x contains the final estimate of the
		solution vector.

	fvec: real array of length M
		contains the evaluation of fun_mn at the out put x

	ftol: real scalar, nonnegative
		termination occurs when both the actual and predicted
		relative reductions in the sum of squares are at most
		ftol. therefore, ftol measures the relative error desired
		in the sum of squares

	xtol: real scalar, nonnegative
		termination occurs when the relative error between two
		consecutive iterates is at most xtol. therefore, xtol
		measures the relative error desired in the approximate
		solution.

	gtol: real scalar, nonnegative
		termination occurs when the cosine of the angle between
		fvec and any column of the jacobian is at most gtol in
		absolute value. therefore, gtol measures the orthogonality
		desired between the function vector and the columns of the
		jacobian.

	maxfev: int scalar, positive
		termination occurs when the number of calls to fcn is at
		least maxfev by the end of an iteration.

	epsfcn: real scalar
		used in determining a suitable step length for the
		forward-difference approximation. this approximation assumes
		that the relative errors in the functions are of the order
		of epsfcn. if epsfcn is less than the machine precision, it
		is assumed that the relative errors in the functions are of
		the order of the machine precision.

	diag: real array of length N
		if mode = 1 (see below), diag is internally set.
		if mode = 2, diag must contain positive entries that serve
		as multiplicative scale factors for the variables.

	mode: int scalar
		if mode = 1, the variables will be scaled internally.
		if mode = 2,the scaling is specified by the input diag.
		other values of mode are equivalent to mode = 1.

	factor: real scalar
		used in determining the initial step bound. this bound is
		set to the product of factor and the euclidean norm of diag*x
		if nonzero, or else to factor itself. in most cases factor
		should lie in the interval (.1,100.). 100. is a generally
		recommended value

	nfev: int scalar [output]
		number of calls to fcn.

	fjac: real array of length m by n [output]
		the upper n by n submatrix of fjac contains an upper triangular
		matrix r with diagonal elements of nonincreasing magnitude
		such that
					  t      t              t
					 p * (jac * jac) * p = r * r,

		where p is a permutation matrix and jac is the final calculated
		jacobian. column j of p is column ipvt(j) (see below) of the
		identity matrix. the lower trapezoidal part of fjac contains
		information generated during the computation of r.

	ldfjac: int scalar, positive, not less than m
		 which specifies the leading dimension of the array fjac

	ipvt: int array of length n [output]
		defines a permutation matrix p such that jac*p = q*r,
		where jac is the final calculated jacobian, q is orthogonal
		(not stored), and r is upper triangular with diagonal elements
		of nonincreasing magnitude. column j of p is column ipvt(j)
		of the identity matrix

	qtf: real array of length n [output]
		contains the first n elements of the vector (q transpose)*fvec.

	wa1, wa2, wa3: real arrays of length n

	wa4: real array of length m

	info: int scalar [output]

		info = 0  improper input parameters.

		info = 1  both actual and predicted relative reductions
					in the sum of squares are at most ftol.

		info = 2  relative error between two consecutive iterates
					is at most xtol.

		info = 3  conditions for info = 1 and info = 2 both hold.

		info = 4  the cosine of the angle between fvec and any
					column of the jacobian is at most gtol in
					absolute value.

		info = 5  number of calls to fcn has reached or
					exceeded maxfev.

		info = 6  ftol is too small. no further reduction in
					the sum of squares is possible.

		info = 7  xtol is too small. no further improvement in
					the approximate solution x is possible.

		info = 8  gtol is too small. fvec is orthogonal to the
					columns of the jacobian to machine precision. 	*/
{
#pragma region Variable Initialization

	int glb_i = ggi(0);
	int glb_j = ggi(1);
	int index = glb_i * ROI_L + glb_j;

	real d1, d2;

	int i, j, l;
	local real par;
	local int iter;
	real temp, temp1, temp2, sum;
	local real loc_temp;

	local real delta;
	local real ratio;
	local real fnorm, gnorm;
	local real pnorm, xnorm, fnorm1, actred, dirder, epsmch, prered;

	if (index == 0)
	{
		delta = 0;
		xnorm = 0.0;
		epsmch = dpmpar(1);	// [Verified]
		*info = -1;
		*nfev = 0;
	}

	// > check the input parameters for errors
	if (index == 0)
		if (N <= 0 || M < N || ldfjac < M || ftol < 0. || xtol < 0. ||
			gtol < 0. || maxfev <= 0 || factor <= 0.) *info = 0;

	if (mode == 2 && index < N && diag[index] <= 0) *info = 0;

	loc_bar;

	if (*info >= 0) return;

	if (index == 0) *info = 0;

#pragma endregion

#pragma region Preparation

	fcn_mn(p, x, fvec);  /// less than 5 us
	if (index == 0) *nfev = 1;
	loc_bar;

	//enorm_p(M, fvec, &fnorm, wa4);		// TODO: SYNC

	if (index == 0)
	{
		fnorm = enorm(M, fvec);
		par = 0.0;
		iter = 1;
	}

#pragma region [ Verification ]
#if 0
	if (index == 0) printf(">>> fnorm = %.16f\n", fnorm);
#endif
#pragma endregion

#pragma endregion

#pragma region Outer Loop

	for (;;)
	{
		// > calculate the jacobian matrix 
		/// about 20 us
		fdjac2(p, x, fvec, fjac, ldfjac, epsfcn, wa4);
		if (index == 0) *nfev += N;

		// > compute the qr factorization of the jacobian
		/// about 550 us
		qrfac(fjac, ldfjac, true, ipvt, wa1, wa2, wa3);

		loc_bar;

#pragma region [ Verification ]
#if 0
		if (index == 0)
		{
			printf(">> ipvt = [%d %d %d %d %d %d %d]\n",
				   ipvt[0], ipvt[1], ipvt[2], ipvt[3],
				   ipvt[4], ipvt[5], ipvt[6]);

			printf(">> rdiag = \n[%.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f]\n",
				   wa1[0], wa1[1], wa1[2], wa1[3], wa1[4], wa1[5], wa1[6]);

			printf(">> acnorm = \n[%.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f]\n",
				   wa2[0], wa2[1], wa2[2], wa2[3], wa2[4], wa2[5], wa2[6]);
		}
#endif
#pragma endregion

		// > on the first iteration and if mode is 1, scale 
		//    according to the norms of the columns of the 
		//    initial jacobian. 
		if (iter == 1 && index < N)
		{
			j = index;
			if (mode != 2)
			{
				diag[j] = wa2[j];
				if (wa2[j] == 0.0) diag[j] = 1.0;
			}
			// >> on the first iteration, calculate the norm 
			//     of the scaled x and initialize the step bound delta
			wa3[j] = diag[j] * x[j];
			if (j == 0)
			{
				xnorm = enorm(N, wa3);
				delta = factor * xnorm;
				if (delta == 0.0) delta = factor;
			}
		}

#pragma region [ Verification ]
#if 0
		if (index == 0) printf(">>> delta = %.10f\n", delta);
#endif
#pragma endregion

		// > form (q transpose)*fvec and store the first n components  
		//	  int qtf
		wa4[index] = fvec[index];

		/// => 585 us
		for (j = 0; j < N; ++j)  /// about 59 us
		{
			// :: if v_k[0] is 0, H_k does nothing
			if (fjac[j + j * ldfjac] != 0.)
			{
				loc_bar;

				// >> use one work item to calculate -beta * (v^t * r) 
				if (index == 0)
				{
					sum = 0.0;
					for (i = j; i < M; ++i)
						sum += fjac[i + j * ldfjac] * wa4[i];
					loc_temp = -sum / fjac[j + j * ldfjac];
				}

				loc_bar;

				i = index;
				if (j <= i)
					wa4[i] += fjac[i + j * ldfjac] * loc_temp;
			}
			if (index == 0)
			{
				fjac[j + j * ldfjac] = wa1[j];
				qtf[j] = wa4[j];
			}
		}

		loc_bar;

#pragma region [ Verification ]
#if 0
		if (index < N) printf(">>> qtf[%d] = %.10f\n", index, qtf[index]);
#endif
#pragma endregion

		if (gtol != 0.0)
		{
			/// => 643 us
			// > compute the norm of the scaled gradient
			/// about 45 us
			if (fnorm != 0.0)
			{
				j = index;
				if (index < N)
				{
					l = ipvt[j] - 1;
					if (wa2[l] != 0.0)
					{
						sum = 0.0;
						for (i = 0; i <= j; ++i)
							sum += fjac[i + j * ldfjac] * (qtf[i] / fnorm);

						wa1[l] = fabs(sum / wa2[l]);
					}
				}
			}

			loc_bar;

			if (index == 0)
			{
				for (j = 0; j < N; j++)
					if (wa1[j] > gnorm)
						gnorm = wa1[j];

				// > test for convergence of the gradient norm
				if (gnorm <= gtol) *info = 4;
			}

			loc_bar;

			if (*info != 0) return;
		}

#pragma region [ Verification ]
#if 0
		if (index == 0)
			printf(">>> gnorm = %.10f\n", gnorm);
#endif
#pragma endregion

		// > rescale if necessary
		/// => 650 us
		if (mode != 2 && index < N)
		{
			d1 = diag[index];
			d2 = wa2[index];
			diag[index] = max(d1, d2);
		}

		loc_bar;

#pragma region Inner Loop

		do
		{


			break;
		} while (ratio < p0001);

#pragma endregion

		break;
	}

#pragma endregion
}

#endif
