#ifndef LMDIF_CL_INCLUDED
#define LMDIF_CL_INCLUDED

#include "clminpack.clh"

void lmdif(local fcndata *p, local real *x, local real *fvec,
		   real ftol, real xtol, real gtol, int maxfev, real epsfcn,
		   local real *diag, int mode, real factor, local int *nfev,
		   local real *fjac, int ldfjac, local int *ipvt, local real *qtf,
		   local real *wa1, local real *wa2, local real *wa3,
		   local real *wa4, local int *inta, local real *reala,
		   local real *wa0)
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

	inta: int array of length INTN

	reala: real array of length REALN

	info(inta[INFO]): int scalar [output]

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

	// :: private variables
	int index = INDEX;
	int i, j, l;

	real d1, d2;
	real temp, temp1, temp2, sum;
	real fnorm1, actred;
	real ratio; 
	/// S9150, double, GS64: 13 us
	/// S9150, double, GS11x11: 12 us
	if (index == N)
	{
		reala[DELTA] = 0;
		reala[XNORM] = 0.0;
		reala[EPSMCH] = dpmpar(1);
		inta[INFO] = -1;
		*nfev = 0;
	}
	/// S9150, double, GS11x11: 11~12 us
	/// S9150, double, GS64: 14~15 us
	// > check the input parameters for errors
	if (index == N)
		if (N <= 0 || M < N || ldfjac < M || ftol < 0. || xtol < 0. ||
			gtol < 0. || maxfev <= 0 || factor <= 0.) inta[INFO] = 0;

	if (mode == 2 && index < N && diag[index] <= 0) inta[INFO] = 0;

	loc_bar;

	if (inta[INFO] >= 0) return;

	if (index == N) inta[INFO] = 0;
	
#pragma endregion
	
#pragma region Preparation
	/// S9150, double, GS64: 14~15 us
	/// S9150, double, GS11x11: 12~13 us
	fcn_mn(p, x, fvec);	
	/// S9150, double, GS11x11: 17~18 us
	/// S9150, double, GS64: 21~22 us
	if (index == N) *nfev = 1;
	
	loc_bar;	/// S9150, double, GS11x11: 17~18 us.

	enorm_p(M, fvec, &reala[FNORM], wa4);

	if (index == N)
	{
		//reala[FNORM] = enorm_w(M, fvec);

		reala[PAR] = 0.0;
		inta[LMDIF_ITER] = 1;
	}

	/// S9150, double, GS11x11: 22~23 us

#pragma region [ Verification ]
#if 0
	loc_bar;
	if (index == N) printf("# reala[FNORM] = %.16f\n", reala[FNORM]);
	return;
#endif
#pragma endregion

#pragma endregion

#pragma region Outer Loop

	for (;;)
	{
		// > calculate the jacobian matrix 
		/// S9150, double, GS11x11: 22~23 us
		fdjac2(p, x, fvec, fjac, ldfjac, epsfcn, wa4, reala);
		/// S9150, double, GS11x11: 40~41 us
		if (index == N) *nfev += N;

		// > compute the qr factorization of the jacobian
		/// S9150, double, GS11x11: 40~41 us
		qrfac(fjac, ldfjac, true, ipvt, wa1, wa2, wa3, inta, reala, wa0);
		/// S9150, double, GS11x11: 416~417 us

#pragma region [ Verification ]
#if 0
		loc_bar;
		if (index == N)
		{
			//printf("# ipvt = [%d %d %d %d %d %d %d]\n",
			//	   ipvt[0], ipvt[1], ipvt[2], ipvt[3],
			//	   ipvt[4], ipvt[5], ipvt[6]);

			//printf("# acnorm = \n[%.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f]\n",
			//	   wa2[0], wa2[1], wa2[2], wa2[3], wa2[4], wa2[5], wa2[6]);

			printf("# rdiag = \n[%.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f\n %.10f]\n",
				   wa1[0], wa1[1], wa1[2], wa1[3], wa1[4], wa1[5], wa1[6]);
		}
		return;
#endif
#pragma endregion

		// > on the first iteration and if mode is 1, scale 
		//    according to the norms of the columns of the 
		//    initial jacobian. 
		if (inta[LMDIF_ITER] == 1 && index < N)
		{
			j = index;
			if (mode != 2)
			{
				diag[j] = wa2[j];
				if (wa2[j] == 0.0) diag[j] = 1.0;
			}
			// >> on the first iteration, calculate the norm 
			//     of the scaled x and initialize the step bound reala[DELTA]
			wa3[j] = diag[j] * x[j];
			if (j == 0)
			{
				reala[XNORM] = enorm_w(N, wa3);
				reala[DELTA] = factor * reala[XNORM];
				if (reala[DELTA] == 0.0) reala[DELTA] = factor;
			}
		}
		/// S9150, double, GS11x11: 416~417 us
#pragma region [ Verification ]
#if 0
		loc_bar;
		if (index == 0) printf("$ reala[DELTA] = %.10f\n", reala[DELTA]);
		return;
#endif
#pragma endregion
		
		// > form (q transpose)*fvec and store the first n components  
		//	  int qtf
		wa4[index] = fvec[index];
		
		/// S9150, double, GS11x11: 411 us
		for (j = 0; j < N; ++j)  /// about 59 us
		{
			// :: if v_k[0] is 0, H_k does nothing
			if (fjac[j + j * ldfjac] != 0.)
			{
				loc_bar;

				// >> use one work item to calculate -beta * (v^t * r) 
				if (index == N)
				{
					sum = 0.0;
					for (i = j; i < M; ++i)
						sum += fjac[i + j * ldfjac] * wa4[i];
					reala[LMDIF_TEMP] = -sum / fjac[j + j * ldfjac];
				}
				
				loc_bar;

				i = index;
				if (j <= i)
					wa4[i] += fjac[i + j * ldfjac] * reala[LMDIF_TEMP];
			}
			if (index == N)
			{
				fjac[j + j * ldfjac] = wa1[j];
				qtf[j] = wa4[j];
			}
		}

		loc_bar;
		/// S9150, double, GS11x11: 600 us
#pragma region [ Verification ]
#if 0
		if (index < N) printf("# qtf[%d] = %.10f\n", index, qtf[index]);
		return;
#endif
#pragma endregion

		/// => 646~647 us
		// > compute the norm of the scaled gradient
		/// about 45 us
		if (reala[FNORM] != 0.0)
		{
			j = index;
			if (index < N)
			{
				l = ipvt[j] - 1;
				if (wa2[l] != 0.0)
				{
					sum = 0.0;
					for (i = 0; i <= j; ++i)
						sum += fjac[i + j * ldfjac] * (qtf[i] / reala[FNORM]);

					wa1[l] = fabs(sum / wa2[l]);
				}
			}
		}
		
		loc_bar;

		if (index == N)
		{
			reala[GNORM] = 0.0;
			for (j = 0; j < N; j++)
				if (wa1[j] > reala[GNORM])
					reala[GNORM] = wa1[j];

			// > test for convergence of the gradient norm
			if (reala[GNORM] <= gtol) inta[INFO] = 4;
		}

		loc_bar;  // => 690~691 us

		if (inta[INFO] != 0) return;

#pragma region [ Verification ]
#if 0
		if (index == 0)
			printf(">>> gnorm = %.10f\n", reala[GNORM]);
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
		/// S9150, double, GS11x11: 615 us
		do
		{
			// > determine the levenberg-marquardt parameter
			lmpar(fjac, ldfjac, ipvt, diag, qtf, reala[DELTA],
				  &reala[PAR], wa1, wa2, wa3, wa4, inta, reala);

			loc_bar;  /// => 672 us

			j = index;
			if (j < N) {
				wa1[j] = -wa1[j];
				wa2[j] = x[j] + wa1[j];
				wa3[j] = diag[j] * wa1[j];
			}

			loc_bar;  /// => 672 us

			if (index == N) {
				reala[PNORM] = enorm_w(N, wa3);
				// > on the first iteration, adjust the initial step bound
				// !! private -> local: + 5 us
				if (inta[LMDIF_ITER] == 1) reala[DELTA] = min(reala[DELTA], reala[PNORM]);
			}
			
			loc_bar;  /// => 686~687 us

#pragma region [ Verification ]
#if 0
			if (index == N)
				printf("# pnorm = %.10f\n", reala[PNORM]);
#endif
#pragma endregion

			// > evaluate the function at x + p and calculate its norm
			fcn_mn(p, wa2, wa4);
			/// => 689 us
			if (index == N) {
				++(*nfev);
				fnorm1 = enorm_w(M, wa4);
				/// => 691 us
				// > compute the scaled actual reduction
				actred = -1.0;
				if (p1 * fnorm1 < reala[FNORM]) {
					// - computing 2nd power 
					d1 = fnorm1 / reala[FNORM];
					actred = 1. - d1 * d1;
				}
			}
			
			loc_bar;	/// !! => 729 us

			/* > compute the scaled predicted reduction and the
				 scaled directional derivative

				:: pre_red = (m(0) - m(p)) / m(0)
				::              t   t           t
				::         =  (p * J * J * p + J * r * p) / m(0)

				::              t    t   t           t       t       t
				:: J = Q * R * P => p * J * J * p = p * P * R * R * P * p
				::
				:: m(0) = fnorm * fnorm							*/

			if (index == N) {  /// costs little time, still can be optimized
				for (j = 0; j < N; ++j) {
					wa3[j] = 0.;
					l = ipvt[j] - 1;
					temp = wa1[l];
					for (i = 0; i <= j; ++i) {
						wa3[i] += fjac[i + j * ldfjac] * temp;
					}
				}
			}

			loc_bar;	/// => 732~734

			if (index == N) {
				temp1 = enorm(N, wa3) / reala[FNORM];
				temp2 = (sqrt(reala[PAR]) * reala[PNORM]) / reala[FNORM];
				reala[PRERED] = temp1 * temp1 + temp2 * temp2 / p5;
				reala[DIRDER] = -(temp1 * temp1 + temp2 * temp2);

				// > compute the ratio of the actual to the predicted 
				//   reduction
				ratio = 0.0;
				if (reala[PRERED] != 0.0) ratio = actred / reala[PRERED];
				/// => 729~732
				// > update the step bound
				if (ratio <= p25) {
					if (actred >= 0.0) temp = p5;
					else temp = p5 * reala[DIRDER] / (reala[DIRDER] + p5 * actred);
					if (p1 * fnorm1 >= reala[FNORM] || temp < p1) temp = p1;
					// > computing min
					d1 = reala[PNORM] / p1;
					reala[DELTA] = temp * min(reala[DELTA], d1);
					reala[PAR] /= temp;
				}
				else {
					if (reala[PAR] == 0.0 || ratio >= p75) {
						reala[DELTA] = reala[PNORM] / p5;
						reala[PAR] = p5 * reala[PAR];
					}
				}
				/// => 787~790 !!!
				inta[LMDIF_FLAG] = ratio >= p0001;
			}

			loc_bar;

#pragma region [ Verification ]
#if 0
			if (index == N) {
				printf("# prered = %.10f\n", reala[PRERED]);
				printf("# ratio = %.10f\n", ratio);
				printf("# actred = %.10f\n", actred);
			}
			return;
#endif
#pragma endregion

			// > test for successful iteration
			if (inta[LMDIF_FLAG]) {
				// successful iteration. update x, fvec, and their norms
				j = index;
				if (j < N) {
					x[j] = wa2[j];
					wa2[j] = diag[j] * x[j];
				}
				fvec[index] = wa4[index];

				loc_bar;

				if (index == N) {
					reala[XNORM] = enorm(N, wa2);
					reala[FNORM] = fnorm1;
					++inta[LMDIF_ITER];
				}
			}

			loc_bar;	/// => 787~788 us

			if (index == N) {
				// > tests for convergence
				if (fabs(actred) <= ftol && reala[PRERED] <= ftol && p5 * ratio <= 1.) {
					inta[INFO] = 1;
				}
				if (reala[DELTA] <= xtol * reala[XNORM]) {
					inta[INFO] = 2;
				}
				if (fabs(actred) <= ftol && reala[PRERED] <= ftol && p5 * ratio <= 1. && inta[INFO] == 2) {
					inta[INFO] = 3;
				}
			}
			
			loc_bar;   /// => 795~797 us
			if (inta[INFO] != 0) return;

			if (index == N)
			{
				// > tests for termination and stringent tolerances
				if (*nfev >= maxfev) {
					inta[INFO] = 5;
				}
				if (fabs(actred) <= reala[EPSMCH] && reala[PRERED] <= reala[EPSMCH] && p5 * ratio <= 1.) {
					inta[INFO] = 6;
				}
				if (reala[DELTA] <= reala[EPSMCH] * reala[XNORM]) {
					inta[INFO] = 7;
				}
				if (reala[GNORM] <= reala[EPSMCH]) {
					inta[INFO] = 8;
				}
				inta[LMDIF_FLAG] = ratio < p0001;
			}

#pragma region [ Verification ]
#if 0
			if (index == N) {
				printf("## info = %d\n", inta[INFO]);
				printf("## gnorm = %.10f\n", reala[GNORM]);
			}
			return;
#endif
#pragma endregion

			loc_bar;
			if (inta[INFO] != 0) return;

		} while (inta[LMDIF_FLAG]);

#pragma endregion

	}

#pragma endregion
}

#endif
