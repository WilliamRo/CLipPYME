#ifndef LMPAR_CL_INCLUDED
#define LMPAR_CL_INCLUDED

#include "clminpack.h"

void lmpar(local real* r, int ldr, local int *ipvt,
		   local real *diag, local real *qtb, real delta,
		   local real *par, local real *x, local real *sdiag,
		   local real *wa1, local real *wa2)
	/* LMPAR

	Solves the sub-problem in the levenberg-marquardt algorithm.
	 By using the trust region framework, the L-M algorithm can be
	 regarded as solving a set of minimization problems:

							 2
		 min || J * p + r ||_2     s.t. || D * p || <= Delta
		  p

	 By introducing a parameter lambda into this sub-problem, the
	 constrained optimization problem can be converted to an
	 unconstrained optimization problem:

			  ||   /         J         \       /  r  \  ||
		  min ||  |                    | p +  |      |  ||
		   p  ||  \  sqrt(lambda) * D /       \  0  /   ||

	 This routine determines the value lambda and as a by-product,
	 it gives a nearly exact solution to the minimization problem

	 Let a = J, d = D, b = -r, x = p, par = lambda, we denoted the
	 optimization problem as :

			  ||   /         a         \       /  b  \  ||
		  min ||  |                    | x -  |      |  ||
		   x  ||  \   sqrt(par) * d   /       \  0  /   ||


	Parameters
	----------
	r: real array of length N by N
		on input the full upper triangle must contain the full
		upper triangle of the matrix r. on output the full upper
		triangle is unaltered, and the strict lower triangle
		contains the strict upper triangle (transposed) of the
		upper triangular matrix s such that

			   t   t                2     t
			  P *(J * J + lambda * D ) * P = s * s

	ldr: int scalar, positive, not less than N
		specifies the leading dimension of the array r

	ipvt: int array of length N
		defines the	permutation matrix p such that a*p = q*r.
		column j of p is column ipvt(j) of the identity matrix

	diag: real array of length N
		contains the diagonal elements of the matrix D

	qtb: real array of length N						 t
		contains the first n elements of the vector q * b

	delta: real scalar, positive
		specifies an upper bound on the euclidean norm of D*x

	par: real pointer, non-negative
		contains an initial estimate of the levenberg-marquardt
		parameter.

	x: real array of length N [output]
		contains the least squares solution of the system
		J*x = r, sqrt(par)*D*x = 0 for the output par

	sdiag: real array of length N [output]
		contains the diagonal elements of the upper triangular
		matrix s

	wa1, wa2: real array of length N							*/
{
#pragma region Variable Initialization

	// :: private variables of index N
	int index = INDEX;
	real d1, d2;

	int i, j, k, l;
	real sum;
	real parc, parl;
	real temp, paru, dwarf, epsmch;
	real gnorm;

	dwarf = dpmpar(2);
	epsmch = dpmpar(1);

	// :: local variables
	local int nsing;
	local int iter;
	local real fp, dxnorm, loc_temp;
	local int flag;

#pragma endregion

#pragma region Compute Gauss-Newton direction

	// > compute and store in x the gauss-newton direction. 
	//    if the jacobian is rank-deficient, obtain a least 
	//    squares solution
	/// => 650 us
	if (index == N) {
		nsing = N;
		for (j = 0; j < N; j++) {
			wa1[j] = qtb[j];
			if (fabs(r[j + j * ldr]) < 1e-8) {
				r[j + j * ldr] = 0.0;
			}
			if (r[j + j * ldr] == 0.0 && nsing == N) {
				nsing = j;
			}
			if (nsing < N) {
				wa1[j] = 0.0;
			}
		}

		if (nsing >= 1) {
			for (k = 1; k <= nsing; ++k) {
				j = nsing - k;
				wa1[j] /= r[j + j * ldr];
				temp = wa1[j];
				if (j >= 1) {
					for (i = 0; i < j; ++i) {
						wa1[i] -= r[i + j * ldr] * temp;
					}
				}
			}
		}

		for (j = 0; j < N; ++j) {
			l = ipvt[j] - 1;
			x[l] = wa1[j];
		}
	}

	loc_bar;  /// => 654 us

#pragma region [ Verification ]
#if 0
	if (index < N) printf("# x[%d] = %.10f\n", index, x[index]);
	return;
#endif
#pragma endregion

#pragma endregion

#pragma region Preparation

	/* > initialize the iteration counter. evaluate the function
		  at the origin, and test for acceptance of the
		  gauss-newton direction.							*/
	if (index < N) wa2[index] = diag[index] * x[index];

	loc_bar;

	if (index == N) {
		iter = 0;
		dxnorm = enorm(N, wa2);
		// TODO: private -> local costs 10 us ...
		fp = dxnorm - delta;
		parl = 0.0;
	}

	loc_bar;
	// ######################## Debug Switch ##########################
	if (fp <= p1 * delta) {
		*par = 0.0;  return;
	}

	/// => 668 us
	if (nsing >= N) {
		if (index < N) {
			l = ipvt[index] - 1;
			wa1[index] = diag[l] * (wa2[l] / dxnorm);
		}
		loc_bar;
		if (index == N) {
			for (j = 0; j < N; ++j) {
				sum = 0.0;
				if (j >= 1) {
					for (i = 0; i < j; ++i) {
						sum += r[i + j * ldr] * wa1[i];
					}
				}
				wa1[j] = (wa1[j] - sum) / r[j + j * ldr];
			}
			temp = enorm(N, wa1);
			parl = fp / delta / temp / temp;
		}
	}

	loc_bar;  /// => 680~682 us

#pragma region [ Verification ]
#if 0
	loc_bar;
	if (index == N) {
		printf("# parl = %.10f\n", parl);
		printf("# temp = %.10f\n", temp);
		printf("# delta = %.10f\n", delta);
		//printf("# nsing = %d\n", nsing);
	}
	return;
#endif
#pragma endregion

	// > calculate an upper bound, paru, for the zero of the function
	j = index;
	if (j < N) {
		sum = 0.0;
		for (i = 0; i <= j; ++i) {
			sum += r[i + j * ldr] * qtb[i];
		}
		l = ipvt[j] - 1;
		wa1[j] = sum / diag[l];
	}

	loc_bar;   /// => 682~683 us

	if (index == N) {
		gnorm = enorm(N, wa1);
		paru = gnorm / delta;
		if (paru == 0.) {
			paru = dwarf / min(delta, (real)p1) /* / p001 ??? */;
		}

		/// => 682~684 us

		// > if the input par lies outside of the interval(parl, paru), 
		//   set par to the closer endpoint
		// !! private -> local costs 10 us
		*par = max(*par, parl);
		*par = min(*par, paru);
		if (*par == 0.0) *par = gnorm / dxnorm;
	}

#pragma region [ Verification ]
#if 0
	if (index == N) {
		printf("# paru = %.10f\n", paru);
		printf("# par = %.10f\n", *par);
	}
	return;
#endif
#pragma endregion

#pragma endregion

#pragma region Iterations

	/// => 703~706 us
	for (;;) {

		loc_bar;

		if (index == N) {
			iter++;
			// > evaluate the function at the current value of par
			if (*par == 0.0) {
				// > computing max 
				d1 = dwarf, d2 = p001 * paru;
				*par = max(d1, d2);
			}
			loc_temp = sqrt(*par);
		}

		loc_bar;

		if (index < N) wa1[index] = loc_temp * diag[index];

#pragma region [ Verification ]
#if 0
		loc_bar;
		if (index < N) {
			printf("# wa1[%d] = %.10f\n", index, wa1[index]);
		}
		return;
#endif
#pragma endregion

		/// => 704~707 us
		qrsolv(r, ldr, ipvt, wa1, qtb, x, sdiag, wa2);

		loc_bar;
		/// => 772 us
		j = index;
		if (j < N) wa2[j] = diag[j] * x[j];
		loc_bar;
		/// => 772 us
		if (index == N) {
			dxnorm = enorm(N, wa2);
			temp = fp;
			fp = dxnorm - delta;
		}

		loc_bar;

#pragma region [ Verification ]
#if 0
		if (index == N) {
			printf("# fp = %.10f\n", fp);
		}
		return;
#endif
#pragma endregion

		/// => 782 us [private -> local]
		/* > if the function is small enough, accept the current value
			 of par. also test for the exceptional cases where parl
			 is zero or the number of iterations has reached 10.    */
		if (index == N) {
			flag = fabs(fp) <= p1 * delta
				|| (parl == 0. && fp <= temp && temp < 0.)
				|| iter == 10;
		}

		loc_bar;

		if (flag) return;

		/* > compute the newton correction
		   ::
		   ::            / ||d*x|| \2  ||d*x|| - delta
		   ::   par_c = | -------- |  -----------------
		   ::            \  ||y||  /        delta
		   ::     t
		   ::    r * y = x, fp = ||d*x|| - delta		*/
		j = index;
		if (j < N) {
			l = ipvt[j] - 1;
			wa1[j] = diag[l] * (wa2[l] / dxnorm);
		}

		loc_bar; /// => 784~785 us

		for (j = 0; j < N; j++) {
			if (index == N) {
				wa1[j] /= sdiag[j];

				/*	if (N > j + 1) {
						int i;
						for (i = j + 1; i < N; ++i) {
							wa1[i] -= r[i + j * ldr] * wa1[j];
						}
					}*/
			}
			loc_bar;
			if (1 && j < index && index < N) {
				wa1[index] -= r[index + j * ldr] * wa1[j];
			}
			loc_bar;
		}
		loc_bar; // TODO: without this line, next verification will fail
		/// => 787~789 us
		if (index == N) {
			temp = enorm(N, wa1);
			parc = fp / delta / temp / temp;
		}

#pragma region [ Verification ]
#if 0
		loc_bar;
		if (index == N) {
			printf("# temp = %.10f\n", temp);
			printf("# parc = %.10f\n", parc);
		}
		return;
#endif
#pragma endregion

		if (index == N) {
			// > depending on the sign of the function, update parl or paru.
			if (fp > 0.0) parl = max(parl, *par);
			if (fp < 0.0) paru = min(paru, *par);
			// > compute an improved estimate for par
			d1 = parl, d2 = *par + parc;
			*par = max(d1, d2);
		}

#pragma region [ Verification ]
#if 0
		loc_bar;
		if (index == N) {
			printf("# par = %.10f\n", *par);
		}
		return;
#endif
#pragma endregion

		loc_bar;  /// => 796~798 us
		//return;  // ###########################################################
	}

#pragma endregion
}

#endif
