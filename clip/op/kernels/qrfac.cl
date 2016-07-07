#ifndef QRFAC_CL_INCLUDED
#define QRFAC_CL_INCLUDED

#include "clminpack.h"

// [Verified]
void qrfac(local real *a, int lda, int pivot, local int *ipvt,
		   local real *rdiag, local real *acnorm, local real *wa)
	/*
	uses householder transformations with column
	 pivoting (optional) to compute a qr factorization of the
	 m by n matrix a. that is, qrfac determines an orthogonal
	 matrix q, a permutation matrix p, and an upper trapezoidal
	 matrix r with diagonal elements of nonincreasing magnitude,
	 such that a*p = q*r. the householder transformation for
	 column k, k = 1,2,...,min(m,n), is of the form

									 t
			   i - (1 / u(k)) * u * u

	 where u has zeros in the first k-1 positions

	Parameters
	----------
	a: real array of length M by N.
		on input a contains the matrix for which the qr
		factorization is to be computed. on output the strict
		upper trapezoidal part of a contains the strict upper
		trapezoidal part of r, and the lower trapezoidal part
		of a contains a factored form of q (the non-trivial
		elements of the u vectors described above)

	lda: int scalar, positive, not less than M
		specifies the leading dimension of the array a

	pivot: logical
		if pivot is set true, then column pivoting is enforced.
		if pivot is set false, then no column pivoting is done

	ipvt: int array of length N [output]
		ipvt defines the permutation matrix p such that a*p = q*r.
		column j of p is column ipvt(j) of the identity matrix.
		if pivot is false, ipvt is not referenced

	rdiag: real array of length N [output]
		contains the diagonal elements of r

	acnorm: real array of length N [output]
		contains the norms of the corresponding columns of the
		input matrix a. if this information is not needed,
		then acnorm can coincide with rdiag.

	waN: real array of length N

	waM: real array of length M									*/
{
#pragma region Variable Initialization

	int glb_i = ggi(0);
	int glb_j = ggi(1);
	int index = glb_i * ROI_L + glb_j;

	real d1;
	int i, j, k, jp1;
	real sum;
	local real temp;
	local int minmn, kmax;
	local real epsmch;
	local real ajnorm;

	// > get epsmch
	if (index == 0) epsmch = dpmpar(1);

#pragma endregion

#pragma region Preparation

	// > compute the initial column norms and initialize several arrays.
	if (index < N)
	{
		//enorm(M, &a[index * lda], &acnorm[index], waM);
		acnorm[index] = enorm(M, &a[index * lda]);
		rdiag[index] = acnorm[index];
		wa[index] = rdiag[index];
		if (pivot)
		{
			ipvt[index] = index + 1;
		}
	}

#pragma endregion

#pragma region Main Loop

	// > reduce a to r with householder transformations
	if (index == N + 1) minmn = min(M, N);

	loc_bar;

	i = index;
	for (j = 0; j < minmn; j++)
	{
		if (pivot)
		{
			// > bring the column of largest norm into the pivot position
			if (index == 0)
			{
				kmax = j;
				for (k = j; k < N; ++k)
					if (rdiag[k] > rdiag[kmax]) kmax = k;
			}

			loc_bar;

			if (kmax != j)
			{
				temp = a[i + j * lda];
				a[i + j * lda] = a[i + kmax * lda];
				a[i + kmax * lda] = temp;

				if (index == 0)
				{
					rdiag[kmax] = rdiag[j];
					wa[kmax] = wa[j];
					k = ipvt[j];
					ipvt[j] = ipvt[kmax];
					ipvt[kmax] = k;
				}
			}
		}

		loc_bar;

		// > compute the householder transformation to reduce the 
		//    j-th column of a to a multiple of the j-th unit vector
		if (i == 0) ajnorm = enorm(M - (j + 1) + 1, &a[j + j * lda]);

		loc_bar;

		if (ajnorm != 0.0)
		{
			if (i == 0 && a[j + j * lda] < 0.0)
				ajnorm = -ajnorm;

			loc_bar;

			if (j <= i) a[i + j * lda] /= ajnorm;
			if (i == 0) a[j + j * lda] += 1.0;

			loc_bar;

			// > apply the transformation to the remaining columns 
			//    and update the norms
			jp1 = j + 1;
			if (N > jp1)
			{
				for (k = jp1; k < N; k++)
				{
					if (i == 0)
					{
						sum = 0.0;
						for (i = j; i < M; ++i)
							sum += a[i + j * lda] * a[i + k * lda];
						temp = sum / a[j + j * lda];
						i = index;
					}

					loc_bar;

					if (j <= i) a[i + k * lda] -= temp * a[i + j * lda];

					loc_bar;

					if (i == 0 && pivot && rdiag[k] != 0.0)
					{
						real temp1 = a[j + k * lda] / rdiag[k];
						/* Computing MAX */
						d1 = 1.0 - temp1 * temp1;
						rdiag[k] *= sqrt((max((real)0.0, d1)));
						/* Computing 2nd power */
						d1 = rdiag[k] / wa[k];
						if (p05 * (d1 * d1) <= epsmch) {
							rdiag[k] = enorm(M - (j + 1), &a[jp1 + k * lda]);
							wa[k] = rdiag[k];
						}
					}
				}
			}
		}
		if (i == 0) rdiag[j] = -ajnorm;
	}

#pragma endregion
}

#endif
