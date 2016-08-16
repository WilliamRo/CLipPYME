#ifndef QRFAC_CL_INCLUDED
#define QRFAC_CL_INCLUDED

#include "clminpack.clh"

// [Verified]
void qrfac(local real *a, int lda, int pivot, local int *ipvt,
		   local real *rdiag, local real *acnorm, local real *wa,
		   local int * inta, local real *reala, local real *wa0)
	/* QRFAC

	Uses householder transformations with column
	 pivoting (optional) to compute a qr factorization of the
	 m by n matrix a. That is, qrfac determines an orthogonal
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
		then acnorm can coincide with rdiag.					*/
{
#pragma region Variable Initialization

	int index = INDEX;

	int i, j, k, jp1, minmn;
	real d1, sum, temp;

	// > get epsmch
	if (index == 0) reala[QR_EPSMCH] = dpmpar(1);

#pragma endregion

#pragma region Preparation
	/// S9150, double, GS11x11: 40~41 us
	// > compute the initial column norms and initialize several arrays.
	if (index < N)
	{
		acnorm[index] = enorm_w(M, &a[index * lda]);
		rdiag[index] = acnorm[index];
		wa[index] = rdiag[index];
		if (pivot)
		{
			ipvt[index] = index + 1;
		}
	}
	/// S9150, double, GS11x11: 47~48 us

#pragma endregion

#pragma region Main Loop

	// > reduce a to r with householder transformations
	minmn = min(M, N);
	/// S9150, double, GS11x11: 47~48 us
	i = index;
	for (j = 0; j < minmn; j++)
	{
		loc_bar;

		if (pivot)
		{
			// > bring the column of largest norm into the pivot position
			if (index == 0)
			{
				inta[KMAX] = j;
				for (k = j; k < N; ++k)
					if (rdiag[k] > rdiag[inta[KMAX]]) inta[KMAX] = k;
			}

			loc_bar;

			if (inta[KMAX] != j)
			{
				reala[QRFAC_TEMP] = a[i + j * lda];
				a[i + j * lda] = a[i + inta[KMAX] * lda];
				a[i + inta[KMAX] * lda] = reala[QRFAC_TEMP];

				if (index == N)
				{
					rdiag[inta[KMAX]] = rdiag[j];
					wa[inta[KMAX]] = wa[j];
					k = ipvt[j];
					ipvt[j] = ipvt[inta[KMAX]];
					ipvt[inta[KMAX]] = k;
				}
			}
		}
		/// S9150, double, GS11x11: 52~53 us

		loc_bar;

		// > compute the householder transformation to reduce the 
		//    j-th column of a to a multiple of the j-th unit vector
		if (i == N) reala[AJNORM] = enorm_w(M - (j + 1) + 1, &a[j + j * lda]);
		/// S9150, double, GS11x11: 58~59 us
		loc_bar;

#pragma region [ Verification ]
#if 0
		if (index == N) printf("# reala[AJNORM] = %f\n", reala[AJNORM]);
#endif
#pragma endregion

		if (reala[AJNORM] != 0.0)
		{
			i = index;
			if (i == 0 && a[j + j * lda] < 0.0)
				reala[AJNORM] = -reala[AJNORM];

			loc_bar;

			if (j <= i) a[i + j * lda] /= reala[AJNORM];
			if (i == j) a[j + j * lda] += 1.0;

#pragma region [ Verification ]
#if 0
			if (index == N) printf("# a[j + j * lda] = %f\n", a[j + j * lda]);
#endif
#pragma endregion

			loc_bar; 
			/// S9150, double, GS11x11: 60~61 us

			// > apply the transformation to the remaining columns 
			//    and update the norms
			if (j < N - 1) {
				k = index;
				if (j < k && k < N) {
					sum = 0.0;
					for (i = j; i < M; i++){
						sum = mad(a[i + j * lda], a[i + k * lda], sum);
					}
					wa0[k] = sum / a[j + j * lda];
				}
				loc_bar;

				// > find subgroup id
				k = index % (N - 1 - j);			// subgroup id
				i = (index - k) / (N - 1 - j);		// local id in subgroup
				jp1 = M / (N - 1 - j);
				if (k < (M % (N - 1 - j))) jp1++;
				k = k + j + 1;
				/// S9150, double, GS11x11: 75~76 us
				for (i = i + j; i < M; i += jp1)
					a[i + k * lda] -= wa0[k] * a[i + j * lda];

				i = k = index;
				loc_bar;
				/// S9150, double, GS11x11: 78~80 us
				if (j < k && k < N && pivot && rdiag[k] != 0.0) {
					real temp1 = a[j + k * lda] / rdiag[k];

					// Computing MAX 
					d1 = 1.0 - temp1 * temp1;
					rdiag[k] *= sqrt((fmax((real)0.0, d1)));

					// Computing 2nd power 
					d1 = rdiag[k] / wa[k];
					if (p05 * (d1 * d1) <= reala[QR_EPSMCH]) {
						rdiag[k] = enorm(M - (j + 1), &a[jp1 + k * lda]);
						wa[k] = rdiag[k];
					}
				}
			}
		}

		if (i == N) rdiag[j] = -reala[AJNORM];
		/// S9150, double, GS11x11: 82 us
	}

#pragma endregion
}

#endif