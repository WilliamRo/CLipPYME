#ifndef QRSOLV_CL_INCLUDED
#define QRSOLV_CL_INCLUDED

#include "clminpack.h"

void qrsolv(local real *r, int ldr, local int *ipvt,
			local real *diag, local real *qtb, local real *x,
			local real *sdiag, local real *wa)
	/* QRSOLV

	Solves the linear least square problem:

				||   /  a  \      / b \  || 2
			min ||  |      | x - |    |  ||
			 x  ||  \  d  /      \ 0 /   || 2

	 in which a is an M by N matrix, d is an N by N diagonal matrix,
	 b is an M-vector. The necessary information must be provided:

		(1) the q-r factorization with column pivoting of a:

				a * p = q * r
												t
		(2) the first N elements of the vector q * b

	 With these information, we have
				t                          t
			 / q      \  /  a  \    / r * p \
			|         | |      | = |    0   |
			\      i /  \  d  /    \    d  /

	 This routine uses a set of givens transformation to convert
	 the right-most matrix to an upper triangular matrix and
	 then use back substitution to obtain the solution

	 Parameters
	 ----------
	 r: real array of length N by N
		on input the full upper triangle must contain the full
		upper triangle of the matrix r. on output the full upper
		triangle is unaltered, and the strict lower triangle
		contains the strict upper triangle (transposed) of the
		upper triangular matrix s

	 ldr: int scalar, positive, not less than N
		specifies the leading dimension of the array r

	 ipvt: int array of lengtg N
		defines the permutation matrix p such that a*p = q*r.
		column j of p is column ipvt(j) of the identity matrix

	 diag: real array of length N
		contains the diagonal elements of the matrix d

	 qtb: real array of length N					 t
		contains the first N elements of the vector q * b

	 x: real array of length N [output]
		contains the least squares solution of the system
			a * x = b, d * x = 0

	 sdiag: real array of length N [output]
		contains the diagonal elements of the upper triangular
		matrix s satisfies
			 t    t                    t
			p * (a * a + d * d) * p = s * s

		In effect, s is the Cholesky factorization of
		the left matrix

	wa: real array of length N							*/
{
#pragma region Variables Initialization

	// :: private variables of index N
	int index = INDEX;

	int i, j, k, l;
	real sum, temp;
	real qtbpj;

	// :: local variables
	local real cos, sin;
	local int nsing;

#pragma endregion

#pragma region Preparation

	/// 703~706 us
	/* > copy r and qtb to preserve input and initialize s.
		 in particular, save the diagonal elements of r in x.	*/
#ifdef GREAT_M_DIM_2
	i = gli(0);
	j = gli(1);
	if (j < N) {
		if (j <= i && i < N)
			r[i + j * ldr] = r[j + i * ldr];
		if (i == j) {
			x[j] = r[j + j * ldr];
			wa[j] = qtb[j];
		}
	}
#else
	j = index;
	if (j < N) {
		for (i = j; i < N; i++) {
			r[i + j * ldr] = r[j + i * ldr];
		}
		x[j] = r[j + j * ldr];
		wa[j] = qtb[j];
	}
#endif

#pragma endregion

#pragma region Givens Rotation

	/* > eliminate the diagonal matrix d using a givens rotation
								  t              _    t
			   n by n      / r * p \           / r * p \
		 (m - n) by n     |    0    | = q_g * |    0    |
			   n by n      \   d   /           \   0   /     	*/

	loc_bar;  /// 704~706 us

	for (j = 0; j < N; j++) {
		/* > prepare the row of d to be eliminated, locating the
			 diagonal element using p from the qr factorization. */
		l = ipvt[j] - 1;
		if (diag[l] != 0.0) {
			k = index;
			if (j <= k && k < N) sdiag[k] = 0.0;
			if (k == N) {
				sdiag[j] = diag[l];
				qtbpj = 0.0;
			}

			/* > the transformations to eliminate the row of d
				 modify only a single element of (q transpose)*b
				 beyond the first N, which is initially zero.    */
			for (k = j; k < N; ++k) {

				loc_bar;  /// 705~708 us

				/* > determine a givens rotation which eliminates the
					 appropriate element in the current row of d. */
				if (sdiag[k] != 0.0) {
					if (index == N) {
						if (fabs(r[k + k * ldr]) < fabs(sdiag[k])) {
							real cotan;
							cotan = r[k + k * ldr] / sdiag[k];
							sin = p5 / sqrt(p25 + p25 * (cotan * cotan));
							cos = sin * cotan;
						}
						else {
							real tan;
							tan = sdiag[k] / r[k + k * ldr];
							cos = p5 / sqrt(p25 + p25 * (tan * tan));
							sin = cos * tan;
						}
						/* > compute the modified diagonal element of r and
							 the modified element of ((q^t * b, 0).      */
						temp = cos * wa[k] + sin * qtbpj;
						/// 705~708 us
						qtbpj = -sin * wa[k] + cos * qtbpj;
						wa[k] = temp;
						/// 722~724 us [private & local issue]
						r[k + k * ldr] = cos * r[k + k * ldr] + sin * sdiag[k];
					}

					loc_bar;  /// 725~727 us

					i = index;
					if (N > k + 1 && k < i && i < N) {
						temp = cos * r[i + k * ldr] + sin * sdiag[i];
						/* UNDER SOME PRECONDITION
							On NVDIA 980M GPU, in routine

							  sdiag[i] = -sin * r[i + k * ldr] + cos * sdiag[i];

							 local sdiag will cost about 45 us more than
							 private sdiag. Notice that if sdiag is private,
							 all time span in this kernel till this line is
							 about merely 20 us.
						*/
						sdiag[i] = -sin * r[i + k * ldr] + cos * sdiag[i];
						r[i + k * ldr] = temp;
					}
				}
			}
		}

		/// 770 us
		loc_bar;
		/* > store the diagonal element of s and restore
			 the corresponding diagonal element of r     */
		if (index == N) {
			sdiag[j] = r[j + j * ldr];
			r[j + j * ldr] = x[j];
		}
	}

#pragma endregion

#pragma region Solve System

	/* > solve the triangular system for z. if the system is singular,
		 then obtain a least squares solution
						  t
	 :: r * z = qtb, z = p * x and qtb is stored in wa          */

	loc_bar;  /// 774 us

	if (index == N) {
		nsing = N;
		for (j = 0; j < N; ++j) {
			if (sdiag[j] == 0. && nsing == N) {
				nsing = j;
			}
			if (nsing < N) {
				wa[j] = 0.0;
			}
		}
		/*
		if (nsing >= 1) {
			for (k = 1; k <= nsing; ++k) {
				j = nsing - k;
				sum = 0.;
				if (nsing > j + 1) {
					for (i = j + 1; i < nsing; ++i) {
						sum += r[i + j * ldr] * wa[i];
					}
				}
				wa[j] = (wa[j] - sum) / sdiag[j];
			}
		}*/
	}

	loc_bar;  /// 770 us ???

	if (nsing >= 1) {
		for (k = 1; k <= nsing; ++k) {

			loc_bar;

			j = nsing - k;
			if (index == N) wa[j] /= sdiag[j];

			loc_bar;

			if (0 <= index && index < j)
				wa[index] -= wa[j] * r[j + index * ldr];
		}
	}

	loc_bar;

#pragma region [ Verification ]
#if 0
	//if (index == N)
	//	printf("# nsing = %d\n", nsing);
	if (index < N) {
		printf("# wa[%d] = %.10f\n", index, wa[index]);
	}
	return;
#endif
#pragma endregion

	/// 777 us

	// > permute the components of z back to components of x
	j = index;
	if (j < N) {
		l = ipvt[j] - 1;
		x[l] = wa[j];
	}

	/// 772 us ???

#pragma endregion
}

#endif
