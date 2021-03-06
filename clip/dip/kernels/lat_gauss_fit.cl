#ifndef LAT_GAUSS_FIT_CL_INCLUDED
#define LAT_GAUSS_FIT_CL_INCLUDED

#include "clminpack.clh"

kernel void fit(global real *img, global real *sigma,
				global real *X, global real *Y,
				global real *x0, global int *wa,
				global real *output, int w,
				global int *roi_num, int CU_count)
	/* FIT

	Parameters
	----------
	img:
		an image containing serveral ROIs, usually stay in
		global memory until ROIs have been traversed

	sigma:
		image sigma used for evaluating fcn, usually stay in
		global memory until ROIs have been traversed

	X and Y: real array of length ROI_L * group_size
		index grid of ROI times by voxel size in nanometers

	x0: real array of length N * group_size
		on input x0 contains start parameters for fitting.
		on output x0 contains the final estimation

	wa: int array of length 2 * group_size
		on output wa[0 + 2 * group_size] = info,
				  wa[1 2 * group_size] = nfev

	output: real array

	w: int scalar
		image width										*/
{
	// > get NDRange info
	int index = INDEX;
	int groupID = ggri(0) + ggri(1) * gng(0);
	int gSize = GS;
	int ROI_NUM = roi_num[1];
	int groupCount = gng(0) * gng(1);

	//int CU_count = 12;

#pragma region Test
#if 0
	if (index == 0)
		printf("# groupID = %d\n", groupID);
	return;
#endif
#pragma endregion

	// ======================================================
	// > declarations
	// >> private variables
	int loc_i, loc_j, roi_i, roi_j, i;
	int dX = X[1] - X[0];
	int dY = Y[1] - Y[0];
	//int maxfev = 200 * (N + 1);
	int maxfev = 300;
	int mode = 1;
	int ldfjac = M;

	real ftol = 1.49012e-08;
	real xtol = 1.49012e-08;
	real gtol = 0.0;
	real epsfcn = 2.22044604925e-16;
	real factor = 100.0;

	// >> local variables < (M + 4M + 6N + N * M) * sizeof(real) Byte
	//    use double buffer
#define K 1
	local int inta[INTN * K];
	local int ipvt[N * K];
	local int nfev[K];

	local real reala[REALN * K];
	local real fjac[N * M * K];
	local real fvec[M * K];
	local real x[N * K];
	local real diag[N * K];
	local real qtf[N * K];
	local real wa0[N * K];
	local real wa1[N * K];
	local real wa2[N * K];
	local real wa3[N * K];
	local real wa4[M * K];

	local fcndata p[K];

	loc_bar;	/// S9150, double, GS64: 2 us

	int cursor, flag = 0;
	// ======================================================
	for (; groupID < ROI_NUM; groupID += groupCount) {
		cursor = groupID;
		//cursor = groupID;

		//flag = groupID / CU_count % 2;
		//flag = 0;
		//if (index == 0) printf("#[%d] %d\n", groupID, flag);

		// > wrap fcn data
#ifdef M_WORKERS_DIM_2
		loc_i = gli(0);
		loc_j = gli(1);
		if (loc_j == 0) p[flag].X[loc_i] = X[loc_i + ROI_L * cursor];
		else if (loc_j == 1)
			p[flag].Y[loc_i] = Y[loc_i + ROI_L * cursor];


		loc_bar;

		roi_i = p[flag].X[loc_i] / dX;
		roi_j = p[flag].Y[loc_j] / dY;
		p[flag].y[index] = img[w * roi_i + roi_j];
		p[flag].sigma[index] = sigma[w * roi_i + roi_j];

#else
	//! each work group must contain not less than ROI_L work items
		if (index < ROI_L) {
			p.X[index] = X[index + ROI_L * groupID];
			p.Y[index] = Y[index + ROI_L * groupID];
		}
		loc_bar;
		/// S9150, double, GS64: 4 us
		for (i = index; i < M; i += gSize) {
			// >>> restore loc_i and loc_j
			loc_i = i / ROI_L;
			loc_j = i - ROI_L * loc_i;
			// >>> locate in img
			roi_i = p.X[loc_i] / dX;
			roi_j = p.Y[loc_j] / dY;
			p.y[i] = img[w * roi_i + roi_j];
			p.sigma[i] = sigma[w * roi_i + roi_j];
		}
#endif
		/// S9150, double, GS64: 12 us
		/// S9150, double, GS11x11: 10~11 us
		// > set x
		if (index < N) x[index] = x0[index + N * cursor];

		loc_bar;	/// S9150, double, GS64: 13~14 us
					/// S9150, double, GS11x11: 11~12 us
		// ======================================================
		// > call lmdif
		lmdif(&p[flag], x + flag * N, fvec + flag * M, ftol, xtol, gtol,
			  maxfev, epsfcn, diag + flag * N, mode, factor,
			  nfev + flag, fjac + flag * N * M, ldfjac,
			  ipvt + flag * N, qtf + flag * N, wa1 + flag * N,
			  wa2 + flag * N, wa3 + flag * N, wa4 + flag * M,
			  inta + flag * INTN, reala + flag * REALN,
			  wa0 + flag * N);
		//return;//##############################################
		// ======================================================

#pragma region [ Verification ]
#if 0
		if (index == N) {
			real clres = enorm(M, fvec);
			printf("# ||fvec|| = %.10f\n", clres);
			printf("# nfev = %d\n", nfev);
		}
#endif
#pragma endregion

		loc_bar;

		if (index < N) x0[index + N * cursor] = x[index];
		if (index == N) {
			wa[0 + 2 * cursor] = inta[INFO + flag * INTN];
			wa[1 + 2 * cursor] = nfev[flag];
			output[cursor] = reala[FNORM + flag * REALN];
		}

		loc_bar;
	}
}

void fcn_mn(local fcndata *p,
			local real *x,
			local real *fvec)
{
#ifdef M_WORKERS_DIM_2
	// > get global index
	int loc_i = gli(0);
	int loc_j = gli(1);
	int index = INDEX;

	// > evaluate fvec
	real dx = p->X[loc_i] - x[1];
	real dy = p->Y[loc_j] - x[2];
	/// S9150, double, GS11x11: 6 us
	/*fvec[index] = (p->y[index] - (x[0] * exp(-(dx * dx + dy * dy) /
		(2 * x[3] * x[3])) + x[4] + x[5] * dx + x[6] * dy)) / p->sigma[index];*/

		/// S9150, double, GS11x11: 5~6 us
	fvec[index] = (p->y[index] - mad(x[0], exp(-(mad(dx, dx, dy * dy)) /
		(2 * x[3] * x[3])), mad(x[6], dy, mad(x[5], dx, x[4])))) / p->sigma[index];
#else
	// > get NDRange infos
	int index = INDEX;
	int gSize = GS;
	// > declare private variables
	int i, loc_i, loc_j;
	real dx, dy;
	real x0 = x[0];
	real x1 = x[1];
	real x2 = x[2];
	real x3 = x[3];
	real x4 = x[4];
	real x5 = x[5];
	real x6 = x[6];

	// > evaluate fvec
	for (i = index; i < M; i += gSize) {
		// >> restore loc_i and loc_j
		loc_i = i / ROI_L;
		loc_j = i - ROI_L * loc_i;

		dx = p->X[loc_i] - x1;
		dy = p->Y[loc_j] - x2;

		fvec[i] = (p->y[i] - (x0 * exp(-(dx * dx + dy * dy) /
			(2 * x3 * x3)) + x4 + x5 * dx + x6 * dy)) / p->sigma[i];
	}
#endif
}

#endif // !LAT_GAUSS_FIT_CL_INCLUDED
