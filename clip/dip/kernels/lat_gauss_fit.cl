#ifndef LAT_GAUSS_FIT_CL_INCLUDED
#define LAT_GAUSS_FIT_CL_INCLUDED

#include "clminpack.h"

kernel void fit(global real *img, global real *sigma,
				global real *X, global real *Y,
				global real *x0, int w)
	/* FIT

	Parameters
	----------
	img:
		an image containing serveral ROIs, usually stay in
		global memory until ROIs have been traversed

	sigma:
		image sigma used for evaluating fcn, usually stay in
		global memory until ROIs have been traversed

	X and Y:
		index grid of ROI times by voxel size in nanometers

	x0: real array of length N
		on input x0 contains start parameters for fitting.
		on output x0 contains the final estimation

	w:
		image width										*/
{
	// > get global index
	int glb_i = ggi(0);
	int glb_j = ggi(1);
	int index = glb_i * ROI_L + glb_j;
	// > restore ROI index
	int roi_i = X[glb_i] / (X[1] - X[0]);
	int roi_j = Y[glb_j] / (Y[1] - Y[0]);

	// ==========================================
	// > declarations
	local fcndata p;
	local real x[N];
	local real fvec[M];
	real ftol = 1.49012e-08;
	real xtol = 1.49012e-08;
	real gtol = 0.0;
	int maxfev = 200 * (N + 1);
	real epsfcn = 2.22044604925e-16;

	local real diag[N];
	int mode = 1;
	real factor = 100.0;
	local int nfev;
	local real fjac[N * M];
	int ldfjac = M;
	local int ipvt[N];
	local real qtf[N];
	local real wa1[N];
	local real wa2[N];
	local real wa3[N];
	local real wa4[M];
	local int info;

	// ==========================================
	// > wrap fcn data
	// >> X and Y
	if (glb_j == 0) p.X[glb_i] = X[glb_i];
	else if (glb_j == 1) p.Y[glb_i] = Y[glb_i];
	// >> y
	p.y[index] = img[w * roi_i + roi_j];
	// >> sigma
	p.sigma[index] = sigma[w * roi_i + roi_j];

	// > set x
	if (index < N) x[index] = x0[index];

	// > synchronize [ Verified ]
	loc_bar;

	// ==========================================
	// > call lmdif
	lmdif(&p, x, fvec, ftol, xtol, gtol, maxfev, epsfcn,
		  diag, mode, factor, &nfev, fjac, ldfjac, ipvt,
		  qtf, wa1, wa2, wa3, wa4, &info);

	// ==========================================
}

void fcn_mn(local fcndata *p,
			local real *x,
			local real *fvec)
{
	// [Verified]
	// > get global index
	int glb_i = ggi(0);
	int glb_j = ggi(1);
	int index = glb_i * ROI_L + glb_j;

	// > evaluate fvec
	real dx = p->X[glb_i] - x[1];
	real dy = p->Y[glb_j] - x[2];
	fvec[index] = (p->y[index] - (x[0] * exp(-(dx * dx + dy * dy) /
		(2 * x[3] * x[3])) + x[4] + x[5] * dx + x[6] * dy)) / p->sigma[index];
}

#pragma region Test Functions

kernel void test(global int * x)
{
	int i = ggi(0);
	printf("test >> [%d] %d\n", i, x[i]);
}

kernel void test_group()
{
	int id = ggi(0);
	int grp_id = ggri(0);
	glb_bar;

	printf("[%d]glo_siz(%d) ", id, (int)ggs(0));
	if (id == 0) printf("\n");
	glb_bar;

	printf("[%d]loc_siz(%d) ", id, (int)gls(0));
	if (id == 0) printf("\n");
	glb_bar;

	printf("[%d]loc_id(%d)  ", id, (int)gli(0));
	if (id == 0) printf("\n");
	glb_bar;

	printf("[%d]num_grp(%d) ", id, (int)gng(0));
	if (id == 0) printf("\n");
	glb_bar;

	printf("[%d]grp_id(%d)  ", id, grp_id);
	if (id == 0) printf("\n");
	glb_bar;

	printf("[%d]glo_off(%d) ", id, (int)ggo(0));
	if (id == 0) printf("\n");
}

#pragma endregion

#endif // !LAT_GAUSS_FIT_CL_INCLUDED
