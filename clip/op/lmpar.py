# coding=utf-8
########################################################################
#
#   Created: June 20, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from enorm import euclid_norm as enorm
from dpmpar import get_machine_parameter as dpmpar
from qrsolv import qr_solve

from utility import data_type
import utility

# region: Module Parameters

p1 = data_type(0.1)
p001 = data_type(0.001)

dwarf = dpmpar(2)

wa1 = None
wa2 = None
x = None
sdiag = None


# endregion: Module Parameters

def lm_lambda(n, r, ldr, ipvt, diag, qtb, delta, lam):
    """
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

     Let a = J, d = D, b = -r, x = p, we denoted the optimization
     problem as :

             ||   /         a         \       /  b  \  ||
         min ||  |                    | x -  |      |  ||
          x  ||  \  sqrt(lambda) * d /       \  0  /   ||


    Parameters
    ----------
        n: int
            a positive integer input variable set to the order of r
        r: ndarray
            an n by n array. on input the full upper triangle must
            contain the full upper triangle of the matrix r. on output
            the full upper triangle is unaltered, and the strict lower
            triangle contains the strict upper triangle (transposed)
            of the upper triangular matrix s such that

                    t   t                2     t
                   P *(J * J + lambda * D ) * P = s * s

        ldr: int
            a positive integer input variable not less than n which
            specifies the leading dimension of the array r
        ipvt: ndarray
            an integer input array of length n which defines the
            permutation matrix p such that a*p = q*r. column j of p
            is column ipvt(j) of the identity matrix
        diag: ndarray
            an input array of length n which must contain the diagonal
            elements of the matrix D
        qtb: ndarray
            an input array of length n which must contain the first
            n elements of the vector (q transpose)*b
        delta: float
            a positive input variable which specifies an upper
            bound on the euclidean norm of D*x
        lam: float
            a non-negative variable containing an initial estimate
            of the levenberg-marquardt parameter.

    Returns
    -------
        lam: float
            final estimate
        x: ndarray
            an output array of length n which contains the least
            squares solution of the system J*x = r, sqrt(lam)*D*x = 0,
            for the output lam
        sdiag: ndarray
            an output array of length n which contains the
            diagonal elements of the upper triangular matrix s

    """

    # region : Initialize parameters
    # ----------------------------------------
    global p1, p001, dwarf
    global wa1, wa2, x, sdiag

    if wa1 is None or wa1.size is not n:
        wa1 = np.zeros(n, data_type)
    if wa2 is None or wa2.size is not n:
        wa2 = np.zeros(n, data_type)
    if x is None or x.size is not n:
        x = np.zeros(n, data_type)
    if sdiag is None or sdiag.size is not n:
        sdiag = np.zeros(n, data_type)

    # ----------------------------------------
    # endregion : Initialize parameters

    # region : Compute Gauss-Newton direction
    # ------------------------------------------
    # :: stored in x. If the jacobian is rank-deficient,
    #    obtain a least squares solution :
    # ::        t
    # ::   R * P * x = -qtb
    nsing = n
    for j in range(n):
        wa1[j] = qtb[j]
        #
        if np.abs(r[j + j * ldr]) < 1e-8:
            r[j + j * ldr] = 0.0
        if r[j + j * ldr] == 0.0 and nsing is n:
            nsing = j
        if nsing < n:
            wa1[j] = 0.0

    # :: solving R * z = qtb using back substitution
    if nsing >= 1:
        for k in range(1, nsing + 1):
            # ::         wa1[j] - z[j+1]*r[j][j+1] -...- z[n]*r[j][n]
            # :: z[j] = ----------------------------------------------
            # ::                           r[j][j]
            j = nsing - k
            wa1[j] /= r[j + j * ldr]
            temp = wa1[j]
            if j >= 1:
                for i in range(j):
                    wa1[i] -= r[i + j * ldr] * temp
                    # if abs(wa1[i]) > 1e5:
                    #     aaa = 1

    # ::      t
    # :: x = P * z
    for j in range(n):
        l = ipvt[j] - 1
        x[l] = wa1[j]

    if utility.lam_trace:
        print ">> ||p^{GN}|| = %.10f" % enorm(x)
    # ------------------------------------------
    # endregion : Compute Gauss-Newton direction

    # region : Preparation
    # ------------------------------------------------
    # > initialize the iteration counter
    iter = 0
    # > evaluate the function at the origin, and test
    #   for acceptance of the gauss-newton direction
    for j in range(n):
        wa2[j] = diag[j] * x[j]
    dxnorm = enorm(wa2)
    # :: ||x||_2 = Delta + epsilon is acceptable
    fp = dxnorm - delta
    if fp <= p1 * delta:
        lam = data_type(0.0)
        return [lam, x, sdiag]
    # ------------------------------------------------
    # endregion : Preparation

    # region : Set bound
    # TODO: make comments
    # :: f(lam) = || D * x ||_2 - delta
    #    A root-finding Newton's method will be performed
    # :: If the jacobian is not rank deficient, the newton step provides
    #    a lower bound, lam_l, for the zero of the function.
    #    Otherwise set this bound to zero
    lam_l = data_type(0.0)
    if nsing >= n:
        for j in range(n):
            l = ipvt[j] - 1
            # :: wa2 stores D * x in which x is gauss-newton direction
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        # :: wa1 stores ...
        for j in range(n):
            sum = data_type(0.0)
            if j >= 1:
                for i in range(j):
                    sum += r[i + j * ldr] * wa1[i]
            wa1[j] = (wa1[j] - sum) / r[j + j * ldr]

        temp = enorm(wa1)
        lam_l = fp / delta / temp / temp

    # > calculate an upper bound, lam_u, for the zero of the function
    for j in range(n):
        sum = 0.0
        for i in range(j + 1):
            sum += r[i + j * ldr] * qtb[i]
        l = ipvt[j] - 1
        wa1[j] = sum / diag[l]
    gnorm = enorm(wa1)
    lam_u = gnorm / delta
    if lam_u == 0.0:
        lam_u = dwarf / min(delta, p1)

    # > if the input lam lies outside of the interval (lam_l, lam_u)
    #   set lam to the closer endpoint
    lam = max(lam, lam_l)
    lam = min(lam, lam_u)
    if lam == 0.0:
        lam = gnorm / dxnorm

    # endregion : Set bound

    # region : Iteration

    while True:
        iter += 1

        if utility.lam_trace:
            print '>> Step %d, lam ∈(%.8f, %.8f):' % (iter,
                                                     lam_l, lam_u)

        # > evaluate the function at the current value of lam
        if lam == 0.0:
            d1 = dwarf
            d2 = p001 * lam_u
            lam = max(d1, d2)
        temp = np.sqrt(lam)
        for j in range(n):
            wa1[j] = temp * diag[j]

        if utility.lam_trace:
            print '   lam = %.8f' % lam

        qr_solve(n, r, ldr, ipvt, wa1, qtb, x, sdiag)
        for j in range(n):
            wa2[j] = diag[j] * x[j]
        dxnorm = enorm(wa2)
        temp = fp
        fp = dxnorm - delta

        if utility.lam_trace:
            print '   Dx - delta = %.8f' % fp

        # > if the function is small enough, accept the current value
        #   of lam. also test for the exceptional cases where lam_l
        #   is zero or the number of iterations has reached 10
        if np.abs(fp) <= p1 * delta \
                or (lam_l == 0.0 and fp <= temp and temp < 0.0) \
                or iter is 10:
            return [lam, x, sdiag]

        # > compute the newton correction
        # ::
        # ::            / ||d*x|| \2  ||d*x|| - delta
        # ::   lam_c = | -------- |  -----------------
        # ::            \  ||y|| /         delta
        # ::     t
        # ::    r * y = x, fp = ||d*x|| - delta
        for j in range(n):
            l = ipvt[j] - 1
            wa1[j] = diag[l] * (wa2[l] / dxnorm)
        for j in range(n):
            wa1[j] /= sdiag[j]
            temp = wa1[j]
            if n > j + 1:
                for i in range(j + 1, n):
                    wa1[i] -= r[i + j * ldr] * temp

        temp = enorm(wa1)
        lam_c = fp / delta / temp / temp

        # > depending on the sign of the function, update lam_l or lam_u
        if fp > 0.0:
            lam_l = max(lam_l, lam)
        if fp < 0.0:
            lam_u = min(lam_u, lam)

        # > compute an improved estimate for lam
        d1 = lam_l
        d2 = lam + lam_c
        lam = max(d1, d2)

    # endregion : Iteration

    return [lam, x, sdiag]
