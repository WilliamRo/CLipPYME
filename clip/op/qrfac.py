########################################################################
#
#   Created: June 19, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from enorm import euclid_norm as enorm
from dpmpar import get_machine_parameter as dpmpar
from utility import data_type

# region : Module parameters

p05 = data_type(0.05)
eps_machine = dpmpar(1)

ipvt = None
rdiag = None
acnorm = None
wa = None


# endregion : Module parameters

def qr(m, n, a, lda, pivot):
    """
    Uses householder transformations with column pivoting (optional)
    to compute a QR factorization of the m by n matrix a

    Parameters
    ----------
        m: int
            a positive integer input variable set to the number of rows
            of a
        n: int
            a positive integer input variable set to the number of
            columns of a
        a: ndarray
            an m by n array. on input a contains the matrix for which
            the qr factorization is to be computed. on output the
            strict upper trapezoidal part of a contains the strict
            upper trapezoidal part of r, and the lower trapezoidal
            part of a contains a factored form of q (the non-trivial
            elements of the u vectors described above)
        lda: int
            a positive integer input variable not less than m which
            specifies the leading dimension of the array a
        pivot: bool
            a logical input variable. if pivot is set true, then
            column pivoting is enforced. if pivot is set false, then
            no column pivoting is done

    Returns
    -------
        ipvt: ndarray
            an integer output array. ipvt defines the permutation
            matrix p such that a*p = q*r. column j of p is column
            ipvt(j) of the identity matrix. if pivot is false, ipvt
            will be set to None
        rdiag: ndarray
            an output array of length n which contains the diagonal
            elements of r
        acnorm: ndarray
            an output array of length n which contains the norms of
            the corresponding columns of the input matrix a. if this
            information is not needed, then acnorm can coincide with
            rdiag

    """

    # region : Initialize parameters
    # ----------------------------------------
    global p05, eps_machine, ipvt, rdiag, acnorm, wa

    if ipvt is None or ipvt.size is not n:
        ipvt = np.zeros(n, np.int32)
    if rdiag is None or rdiag.size is not n:
        rdiag = np.zeros(n, data_type)
    if acnorm is None or acnorm.size is not n:
        acnorm = np.zeros(n, data_type)
    if wa is None or wa.size is not n:
        wa = np.zeros(n, data_type)

    # ----------------------------------------
    # endregion : Initialize parameters

    # > compute the initial column norms and initialize several arrays
    for j in range(n):
        acnorm[j] = enorm(a[lda * j:lda * (j + 1)])
        rdiag[j] = acnorm[j]
        wa[j] = rdiag[j]
        if pivot:
            ipvt[j] = j + 1

    # > reduce a to r with householder transformations
    min_mn = min(m, n)
    for j in range(min_mn):
        # > if pivot
        # --------------------------------------------------------
        if pivot:
            # >> bring the column of largest norm
            #    into the pivot position
            k_max = j
            for k in range(j, n):
                if rdiag[k] > rdiag[k_max]:
                    k_max = k
            # >> switch
            if k_max is not j:
                for i in range(m):  # traverse rows
                    # >>> switch
                    temp = a[i + j * lda]
                    a[i + j * lda] = a[i + k_max * lda]
                    a[i + k_max * lda] = temp
                # >>> overwrite, acnorm[k_max] still hold
                rdiag[k_max] = rdiag[j]
                wa[k_max] = wa[j]
                # >>> switch
                k = ipvt[j]
                ipvt[j] = ipvt[k_max]
                ipvt[k_max] = k

        # > compute the householder transformation to reduce the
        #   j-th column of a to a multiple of the j-th unit vector
        # ------------------------
        # >> normalize
        # :: v = x - ||x||_2 * e_1
        # :: ajnorm = ||x||_2
        ajnorm = enorm(a[lda * j + j:lda * (j + 1)])
        if ajnorm != 0.0:
            if a[j + j * lda] < 0.0:
                # :: prepare to keep a[i + j * lda] positive
                ajnorm = -ajnorm
            # :: x = sgn(x_1) * x / ||x||_2
            for i in range(j, m):
                a[i + j * lda] /= ajnorm
            # :: a[j + j * lda] temporarily stores v[0]
            # :: one number being subtracted from another close number
            #    has been avoided
            a[j + j * lda] += 1.0

            # > apply the transformation to the remaining columns and
            #   update the norms
            #                                        t
            # :: A[i][k] -= beta * v[i] * w[k], w = A * v
            #            t
            # :: beta = 1 / v[0], can be proved easily
            # :: w[k] = A[k-th column] * v
            jp1 = j + 1  # j plus 1
            if n > jp1:
                for k in range(jp1, n):  # traverse columns
                    sum = data_type(0.0)  # this is w[j]
                    for i in range(j, m):  # traverse rows
                        #      v[i]             A[i][k-th column]
                        sum += a[i + j * lda] * a[i + k * lda]
                    # :: beta * w[k]
                    temp = sum / a[j + j * lda]
                    for i in range(j, m):
                        # :: a[i][k] -= beta * w[k] * v[i]
                        a[i + k * lda] -= temp * a[i + j * lda]

                    # :: rdiag stores information used to pivot
                    # >> update rdiag to ensure that it can present
                    #    alpha = +- ||x||_2
                    if pivot and rdiag[k] != 0:
                        temp = a[j + k * lda] / rdiag[k]
                        # >>> compute max
                        d1 = 1.0 - temp * temp
                        rdiag[k] *= np.sqrt(max(0.0, d1))
                        # >>> compute 2nd power
                        d1 = rdiag[k] / wa[k]
                        # :: if rdiag is to small
                        if p05 * (d1 * d1) <= eps_machine:
                            rdiag[k] = enorm(
                                a[jp1 + k * lda:(k + 1) * lda])
                            wa[k] = rdiag[k]
        # :: sgn(ajnorm) = -sgn(x_0)
        # :: H * x = alpha * e_1
        rdiag[j] = -ajnorm

    # > return
    if pivot:
        return [ipvt, rdiag, acnorm]
    else:
        return [rdiag, acnorm]
