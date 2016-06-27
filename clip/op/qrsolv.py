########################################################################
#
#   Created: June 21, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from utility import data_type

# region : Module parameters

p5 = data_type(0.5)
p25 = data_type(0.25)

wa = None


# endregion : Module parameters

def qr_solve(n, r, ldr, ipvt, diag, qtb, x, sdiag):
    """
    Solves the linear least square problem:

              ||   /  a  \      / b \  || 2
          min ||  |      | x - |    |  ||
           x  ||  \  d  /      \ 0 /   || 2

     in which a is an m by n matrix, d is an n by n diagonal matrix,
     b is an m-vector. The necessary information must be provided:
      (1) the q-r factorization with column pivoting of a:
                a * p = q * r
           t
      (2) q * b
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
        n: int
            a positive integer input variable set to the order of r
        r: ndarray
            is an n by n array. on input the full upper triangle
            must contain the full upper triangle of the matrix r.
            on output the full upper triangle is unaltered, and the
            strict lower triangle contains the strict upper triangle
            (transposed) of the upper triangular matrix s
        ldr: ndarray
            a positive integer input variable not less than n
            which specifies the leading dimension of the array r
        ipvt: ndarray
            an integer input array of length n which defines the
            permutation matrix p such that a*p = q*r. column j of p
            is column ipvt(j) of the identity matrix
        diag: ndarray
            an input array of length n which must contain the
            diagonal elements of the matrix d
        qtb: ndarray
            an input array of length n which must contain the first
            n elements of the vector (q transpose)*b
        x: ndarray
            an output array of length n which contains the least
            squares solution of the system a*x = b, d*x = 0
        sdiag: ndarray
            an output array of length n which contains the
            diagonal elements of the upper triangular matrix s
            satisfies

                  t    t                    t
                 p * (a * a + d * d) * p = s * s

            In effect, s is the Cholesky factorization of
            the left matrix
    """

    # region : Initialize parameters

    global wa, p5, p25
    if wa is None or wa.size is not n:
        wa = np.zeros(n, data_type)

    # endregion : Initialize parameters

    # region : Preparation
    # ----------------------------
    # > copy r and qtb to preserve input and initialize s
    #   in particular, save the diagonal elements of r in x
    for j in range(n):
        for i in range(j, n):
            r[i + j * ldr] = r[j + i * ldr]
        x[j] = r[j + j * ldr]
        wa[j] = qtb[j]

    aa = 1
    # ----------------------------
    # endregion : Preparation

    # region : Givens rotation
    # ---------------------------
    # > eliminate the diagonal matrix d using a givens rotation
    # ::                            t              _    t
    # ::         n by n      / r * p \           / r * p \
    # ::   (m - n) by n     |    0    | = q_g * |    0    |
    # ::         n by n      \   d   /           \   0   /

    for j in range(n):
        # > prepare the row of d to be eliminated, locating the
        #   diagonal element using p from the qr factorization.
        l = ipvt[j] - 1
        if diag[l] != 0.0:
            # :: sdiag[l : n] stores the row j in r temporarily
            for k in range(j, n):
                sdiag[k] = 0
            sdiag[j] = diag[l]

            # :: the transformations to eliminate the row of d
            #    modify only a single element of qtb beyond the
            #    first n, which is initially zero.
            qtbpj = 0.0
            for k in range(j, n):
                # > determine a givens rotation which eliminates the
                #   appropriate element in the current row of d
                if sdiag[k] != 0.0:
                    if np.abs(r[k + k * ldr]) < np.abs(sdiag[k]):
                        cotan = r[k + k * ldr] / sdiag[k]
                        sin = p5 / np.sqrt(p25 + p25 * (cotan * cotan))
                        cos = sin * cotan
                    else:
                        tan = sdiag[k] / r[k + k * ldr]
                        cos = p5 / np.sqrt(p25 + p25 * (tan * tan))
                        sin = cos * tan
                    # > compute the modified diagonal element of r and
                    #   the modified element of (qtb, 0)^t
                    temp = cos * wa[k] + sin * qtbpj
                    qtbpj = -sin * wa[k] + cos * qtbpj
                    wa[k] = temp
                    # > transform the row of s
                    r[k + k * ldr] = cos * r[k + k * ldr] + \
                                     sin * sdiag[k]
                    if n > k + 1:
                        for i in range(k + 1, n):
                            temp = cos * r[i + k * ldr] + sin * sdiag[i]
                            sdiag[i] = -sin * r[i + k * ldr] + \
                                       cos * sdiag[i]
                            r[i + k * ldr] = temp

        # > store the diagonal element of s and restore the
        #   corresponding diagonal element of r
        sdiag[j] = r[j + j * ldr]
        r[j + j * ldr] = x[j]

    # > solve the triangular system for z. if the system is singular,
    #   then obtain a least squares solution
    #                      t
    # :: r * z = qtb, z = p * x and qtb is stored in wa
    nsing = n
    for j in range(n):
        if sdiag[j] == 0.0 and nsing is n:
            nsing = j
        if nsing < n:
            wa[j] = 0.0
    if nsing >= 1:
        # > use back substitution
        for k in range(1, nsing + 1):
            j = nsing - k
            sum = data_type(0)
            if nsing > j + 1:
                for i in range(j + 1, nsing):
                    sum += r[i + j * ldr] * wa[i]
            wa[j] = (wa[j] - sum) / sdiag[j]

    # > permute the components of z back to components of x

    # ---------------------------
    # endregion : Givens rotation
    for j in range(n):
        l = ipvt[j] - 1
        x[l] = wa[j]

    return
