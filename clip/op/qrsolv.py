########################################################################
#
#   Created: June 21, 2016
#   Author: William Ro
#
########################################################################

import numpy as np

# region : Module parameters

p5 = 0.5
p25 = 0.25


# endregion : Module parameters

def qr_solve(n, r, ldr, ipvt, diag, qtb, x, sdiag, wa):
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

        wa: ndarray
            a work array of length n
    """

    pass
