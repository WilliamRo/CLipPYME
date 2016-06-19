########################################################################
#
#   Created: June 19, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from scipy.linalg import norm as enorm

import utility

# region : Module parameters

wa1 = None
wa2 = None
wa3 = None
wa4 = None

p1 = 0.1
p5 = 0.5
p25 = 0.25
p75 = 0.75
p0001 = 1e-4

eps_machine = utility.eps_machine

# endregion : Module parameters

def lmdif(func, x0, args=(), full_output=0,
          col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8,
          gtol=0.0, maxfev=0, epsfcn=None, factor=100, diag=None):
    """
    Minimize the sum of the squares of m nonlinear functions in n
        variables by a modification of the levenberg-marquardt
        algorithm. the user must provide a subroutine which calculates
        the functions. the jacobian is then calculated by a
        forward-difference approximation

    Parameters
    ----------
    func: callable
        should take at least one (possibly length N vector) argument and
        returns M floating point numbers. It must not return NaNs or
        fitting might fail.
    x0: ndarray
        The starting estimate for the minimization.
    args: tuple, optional
        Any extra arguments to func are placed in this tuple.
    full_output: bool, optional
        non-zero to return all optional outputs.
    col_deriv: bool, optional
        non-zero to specify that the Jacobian function computes
        derivatives down the columns (faster, because there is no
        transpose operation).
    ftol: float, optional
        Relative error desired in the sum of squares.
    xtol: float, optional
        Relative error desired in the approximate solution.
    gtol: float, optional
        Orthogonality desired between the function vector and the
        columns of the Jacobian.
    maxfev: int, optional
        The maximum number of calls to the function. If `Dfun` is
        provided then the default `maxfev` is 100*(N+1) where N is the
        number of elements in x0, otherwise the default `maxfev` is
        200*(N+1).
    epsfcn: float, optional
        A variable used in determining a suitable step length for the
        forward-difference approximation of the Jacobian (for
        Dfun=None). Normally the actual step length will be
        sqrt(epsfcn)*x If epsfcn is less than the machine precision,
        it is assumed that the relative errors are of the order of the
        machine precision.
    factor: float, optional
        A parameter determining the initial step bound
        (``factor * || diag * x||``). Should be in interval ``(0.1,
        100)``.
    diag: sequence, optional
        N positive entries that serve as a scale factors for the
        variables.

    Returns
    -------
    x: ndarray
        The solution (or the result of the last iteration for an
        unsuccessful call).
    cov_x: ndarray
        Uses the fjac and ipvt optional outputs to construct an
        estimate of the jacobian around the solution. None if a
        singular matrix encountered (indicates very flat curvature in
        some direction).  This matrix must be multiplied by the
        residual variance to get the covariance of the parameter
        estimates -- see curve_fit.
    infodict: dict
        a dictionary of optional outputs with the key s:

        ``nfev``
            The number of function calls
        ``fvec``
            The function evaluated at the output
        ``fjac``
            A permutation of the R matrix of a QR
            factorization of the final approximate
            Jacobian matrix, stored column wise.
            Together with ipvt, the covariance of the
            estimate can be approximated.
        ``ipvt``
            An integer array of length N which defines
            a permutation matrix, p, such that
            fjac*p = q*r, where r is upper triangular
            with diagonal elements of nonincreasing
            magnitude. Column j of p is column ipvt(j)
            of the identity matrix.
        ``qtf``
            The vector (transpose(q) * fvec).

    mesg: str
        A string message giving information about the cause of failure.
    ier: int
        An integer flag.  If it is equal to 1, 2, 3 or 4, the solution
        was found.  Otherwise, the solution was not found. In either
        case, the optional output variable 'mesg' gives more
        information.
    """

    # region : Initialize part of parameters

    global wa1, wa2, wa3, wa4
    global eps_machine
    global p1, p5, p25, p75, p0001

    # endregion : Initialize  part of parameters

    # region : Check the input parameters for errors

    if ftol < 0. or xtol < 0. or gtol < 0. or maxfev <= 0 \
            or factor <= 0:
        raise ValueError('!!! Some input parameters for lmdif ' +
                         'are illegal')

    if diag is not None:
        for d in diag:
            if d <= 0:
                raise ValueError('!!! Entries in diag must be positive')

    # endregion : Check the input parameters for errors

    # region : Preparation before main loop

    # > evaluate the function at the starting point and calculate
    # its norm
    fvec = func(x0, args)
    fnorm = enorm(fvec)

    # region : initialize other parameters
    # -----------------------------------------
    nfev = 1
    m = fvec.size
    n = x0.size
    if m < n:
        raise ValueError('!!! m < n in lmdif')
    # >> check work arrays
    if wa1 is None or wa1.size is not n:
        wa1 = np.zeros(n)
        wa2 = np.zeros(n)
        wa3 = np.zeros(n)
    if wa4 is None or wa4.size is not m:
        wa4 = np.zeros(m)
    # ------------------------------------------
    # endregion : initialize other parameters

    # endregion : Preparation before main loop

    # region : Main loop

    # > initialize levenberg-marquardt parameter and iteration counter
    par = 0.
    iter = 1

    # > begin outer loop
    while True:
        # > calculate the jacobian matrix
        pass

    # endregion : Main loop

    pass
