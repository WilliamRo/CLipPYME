########################################################################
#
#   Created: June 19, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from numpy import (atleast_1d, dot, take, triu, shape, eye,
                   transpose, zeros, product, greater, array,
                   all, where, isscalar, asarray, inf, abs,
                   finfo, issubdtype, dtype)
from scipy.linalg import norm as enorm

import utility
import qrfac
from fdjac2 import jac
from qrfac import qr
from lmpar import lm_lambda

# region : Module parameters

p1 = 0.1
p5 = 0.5
p25 = 0.25
p75 = 0.75
p0001 = 1e-4

eps_machine = utility.eps_machine

wa4 = None
qtf = None


# endregion : Module parameters

def lmdif(func, x, args=(), full_output=0,
          col_deriv=0, ftol=1.49012e-8, xtol=1.49012e-8,
          gtol=0.0, maxfev=0, epsfcn=1e-8, factor=100, diag=None):
    """
    Minimize the sum of the squares of m nonlinear functions in n
        variables by a modification of the levenberg-marquardt
        algorithm. The user must provide a subroutine which calculates
        the functions. The jacobian is then calculated by a
        forward-difference approximation

    Parameters
    ----------
    func: callable
        should take at least one (possibly length N vector) argument and
        returns M floating point numbers. It must not return NaNs or
        fitting might fail.
    x: ndarray
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
        variables. If set None, the variables will be scaled internally

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
        was found.  O therwise, the solution was not found. In either
        case, the optional output variable 'mesg' gives more
        information.
    """

    # region : Initialize part of parameters

    global eps_machine, wa4, qtf
    global p1, p5, p25, p75, p0001
    ier = 0

    x = np.asarray(x).flatten()
    if not isinstance(args, tuple):
        args = (args,)

    if diag is None:
        mode = 1
    else:
        mode = 2

    # endregion : Initialize  part of parameters

    # region : Check the input parameters for errors

    if ftol < 0. or xtol < 0. or gtol < 0. or factor <= 0:
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
    fvec = func(x, *args)
    fnorm = enorm(fvec)

    # region : initialize other parameters
    # -----------------------------------------
    nfev = 1
    m = fvec.size
    n = x.size
    ldfjac = m
    if m < n:
        raise ValueError('!!! m < n in lmdif')
    if maxfev <= 0:
        maxfev = 200 * (n + 1)
    # > check wa4 and qtf
    if wa4 is None or wa4.size is not m:
        wa4 = np.zeros(m, utility.data_type)
    if qtf is None or qtf.size is not n:
        qtf = np.zeros(n, utility.data_type)
    # ------------------------------------------
    # endregion : initialize other parameters

    # endregion : Preparation before main loop

    # region : Main loop

    # > initialize levenberg-marquardt parameter and iteration counter
    lam = 0.0
    iter = 1

    # > begin outer loop
    while True:
        # > calculate the jacobian matrix
        fjac = jac(func, x, args, fvec, epsfcn)
        nfev += n

        # > compute the qr factorization of the jacobian
        ipvt, rdiag, acnorm = qr(m, n, fjac, ldfjac, True)

        # > on the first iteration
        if iter is 1:
            # >> if the diag is None, scale according to the norms of
            #    the columns of the initial jacobian
            if diag is None:
                diag = np.zeros(n, utility.data_type)
                for j in range(n):
                    diag[j] = qrfac.acnorm[j]
                    if diag[j] == 0.0:
                        diag[j] = 1.0
            # >> calculate the norm of the scaled x and initialize
            #    the step bound delta
            wa3 = qrfac.wa  # 'wa3' is a name left over by lmdif
            for j in range(n):
                wa3[j] = diag[j] * x[j]
            xnorm = enorm(wa3)
            delta = factor * xnorm
            if delta == 0.0:
                delta = factor

        # > form (q^T)*fvec and store the first n components in qtf
        # :: see x_{NG} = - PI * R^{-1} * Q_1^T * fvec
        # :: H * r = r - v * beta * r^T * v
        for i in range(m):
            wa4[i] = fvec[i]
        for j in range(n):  # altogether n times transformation
            # :: here the lower trapezoidal part of fjac contains
            #    a factored form of q, in other words, a set of v
            if fjac[j + j * ldfjac] != 0:
                sum = 0.0  # r^T * v
                for i in range(j, m):
                    sum += fjac[i + j * ldfjac] * wa4[i]
                # :: mul -beta
                temp = -sum / fjac[j + j * ldfjac]
                for i in range(j, m):
                    wa4[i] += fjac[i + j * ldfjac] * temp
            # restore the diag of R in fjac
            fjac[j + j * ldfjac] = qrfac.rdiag[j]
            qtf[j] = wa4[j]

        # > compute the norm(inf norm) of the scaled gradient
        #         t       t    t       t
        # :: g = J * r = R * Q1 * r = R * qtf
        gnorm = 0.0
        wa2 = qrfac.acnorm
        if fnorm != 0:
            for j in range(n):
                # >> get index
                l = ipvt[j] - 1
                if wa2[l] != 0.0:
                    sum = 0.0
                    for i in range(j):
                        sum += fjac[i + j * ldfjac] * (qtf[i] / fnorm)
                    # >>> computing max
                    d1 = np.abs(sum / wa2[l])
                    gnorm = max(gnorm, d1)

        # > test for convergence of the gradient norm
        if gnorm <= gtol:
            ier = 4
            break

        # > rescale if necessary
        if mode is not 2:
            for j in range(n):
                # >> compute max
                d1 = diag[j]
                d2 = wa2[j]
                diag[j] = max(d1, d2)

        # > beginning of the inner loop
        while True:
            # > determine the levenberg-marquardt parameter
            lam, wa1, sdiag = lm_lambda(n, fjac, ldfjac, ipvt,
                                        diag, qtf, delta, lam)

            # store the direction p and x + p. calculate the norm of p
            for j in range(n):
                wa1[j] = -wa1[j]
                wa2[j] = x[j] + wa1[j]
                wa3[j] = diag[j] * wa1[j]
            # :: pnorm = || D * p ||_2
            pnorm = enorm(wa3)

            # > on the first iteration, adjust the initial step bound
            if iter is 1:
                delta = min(delta, pnorm)

            # > evaluate the function at x + p and calculate its norm
            wa4 = func(wa2, *args)
            nfev += 1
            fnorm1 = enorm(wa4)

            # > compute the scaled actual reduction
            act_red = -1
            if p1 * fnorm1 < fnorm:
                # compute 2nd power
                d1 = fnorm1 / fnorm
                act_red = 1.0 - d1 * d1

            # > compute the scaled predicted reduction and the
            #   scaled directional derivative
            #
            # :: pre_red = (m(0) - m(p)) / m(0)
            # ::              t   t           t
            # ::         =  (p * J * J * p + J * r * p) / m(0)
            #
            # ::               t   t           t   t
            # :: J = Q * R => p * J * J * p = p * R * R * p
            # ::
            # :: m(0) = fnorm * fnorm
            for j in range(n):
                wa3[j] = 0
                l = ipvt[j] - 1
                temp = wa1[l]
                for i in range(j + 1):
                    wa3[i] += fjac[i + j * ldfjac] * temp
            # :: now wa3 stores J * p
            temp1 = enorm(wa3) / fnorm
            #                             t
            # :: lam * p = - grad_m(p) = J * r
            temp2 = (np.sqrt(lam) * pnorm) / fnorm
            # :: TODO -  ... / p5
            pre_red = temp1 * temp1 + temp2 * temp2 / p5
            dir_der = -(temp1 * temp1 + temp2 * temp2)

            # > compute the ratio of the actual to the predicted
            #   reduction
            ratio = 0.0
            if pre_red != 0:
                ratio = act_red / pre_red

            # > update the step bound
            if ratio <= p25:
                if act_red >= 0.0:
                    temp = p5
                else:
                    temp = p5 * dir_der / (dir_der + p5 * act_red)
                if p1 * fnorm1 >= fnorm or temp < p1:
                    temp = p1
                # >> compute min, shrink the trust region
                d1 = pnorm / p1
                delta = temp * min(delta, d1)
                lam /= temp
            else:
                if lam == 0.0 or ratio >= p75:
                    # >> expand the trust region
                    delta = pnorm / p5
                    lam = p5 * lam

            # > test for successful iteration
            if ratio >= p0001:
                # >> successful iteration. update x, fvec
                #    and their norms
                for j in range(n):
                    x[j] = wa2[j]
                    wa2[j] = diag[j] * x[j]
                for i in range(m):
                    fvec[i] = wa4[i]
                xnorm = enorm(wa2)
                fnorm = fnorm1
                iter += 1

            # > test for convergence
            if np.abs(act_red) <= ftol and pre_red <= ftol \
                    and p5 * ratio <= 1.0:
                ier = 1
            if delta <= xtol * xnorm:
                ier = 2
            if np.abs(act_red) <= ftol and pre_red <= ftol \
                    and p5 * ratio <= 1.0 and ier is 2:
                ier = 3
            if ier is not 0:
                break

            # > test for termination and stringent tolerances
            if nfev >= maxfev:
                ier = 5
            if np.abs(act_red) <= eps_machine and pre_red <= \
                    eps_machine and p5 * ratio <= 1.0:
                ier = 6
            if delta <= eps_machine * xnorm:
                ier = 7
            if ier is not 0:
                break

            if ratio >= p0001:
                break

        if ier is not 0:
            break

    # endregion : Main loop

    # > wrap results
    errors = {0: ["Improper input parameters.", TypeError],
              1: ["Both actual and predicted relative reductions "
                  "in the sum of squares\n  are at most %f" % ftol,
                  None],
              2: ["The relative error between two consecutive "
                  "iterates is at most %f" % xtol, None],
              3: ["Both actual and predicted relative reductions in "
                  "the sum of squares\n  are at most %f and the "
                  "relative error between two consecutive "
                  "iterates is at \n  most %f" % (ftol, xtol), None],
              4: ["The cosine of the angle between func(x) and any "
                  "column of the\n  Jacobian is at most %f in "
                  "absolute value" % gtol, None],
              5: ["Number of calls to function has reached "
                  "maxfev = %d." % maxfev, ValueError],
              6: ["ftol=%f is too small, no further reduction "
                  "in the sum of squares\n  is possible.""" % ftol,
                  ValueError],
              7: ["xtol=%f is too small, no further improvement in "
                  "the approximate\n  solution is possible." % xtol,
                  ValueError],
              8: ["gtol=%f is too small, func(x) is orthogonal to the "
                  "columns of\n  the Jacobian to machine "
                  "precision." % gtol, ValueError],
              'unknown': ["Unknown error.", TypeError]}

    if ier not in [1, 2, 3, 4] and not full_output:
        if ier in [5, 6, 7, 8]:
            print("!!! leastsq warning: %s" % errors[ier][0])

    mesg = errors[ier][0]

    if full_output:
        cov_x = None
        if ier in [1, 2, 3, 4]:
            from numpy.dual import inv
            from numpy.linalg import LinAlgError
            perm = take(eye(n), ipvt - 1, 0)
            r = triu(transpose(fjac.reshape(n, m))[:n, :])
            R = dot(r, perm)
            try:
                cov_x = inv(dot(transpose(R), R))
            except (LinAlgError, ValueError):
                pass
        dct = {'fjac': fjac, 'fvec': fvec, 'ipvt': ipvt,
               'nfev': nfev, 'qtf': qtf}
        return x, cov_x, dct, mesg, ier
    else:
        return x, ier
