########################################################################
#
#   Created: June 19, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from utility import data_type
from dpmpar import get_machine_parameter as dpmpar

# region : Module parameters

eps_machine = dpmpar(1)
fjac = None
wa = None


# endregion : Module parameters

def jac(func, x, args, fvec, epsfcn):
    """
    Computes a forward-difference approximation to the m by n jacobian
     matrix associated with a specified problem of m functions in n
     variables.

    Parameters
    ----------
    func: callable
        should take at least one (possibly length N vector) argument and
        returns M floating point numbers. It must not return NaNs or
        fitting might fail.
    x: ndarray
        an input array of length n
    args: tuple
        Any extra arguments to func are placed in this tuple.
    fvec: ndarray
        an input array of length m which must contain the functions
        evaluated at x
    epsfcn: float
        A variable used in determining a suitable step length for the
        forward-difference approximation of the Jacobian (for
        Dfun=None). Normally the actual step length will be
        sqrt(epsfcn)*x If epsfcn is less than the machine precision,
        it is assumed that the relative errors are of the order of the
        machine precision.

    Returns
    -------
    fjac: ndarray
        an output m by n array which contains the approximation to the
        jacobian matrix evaluated at x

    """

    # region : Initialize parameters
    # -------------------------------------
    global eps_machine, fjac, wa
    eps = np.sqrt(max(epsfcn, eps_machine))
    n = x.size
    m = fvec.size
    # > check fjac and wa
    if fjac is None or fjac.size is not m * n:
        fjac = np.zeros(m * n, data_type)
    if wa is None or wa.size is not m:
        wa = np.zeros(m, data_type)
    # -------------------------------------
    # endregion : Initialize parameters

    for i in range(n):
        temp = x[i]
        h = eps * abs(temp)
        if h == 0.0:
            h = eps
        x[i] += h
        wa = func(x, *args)
        # >> restore x[i]
        x[i] = temp
        for j in range(m):
            fjac[j + i * m] = (wa[j] - fvec[j]) / h

    return fjac
