########################################################################
#
#   Created: June 19, 2016
#   Author: William Ro
#
########################################################################

import numpy as np
from utility import data_type
from dpmpar import get_machine_parameter as dpmpar

rdwarf = dpmpar(2)
rgiant = dpmpar(3)


def euclid_norm(x):
    """
    Given an n-vector x, this function calculates the
    euclidean norm of x

    The euclidean norm is computed by accumulating the sum of
    squares in three different sums. The sums of squares for the
    small and large components are scaled so that no overflows
    occur. Non-destructive underflows are permitted. Underflows
    and overflows do not occur in the computation of the unscaled
    sum of squares for the intermediate components.
    The definitions of small, intermediate and large components
    depend on two constants, rdwarf and rgiant. The main
    restrictions on these constants are that rdwarf**2 not
    underflow and rgiant**2 not overflow. The constants
    given here are suitable for every known computer"""

    # > initialize parameters
    global rdwarf, rgiant
    n = x.size
    s1 = data_type(0.0)
    s2 = data_type(0.0)
    s3 = data_type(0.0)
    x1max = data_type(0.0)
    x3max = data_type(0.0)
    agiant = rgiant / n

    # > calculate sums
    for i in range(n):
        xabs = np.abs(x[i])
        if xabs >= agiant:
            # :: sum for large components
            if xabs > x1max:
                # > compute 2nd power
                d1 = x1max / xabs
                s1 = 1.0 + s1 * (d1 * d1)
                x1max = xabs
            else:
                # > compute 2nd power
                d1 = xabs / x1max
                s1 += d1 * d1
        elif xabs <= rdwarf:
            # :: sum for small components
            if xabs > x3max:
                # > compute 2nd power
                d1 = x3max / xabs
                s3 = 1.0 + s3 * (d1 * d1)
                x3max = xabs
            elif xabs != 0.0:
                # > compute 2nd power
                d1 = xabs / x3max
                s3 += d1 * d1
        else:
            # :: sum for intermediate components
            s2 += xabs * xabs

    # > calculate norm
    if s1 != 0:
        ret_val = x1max * np.sqrt(s1 + (s2 / x1max) / x1max)
    elif s2 != 0:
        if s2 >= x3max:
            ret_val = np.sqrt(
                s2 * (1.0 + (x3max / s2) * (x3max * s3)))
        else:
            ret_val = np.sqrt(
                x3max * ((s2 / x3max) + (x3max * s3)))
    else:
        ret_val = x3max * np.sqrt(s3)

    return ret_val
