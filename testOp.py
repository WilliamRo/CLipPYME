"""
f(x1, x2, a1, a2) = [a1 * x1^2 + a2 * x2^3,
                     a1 * x1^4 + a2 * x2^5,
                     a1 * x1^6 + a2 * x2^7]

"""

import numpy as np
from clip.op.fdjac2 import jac


def f(x, a1, a2):
    r = np.zeros(3, np.float64)
    r[0] = a1 * x[0] ** 2 + a2 * x[1] ** 3
    r[1] = a1 * x[0] ** 4 + a2 * x[1] ** 5
    r[2] = a1 * x[0] ** 6 + a2 * x[1] ** 7
    return r


def std_jac(x, a1, a2):
    jac = np.zeros(6, np.float64)
    for j in range(3):
        i = 2 * j + 2
        jac[j] = i * a1 * x[0] ** (i - 1)
        jac[j + 3] = (i + 1) * a2 * x[1] ** i
    return jac


if __name__ == "__main__":
    x = np.random.rand(2)
    a = np.random.rand(2)
    fvec = f(x, a[0], a[1])
    epsfcn = 0
    std_j = std_jac(x, a[0], a[1])
    res_j = jac(f, x, (a[0], a[1]), fvec, epsfcn)
    print(std_j - res_j)
