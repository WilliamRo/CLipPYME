import numpy as np
from numpy.linalg import norm as enorm
from clip.op.enorm import euclid_norm

n = 500
delta = np.zeros(n, np.float64)

for i in range(100):
    x = np.random.rand(n).astype(np.float64)
    delta[i] = abs(enorm(x) - euclid_norm(x))

print(">>> max = %f" % enorm(delta))
