from clip.op.qrfac import qr

import numpy as np

if __name__ == "__main__":
    # > generate A
    A = np.zeros(5 * 4, np.float32)
    for i in range(5):
        for j in range(4):
            A[i + j * 5] = i + j * 5 + 1

    ipvt, rdiag, acnorm = qr(5, 4, A, 5, True)
    pass
