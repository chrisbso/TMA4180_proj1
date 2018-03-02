import numpy as np

def evalute_f(z, w, A, c):
    m = w.shape[1]
    n = z.shape[0]
    # z is n x m matrix
    # w is 1 x m matrix (labels of z_i-s)
    # A is n x n matrix
    # c is 1 x n matrix
    f = 0
    for i in range(m):
        z_i = z[:, i]  # extract column i of matrix z, i.e. element z_i, i=0,...,m-1
        if w[i] == 1:
            r_i = max(np.dot((z_i - c), np.dot(A, (z_i - c)))-1, 0);
        elif w[i] == -1:
            r_i = max(1 - np.dot((z_i - c), np.dot(A, (z_i - c))), 0)
        f = f + r_i ** 2;
    return f;


