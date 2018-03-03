import numpy as np
##MODEL 1
def evaluate_r_i_m1(z_i,w_i,A,c):
    if w_i == 1:
        r_i = max(np.dot((z_i - c), np.dot(A, (z_i - c)))-1, 0)
    elif w_i == -1:
        r_i = max(1 - np.dot((z_i - c), np.dot(A, (z_i - c))), 0)
    return r_i

def evaluate_f_m1(z, w, A, c):
    m = w.shape[0]
    n = z.shape[0]
    # z is n x m matrix
    # w is 1 x m matrix (labels of z_i-s)
    # A is n x n matrix
    # c is 1 x n matrix
    f = 0
    for i in range(m):
        z_i = z[:, i]  # extract column i of matrix z, i.e. element z_i, i=0,...,m-1
        r_i = evaluate_r_i_m1(z_i,w[i],A,c)
        f += r_i ** 2
    return f

def evaluate_grad_r_i_m1(z_i,A,c):
    n = A.shape[0]
    l = 0;
    grad_r = np.zeros(int(n * (n + 1) / 2 + n))
    for j in range(n):
        for k in range(j, n):
            if j == k:
                grad_r[l] = (z_i[j] - c[j]) ** 2
            else:
                grad_r[l] = 2 * (z_i[j] - c[j]) * (z_i[k] - c[k])
            l += 1
    return grad_r

def evaluate_grad_f_m1(z,w,A,c):
    n = A.shape[0]
    m = z.shape[1]
    grad_f = np.zeros(int(n*(n+1)/2+n))
    for i in range(m):
        z_i = z[:,i]
        w_i = w[i]
        if (isInRightSet_m1(z_i,w_i,A,c)):
            continue
        else:
            grad_r = w_i*evaluate_grad_r_i_m1(z_i,A,c)
            tail = 2 * np.matmul(c-z_i,A)
            grad_r[int(n*(n+1)/2):] = tail
            r = evaluate_r_i_m1(z_i, w_i, A, c)
            grad_f += r * grad_r
    grad_f *= 2
    return grad_f

def isInRightSet_m1(z_i,w_i,A,c):
    isIn = False
    if np.dot((z_i - c), np.dot(A, (z_i - c)))-1 < 0 and w_i == 1:
        isIn = True
    elif np.dot((z_i - c), np.dot(A, (z_i - c)))-1 > 0 and w_i == -1:
        isIn = True
    return isIn

##MODEL 2