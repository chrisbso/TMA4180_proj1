from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
from evaluate_f_gradf import *
import numpy as np

def main():
    #plot_test()
    evaluate_test()

def evaluate_test():
    A1 = generate_rnd_PD_mx(2)
    A2 = generate_rnd_mx(2)
    c = b = generate_rnd_b_c(2)

    (z, w) = generate_rnd_points_m1(A1, c, 20)
    (z2,w2) = generate_rnd_points_m2(A2,b,60)
    plot_ellipsoid_m1(A1, c, z, w)
    plot_ellipsoid_m2(A2, b, z2, w2)
    A1 = generate_rnd_PD_mx(2)
    A2 = generate_rnd_mx(2)
    c = b = generate_rnd_b_c(2)

    plot_ellipsoid_m1(A1, c, z, w)
    plot_ellipsoid_m2(A2,b,z2,w2)
    plt.show()

    '''
    p = np.random.randn(5)
    f0 = evaluate_f_m1(z, w, A, c)
    g = evaluate_grad_f_m1(z, w, A, c).dot(p)
    (A_p, c_p) = convert_x_To_A_and_c(p)
    # compare directional derivative with finite differences
    for ep in 10.0 ** np.arange(-1, -13, -1):
        g_app = (evaluate_f_m1(z, w, A + ep * A_p, c + ep * c_p) - f0) / ep
        print(g_app)
        error = abs(g_app - g) / abs(g)
        print('ep = %e, error = %e' % (ep, error))

    print(evaluate_grad_f_m1(z, w, A, c))
    '''
    p = np.random.randn(5)
    f0 = evaluate_f_m2(z2, w2, A2, b)
    g = evaluate_grad_f_m2(z2, w2, A2, b).dot(p)
    (A_p, b_p) = convert_x_To_A_and_c(p)
    # compare directional derivative with finite differences
    for ep in 10.0 ** np.arange(-1, -13, -1):
        g_app = (evaluate_f_m2(z2, w2, A2 + ep * A_p, b + ep * b_p) - f0) / ep
        print(g_app)
        error = abs(g_app - g) / abs(g)
        print('ep = %e, error = %e' % (ep, error))
    print(evaluate_grad_f_m2(z2, w2, A2, b))


def generate_rnd_PD_mx(n):
    alpha = 0.2  # to guarantee our matrix is PD and not PSD.
    A = np.random.rand(n, n)  # A is now random n x n matrix
    A = np.dot(A, A.transpose())  # A is now PSD
    A = A + alpha * np.identity(n)  # A is now PD
    return A

def generate_rnd_mx(n):
    A = 4*np.random.rand(n, n) - 2
    A = np.dot(A, A.transpose())
    return A
def generate_rnd_b_c(n):
    return (3*np.random.rand(n)-3/2)

def convert_x_To_A_and_c(x):

    dim = x.shape[0];
    n = int(-3/2+np.sqrt(9/4+2*dim))
    A = np.zeros([n,n])
    l=0;
    for i in range(n):
        for j in range(i,n):
            A[i][j] = x[l]
            A[j][i] = x[l]
            l += 1;
    c = x[int(dim-n):]
    return A,c



##Model 1
def generate_rnd_points_m1(A, c, m):
    n = A.shape[0]
    z = np.zeros([n, m])
    w = np.ones(m)
    (width,height,angle) = ellipsoid_parameters_m1(A)
    for i in range(m):
        z[:, i] = max(width,height)*np.random.rand(1, n)-max(width,height)/2
        for j in range(n):
            z[j,i] += c[j]
        z_i = z[:, i]
        if np.dot((z_i - c), np.dot(A, (z_i - c)))-1 >= 0:
            w[i] = -1.0
    return z, w


def ellipsoid_parameters_m1(A):
    (eigenvals, eigenvecs) = np.linalg.eig(A)
    width = 2 / np.sqrt(eigenvals[0])
    height = 2 / np.sqrt(eigenvals[1])
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[1, 1]);
    return (width, height, angle)


def plot_ellipsoid_m1(A, c, z,w):
    assert (A.shape[0] == 2)
    assert (c.shape[0] == 2)  # make sure we're in 2d
    plotLimit = max(np.max(z), np.abs(np.min(z)));
    plotLimit *= 1.5
    plt.figure()
    ax = plt.gca()
    ax.set_xlim(-plotLimit,plotLimit)
    ax.set_ylim(-plotLimit,plotLimit)
    plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro')
    plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo')

    (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
    ellipse = Ellipse(xy=(c[0], c[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                      edgecolor='k', fc='None', lw=2)
    ax.add_artist(ellipse)
    return plotLimit

''' THIS HAS BEEN MADE REDUNDANT
def plot_test_m1():
    A = generate_rnd_PD_mx(2)
    c = np.array([0,0])
    (z, w) = generate_rnd_points_m1(A, c, 20)
    plot_ellipsoid_m1(A, c,z,w);
    plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro')
    plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo')
    plt.show()
'''



##Model 2

def plot_ellipsoid_m2(A,b,z,w):
    assert(A.shape[0] == 2) #make sure we're in 2d
    assert(b.shape[0] == 2)
    assert(z.shape[0] == 2)

    maxPlotLimit = max(np.max(z),np.abs(np.min(z)));
    maxPlotLimit *= 1.5

    delta = 0.1
    z1 = z2 = np.arange(-maxPlotLimit, maxPlotLimit, delta)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.ones_like(Z1)
    for i in range(len(z1)):
        for j in range(len(z1)):
            z_i = [z1[i], z2[j]]
            Z[j][i] = (np.dot(z_i, np.dot(A, z_i)) + np.dot(b, z_i)) - 1
    plt.figure()
    plt.plot(np.take(z[0, :], np.where(w == 1)[0]), np.take(z[1, :], np.where(w == 1)[0]), 'ro')
    plt.plot(np.take(z[0, :], np.where(w == -1)[0]), np.take(z[1, :], np.where(w == -1)[0]), 'bo')

    plt.contour((Z1),(Z2),Z,0)
    plt.xlim(-maxPlotLimit, maxPlotLimit)
    plt.ylim(-maxPlotLimit, maxPlotLimit)
    plt.show()

def generate_rnd_points_m2(A, b, m):
    n = A.shape[0]
    z = np.zeros([n, m])
    w = np.ones(m)
    for i in range(m):
        z[:, i] = 10*np.random.rand(1, n)-5
        z_i = z[:, i]
        if ((np.dot(z_i, np.dot(A, z_i))+ np.dot(z_i,b))-1) >= 0:
            w[i] = -1.0
    return z, w

if __name__ == "__main__":
    main()
