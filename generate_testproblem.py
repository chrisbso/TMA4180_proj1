from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
import numpy as np
from evalute_f_v2 import *

def main():
    plot_test()


def generate_rnd_PD_mx(n):
    alpha = 0.0005  # to guarantee our matrix is PD and not PSD.
    A = np.random.rand(n, n)  # A is now random n x n matrix
    A = np.dot(A, A.transpose())  # A is now PSD
    A = A + alpha * np.identity(n)  # A is now PD
    return A


def generate_rnd_points(A, c, m):
    n = A.shape[0]
    z = np.zeros([n, m])
    w = np.ones(m)
    (width,height,angle) = ellipsoid_parameters(A)
    for i in range(m):
        z[:, i] = max(width,height)*np.random.rand(1, n)-max(width,height)/2
        z_i = z[:, i]
        if np.dot((z_i - c), np.dot(A, (z_i - c)))-1 > 0:
            w[i] = -1.0
    return z, w


def ellipsoid_parameters(A):
    (eigenvals, eigenvecs) = np.linalg.eig(A)
    width = 2 / np.sqrt(eigenvals[0])
    height = 2 / np.sqrt(eigenvals[1])
    angle = np.arctan2(eigenvecs[1, 0], eigenvecs[1, 1]);
    return (width, height, angle)


def plot_ellipsoid(A, c):
    assert (A.shape[0] == 2)
    assert (c.shape[0] == 2)  # make sure we're in 2d
    (wwidth, hheight, aangle) = ellipsoid_parameters(A)
    plotLimit = max(wwidth, hheight) / 2
    plt.figure()
    ax = plt.gca()
    ax.set_xlim(-plotLimit,plotLimit)
    ax.set_ylim(-plotLimit,plotLimit)

    ellipse = Ellipse(xy=(c[0], c[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                      edgecolor='r', fc='None', lw=2)
    ax.add_artist(ellipse)
    return plotLimit

def plot_test():
    A = generate_rnd_PD_mx(2)
    c = np.array([0,0])
    plot_ellipsoid(A, c)
    (z, w) = generate_rnd_points(A, c, 20)
    plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro')
    plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo')
    plt.show()

if __name__ == "__main__":
    main()