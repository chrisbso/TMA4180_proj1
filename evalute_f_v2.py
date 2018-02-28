from matplotlib.patches import Ellipse
from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np

def main():
    A = generate_rnd_PD_mx(2)
    plot_ellipsoid(A,np.array([[0.5],[0.5]]))

def evalute_f(z,w,A,c):
    m = w.shape[1]
    n = z.shape[0]
    #z is n x m matrix
    #w is 1 x m matrix (labels of z_i-s)
    #A is n x n matrix
    #c is n x 1 matrix
    f = 0
    for i in range(m):
        z_i = z[:,i] #extract column i of matrix z, i.e. element z_i, i=0,...,m-1
        if w[i] == 1:
            r_i = max(np.transpose((z_i-c))*A*(z_i-c)-1,0);
        elif w[i] == -1:
            r_i = max(1-np.transpose((z_i - c)) * A * (z_i - c), 0)
        f = f + r_i**2;
    return f;

def generate_rnd_PD_mx(n):
    alpha = 0.005; #to guarantee our matrix is PD and not PSD.
    A = np.random.rand(n,n); # A is now random n x n matrix
    A = np.dot(A,A.transpose()) # A is now PSD
    A = A + alpha*np.identity(n); # A is now PD
    return A


def generate_rnd_points(A,m):
    n = A.shape[0]
    z = np.zeros([n,m])
    for i in range(m):
        z[:,i] = np.random.rand(1,n)

def plot_ellipsoid(A,c):
    assert(A.shape[0] == 2)
    assert(A.shape[0] == 2) #make sure we're in 2d

    (eigenvals,eigenvecs) = np.linalg.eig(A)
    wwidth = 2*np.sqrt(eigenvals[1])
    hheight = 2*np.sqrt(eigenvals[0])
    plotLimit = max(wwidth,hheight)/2
    print(eigenvals)
    print(eigenvecs)
    aangle = np.arctan2(eigenvecs[0,0],eigenvecs[0,1]);
    plt.figure()
    ax = plt.gca()
    ax.set_xlim(c[0]-plotLimit,c[0]+plotLimit)
    ax.set_ylim(c[1]-plotLimit,c[1]+plotLimit)

    ellipse = Ellipse(xy=(c[0], c[1]), width=wwidth, height=hheight, angle=aangle*180/(np.pi),
                            edgecolor='r', fc='None', lw=2)
    print(aangle)
    ax.add_artist(ellipse)
    plt.show()


if __name__ == "__main__":
    main()