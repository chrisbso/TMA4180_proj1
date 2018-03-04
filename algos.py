# -*- coding: utf-8 -*-

#import packages
import numpy as np
import matplotlib.pyplot as plt
from evaluate_f_gradf import *
from generate_testproblem import *



def phi(x, dim):
    ''' Takes the vector x with entries of A and b or A and c and gives back
    the corresponding matrix A and the vector b/c. Needs as second parameter
    the dimension.
    '''
    matrix = np.zeros([dim,dim], dtype=np.double)
    vector=np.zeros(dim)
    start=0
    k=dim-1
    j=dim-1
    
    # find the right entries
    for i in range(dim):
        
        if(i==dim-1):
            matrix[i,i]=x[start]
        else:
            # add 1 for considering also (i+dim-1-i)-th entry
            matrix[i,i:]=x[start: k+1]
            start=k+1
            k=k+j
            j-=1
    # use symmetry of A to construct it
    diagonal=matrix[np.diag_indices(dim)]
    matrix = matrix + matrix.T
    matrix[np.diag_indices(dim)]-=diagonal
    vector=x[int((dim*(dim+1))/2):]
    return matrix, vector


def phi_inv(matrix, vector):
    '''Takes a matrix and a vector and constructs the vector x with their
    entries under consideration of the symmetry of the matrix.
    '''
    n = len(vector)
    x_vector=np.zeros(int(0.5*(n*(n+1))+n))
    start=0
    k=n-1
    j=n-1
    # fill the vector x with entries
    for i in range(n):
        
        if(i==n-1):
            x_vector[start]=matrix[i,i]
        else:
            # add 1 for considering also (i+dim-1-i)-th entry
            x_vector[start: k+1]=matrix[i,i:]
            start=k+1
            k=k+j
            j-=1
    
    x_vector[int((n*(n+1))/2):]=vector
    return x_vector


def backtracking(func, gradfunc, x,A, vec, p, start, rho, c_1, dim):
    ''' Linesearch algorithm that computes the best fitting steplength alpha
    for the next iteration step.
    '''
    alpha=start  
    # compute matrix, vector out of given x
    A_new,vec_new = phi(x+alpha*p, dim)
    
    # sufficient decrease condition must be fulfilled
    while func(A_new,vec_new)>func(A, vec)+c_1*alpha*gradfunc(A,vec).dot(p):
        # decreasing alpha
        alpha= rho*alpha
        #update
        A_new,vec_new=phi(x+alpha*p, dim)
                
    return alpha



def steepest_descent(func, gradfunc,initial_data, 
                     initial_alpha,rho, c_1, tol, dim, plot_boolean):
    ''' steepest descent algorithm. Takes an argument 'plot_boolean' which
    states if the history of values needed for plotting shall be stored or not.
    Gives back x-vector that minimizes f and two parameters needed for plotting.
    '''
    k=0
    x= initial_data
    alpha=initial_alpha
    # create matrix, vector out of x
    A,vec=phi(x,dim)
    #steepest descent
    p=(-1)*gradfunc(A,vec)
    
    # initialiszing instances for storing plotting data
    conv=np.array(func(A,vec))
    matrix=np.zeros((1,len(x)))
    matrix=x.reshape((1, len(x)))
    
        
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
          (k, func(A,vec), np.linalg.norm(p,2), alpha))
    
    # termination criterion
    while  (np.linalg.norm(p,2)>tol/100):
        
        if(k==1):
            print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
                  (k, func(A,c), np.linalg.norm(p,2), alpha))
        # too many iterations  
        if(k==12000):
            print('break, may not be converging or converges very slow!')
            break
        
        #for plotting
        if((k==50 or k%100==0) and plot_boolean==True):
            
            conv = np.append(conv,func(A,vec))
            matrix=np.concatenate((matrix, x.reshape((1, len(x)))), axis=0)
           
        # calling backtracking method
        alpha= backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        # update x-vector
        x= x+ alpha*p
        A,vec=phi(x,dim)
        # update steepest descent
        p=(-1)*gradfunc(A,vec)
        k += 1
        
        
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
          (k, func(A,vec), np.linalg.norm(p,2), alpha))
 
    return x, conv,matrix


def gauss_newton_m1(func, gradfunc,initial_data, initial_alpha,rho,
                 c_1, tol, dim, z, w,plot_boolean):
    '''Gauss-Newton algorithm for model 1. Takes an argument 'plot_boolean' which
    states if the history of values needed for plotting shall be stored or not.
    Gives back x-vector that minimizes f and two parameters needed for plotting.
    '''
    m = z.shape[1]
    k=0
    x= initial_data
    alpha=initial_alpha
    
    J=np.zeros((m,len(x)))
    r=np.zeros(m)
    A,vec=phi(x,dim)
    
    # plotting 
    conv=np.array(func(A,vec))
    M=np.zeros((1,len(x)))
    M=x.reshape((1, len(x)))
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    
    # termination criterion
    while (np.linalg.norm(gradfunc(A,vec),2)>tol/100): 
        
        if(k==1):
            print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
                  (k, func(A,c), np.linalg.norm(gradfunc(A,vec),2), alpha))
        
        if(k==12000):
            print('break, may not be converging or converges very slow!')
            break
        # create J with grad r_i in it and r-vector
        for i in range(m):
            z_i = z[:,i]
            w_i = w[i]
            J[i,:]=w_i*evaluate_grad_r_i_m1(z_i,A,vec)
            r[i]=evaluate_r_i_m1(z_i,w_i,A,vec)
                
        matrix= np.matmul(np.linalg.inv(np.matmul(J.T,J)), J.T)
        #direction
        p=- np.matmul(matrix,r)
        
        # for plotting
        if((k==50 or k%100==0) and plot_boolean==True):
            conv = np.append(conv,func(A,vec))
            M=np.concatenate((M, x.reshape((1, len(x)))), axis=0)
        
        # call backtracking
        alpha=backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        x= x+ alpha*p
        A,vec=phi(x,dim)
        k += 1
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    return x,conv, M

def gauss_newton_m2(func, gradfunc,initial_data, initial_alpha,rho,
                 c_1, tol, dim, z, w,plot_boolean):
    '''Gauss-Newton algorithm for model 2. Uses other residuals.
    Takes an argument 'plot_boolean' which states if the history of values
    needed for plotting shall be stored or not.
    Gives back x-vector that minimizes f and two parameters needed for plotting.
    '''
    
    m = z.shape[1]
    k=0
    x= initial_data
    alpha=initial_alpha
        
    J=np.zeros((m,len(x)))
    r=np.zeros(m)
  
    A,vec=phi(x,dim)
    
    conv=np.array(func(A,vec))
    M=np.zeros((1,len(x)))
    M=x.reshape((1, len(x)))
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    # tolerance
    while (np.linalg.norm(gradfunc(A,vec),2)>tol/100): 
        
        
        if(k==1):
            print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
                  (k, func(A,c), np.linalg.norm(gradfunc(A,vec),2), alpha))
        if(k==12000):
            print('break, may not be converging or converges very slow!')
            break
        
        for i in range(m):
            z_i = z[:,i]
            w_i = w[i]
            J[i,:]=w_i*evaluate_grad_r_i_m2(z_i,A)
            r[i]=evaluate_r_i_m2(z_i,w_i,A,vec)
                
        matrix= np.matmul(np.linalg.inv(np.matmul(J.T,J)), J.T)
        p=- np.matmul(matrix,r)
        
        # for plotting
        if((k==50 or k%100==0) and plot_boolean==True):
            conv = np.append(conv,func(A,vec))
            M=np.concatenate((M, x.reshape((1, len(x)))), axis=0)
        
        
        alpha=backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        x= x+ alpha*p
        A,vec=phi(x,dim)
        k += 1
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    return x,conv, M


def plot_convergence(conv_1, conv_2, color_1, color_2, label1,label2):
    '''Creates convergence plots over iterations.
    '''
    n=len(conv_1)
    m=len(conv_2)
    grid = np.arange(0,n,1)*100
    grid2=np.arange(0,m,1)*100
               
        
    # plot using loglog scale
    plt.loglog(grid, conv_1,'-',label=label1, color=color_1)
    plt.loglog(grid2, conv_2,'-',label=label2, color=color_2)
    
    plt.xlabel("iterations")
    plt.ylabel("objective function")
    #plot legend
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.4), ncol=2)
    # show also the legend and title
    plt.tight_layout(pad=7)
    
    
def plot_ellipses1(array,z):
    ''' Plots the ellipses of model 1. Takes an array with the stored x-vectors
    during the iterations and the set of points z.
    '''
    plotLimit = max(np.max(z), np.abs(np.min(z)));
    plotLimit *= 1.2
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    ax = plt.gca()
    ax.set_xlim(-plotLimit,plotLimit)
    ax.set_ylim(-plotLimit,plotLimit)
    color=np.array(['y','b','r','c','g'])
    
    # plot all the ellipses stored in the array
    for i in range(len(array)):
        
        matrix, vector=phi(array[i,:],2)
        
        (wwidth, hheight, aangle) = ellipsoid_parameters_m1(matrix)
        ellipse = Ellipse(xy=(vector[0], vector[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                          edgecolor=color[i%5], fc='None', lw=1, alpha=0.5)
        ax.add_artist(ellipse)
    
 
def plot_ellipses2(array,z,w):
    ''' Plots the ellipses of model 1. Takes an array with the stored x-vectors
    during the iterations and the set of points z and the corresponding labels.
    '''
    maxPlotLimit = max(np.max(z),np.abs(np.min(z)));
    maxPlotLimit *= 1.2

    delta = 0.1
    z1 = z2 = np.arange(-maxPlotLimit, maxPlotLimit, delta)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.ones_like(Z1)
    
    
    for r in range(array.shape[0]):
        matrix, vector=phi(array[r,:],2)
        
        for i in range(len(z1)):
            for j in range(len(z1)):
                z_i = [z1[i], z2[j]]
                Z[j][i] = (np.dot(z_i, np.dot(matrix, z_i)) + np.dot(vector, z_i)) - 1
        
        #plot level sets
        plt.contour(Z1,Z2,Z,0, colors=('g'))
        
    plt.xlim(-maxPlotLimit, maxPlotLimit)
    plt.ylim(-maxPlotLimit, maxPlotLimit)
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    

    

if __name__ == "__main__":
    
    '''set boolean True if you want to see the plots. For saving computational
    time, use False.'''
    boolean= True
    
    '''number of dimensions. For plotting equal to 2'''
    dim=2
    
    # generates random initial data to construct set of points z
    A = generate_rnd_PD_mx(dim)
    c = np.random.rand(dim)
    b= np.random.rand(dim)    
    (z, w) = generate_rnd_points_m1(A, c, 200)
    
    
    # for model 1, functions
    f= lambda A,c: evaluate_f_m1(z, w, A, c)
    gradf = lambda A,c: evaluate_grad_f_m1(z,w,A,c)
    
    # for model 2, functions
    g= lambda A,b: evaluate_f_m2(z, w, A, b)
    gradg = lambda A,b: evaluate_grad_f_m2(z,w,A,b)
    
    # compute the tolerance for the termination criteria
    (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
    limit = max(wwidth, hheight) / 2
    
    tolerance=limit
    print("tolerance value for termination criterion= {}".format(tolerance/100))
    print('')
    
    # set initial steplength and other parameters for backtracking
    alpha=20
    rho=0.5
    c_1= 0.25
    # for inital values for the algorithms
    A_initial = generate_rnd_PD_mx(dim)
    c_initial = np.random.rand(dim)
    x_initial1=phi_inv(A_initial, c_initial)
    
    A_initial = generate_rnd_mx(dim)
    b_initial = generate_rnd_b_c(dim)
    x_initial2=phi_inv(A_initial, b_initial)
    
    #print results and figures
    
    print('Model 1')
    #STEEPEST DESCENT
    print('steepest descent')
    min1,conv1,v1=steepest_descent(f, gradf,x_initial1, alpha,rho, c_1, tolerance, dim, boolean) 
    print('')
    #GAUSS NEWTON
    print('Gauß-Newton')
    min2,conv2,v2=gauss_newton_m1(f, gradf,x_initial1, alpha,rho,c_1, tolerance, dim, z, w, boolean)
    
    
    print('')
    
    print('Model 2')
    #STEEPEST DESCENT
    print('steepest descent')
    min3,conv3,v3=steepest_descent(g, gradg,x_initial2, alpha,rho, c_1, tolerance, dim, boolean)   
    print('')
    #GAUSS NEWTON
    print('Gauß-Newton')
    min4,conv4,v4=gauss_newton_m2(g, gradg,x_initial2, alpha,rho,c_1, tolerance, dim, z, w, boolean)
    print('')
    
    
    if(boolean==True and dim==2):
        #first figure
        plt.figure(1)
        plot_convergence(conv1, conv2, 'g', 'c','steepest descent','Gauß-Newton')
        string="Model 1, alpha=20"
        plt.title(string)
        #plt.savefig('comp_m1_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.05) 
        plt.show()
           
        
        plt.figure(2)
        plot_convergence(conv3, conv4, 'g', 'c','steepest descent','Gauß-Newton')
        string="Model 2, alpha=20"
        plt.title(string)
        #plt.savefig('comp_m2_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005) 
        plt.show()
        
        plt.figure(3)
        plot_convergence(conv1, conv3, 'r', 'k','Model 1','Model 2')
        string="steepest descent, alpha=20"
        plt.title(string)
        #plt.savefig('comp_SD_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005)
        plt.show()
        
        plt.figure(4)
        plot_convergence(conv2, conv4, 'r', 'k','Model 1','Model 2')
        string="Gauß-Newton, alpha=20"
        plt.title(string)
        #plt.savefig('comp_GN_b.png', format='png', transporent=True, bbox_inches='tight', pad_inches=0.005) 
        plt.show()
        
        
        
        plt.figure(5)
        plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro', alpha=0.5, ms=5)
        plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo', alpha=0.5, ms=5)
        plot_ellipses1(v1,z)
        (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
        ellipse = Ellipse(xy=(c[0], c[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                              edgecolor='k', fc='None', lw=2, ls='--')
        ax = plt.gca()
        ax.add_artist(ellipse)
        plt.title('Model 1, steepest descent')
        #plt.savefig('Model1_SD_b.png')
        plt.show()
        
        plt.figure(6)
        plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro', alpha=0.5, ms=5)
        plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo', alpha=0.5, ms=5)
        plot_ellipses1(v2,z)
        (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
        ellipse = Ellipse(xy=(c[0], c[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                              edgecolor='k', fc='None', lw=2, ls='--')
        ax = plt.gca()
        ax.add_artist(ellipse)
        plt.title('Model 1, Gauß-Newton')
        #plt.savefig('Model1_GN_b.png')
        plt.show()
        
        
        plt.figure(7)
        plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro', alpha=0.5, ms=5)
        plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo',alpha=0.5, ms=5)
        plot_ellipses2(v3,z,w)
        plt.title('Model 2, steepest descent')
        #plt.savefig('Model2_SD_b.png')
        plt.show()
        
        plt.figure(8)
        plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro', alpha=0.5, ms=5)
        plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo', alpha=0.5, ms=5)
        plot_ellipses2(v4,z,w)
        plt.title('Model 2, Gauß-Newton')
        #plt.savefig('Model2_GN_b.png')
        plt.show()
        
        
        
       