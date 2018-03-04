# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:41:24 2018

@author: Anne Sur
"""

#import packages
import numpy as np
import matplotlib.pyplot as plt
from evaluate_f_gradf import *
from generate_testproblem import *



def phi(x, dim):
    matrix = np.zeros([dim,dim], dtype=np.double)
    vector=np.zeros(dim)
    start=0
    k=dim-1
    j=dim-1
    
    for i in range(dim):
        
        if(i==dim-1):
            matrix[i,i]=x[start]
        else:
            # add 1 for considering also (i+dim-1-i)-th entry
            matrix[i,i:]=x[start: k+1]
            start=k+1
            k=k+j
            j-=1
    diagonal=matrix[np.diag_indices(dim)]
    matrix = matrix + matrix.T
    matrix[np.diag_indices(dim)]-=diagonal
    vector=x[int((dim*(dim+1))/2):]
    return matrix, vector


def phi_inv(matrix, vector):
    n = len(vector)
    x_vector=np.zeros(int(0.5*(n*(n+1))+n))
    start=0
    k=n-1
    j=n-1
    
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
    
    alpha=start  
    
    #alpha=-(np.dot(gradfunc(A,vec),p))/(np.linalg.norm(p,2))
    #print(alpha)
         
    A_new,vec_new = phi(x+alpha*p, dim)
    
    while func(A_new,vec_new)>func(A, vec)+c_1*alpha*gradfunc(A,vec).dot(p):
        
        alpha= rho*alpha
        A_new,vec_new=phi(x+alpha*p, dim)
                
    
    return alpha



def steepest_descent(func, gradfunc,initial_data, initial_alpha,rho, c_1, tol, dim):
    k=0
    x= initial_data
    alpha=initial_alpha
    conv=np.zeros(1)
    matrix=np.zeros((1,len(x)))
    A,vec=phi(x,dim)
    p=(-1)*gradfunc(A,vec)
    
        
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
          (k, func(A,vec), np.linalg.norm(p,2), alpha))
    
    while  (np.linalg.norm(p,2)>tol/100): #and (alpha > tol*10**-3)
        
        if(k==1):
            print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
                  (k, func(A,c), np.linalg.norm(p,2), alpha))
        if(k%100==0):
            conv = np.append(conv,func(A,vec))
            
            matrix=np.concatenate((matrix, x.reshape((1, len(x)))), axis=0)
            
        
        alpha= backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        x= x+ alpha*p
        A,vec=phi(x,dim)
        p=(-1)*gradfunc(A,vec)
        k += 1
        
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
          (k, func(A,vec), np.linalg.norm(p,2), alpha))
 
    return x, conv,matrix


def gauss_newton_m1(func, gradfunc,initial_data, initial_alpha,rho,
                 c_1, tol, dim, z, w):
    
    m = z.shape[1]
    k=0
    x= initial_data
    alpha=initial_alpha
    M=np.zeros((1,len(x)))
    
    J=np.zeros((m,len(x)))
    r=np.zeros(m)
    conv=np.zeros(1)
    A,vec=phi(x,dim)
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    
    while (np.linalg.norm(gradfunc(A,vec),2)>tol/100): #and alpha > tol/100000 
        
        if(k==1):
            print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
                  (k, func(A,c), np.linalg.norm(p,2), alpha))
        
        for i in range(m):
            z_i = z[:,i]
            w_i = w[i]
            J[i,:]=w_i*evaluate_grad_r_i_m1(z_i,A,vec)
            r[i]=evaluate_r_i_m1(z_i,w_i,A,vec)
                
        matrix= np.matmul(np.linalg.inv(np.matmul(J.T,J)), J.T)
        p=- np.matmul(matrix,r)
        
        #print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              #(k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
        if(k%100==0):
            conv = np.append(conv,func(A,vec))
            M=np.concatenate((M, x.reshape((1, len(x)))), axis=0)

        
        alpha=backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        x= x+ alpha*p
        A,vec=phi(x,dim)
        k += 1
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    return x,conv, M

def gauss_newton_m2(func, gradfunc,initial_data, initial_alpha,rho,
                 c_1, tol, dim, z, w):
    
    m = z.shape[1]
    k=0
    x= initial_data
    alpha=initial_alpha
    M=np.zeros((1,len(x)))
    
    J=np.zeros((m,len(x)))
    r=np.zeros(m)
    conv=np.zeros(1)
    A,vec=phi(x,dim)
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    
    while (np.linalg.norm(gradfunc(A,vec),2)>tol/100): #and alpha > tol/100000 
        
        for i in range(m):
            z_i = z[:,i]
            w_i = w[i]
            J[i,:]=w_i*evaluate_grad_r_i_m2(z_i,A)
            r[i]=evaluate_r_i_m2(z_i,w_i,A,vec)
                
        matrix= np.matmul(np.linalg.inv(np.matmul(J.T,J)), J.T)
        p=- np.matmul(matrix,r)
        
        #print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              #(k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
        if(k%100==0):
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
    
    #n=max(len(conv_1),len(conv_2))
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
    plt.legend(loc=9, bbox_to_anchor=(0.5, -0.2), ncol=2)
    # show also the legend and title
    plt.tight_layout(pad=7)
    
def plot_levelsets(array,z):
    maxPlotLimit = max(np.max(z), np.abs(np.min(z)));
    maxPlotLimit *= 1.5

    delta = 0.1
    z1 = z2 = np.arange(-maxPlotLimit, maxPlotLimit, delta)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.ones_like(Z1)
    color = np.array(['y', 'b', 'r', 'c', 'g'])

    for r in range(array.shape[0]):
        A, c = phi(array[r, :], 2)

        for i in range(len(z1)):
            for j in range(len(z1)):
                z_i = [z1[i], z2[j]]
                Z[j][i] = np.dot((z_i - c), np.dot(A, (z_i - c)))-1

        plt.contour(Z1, Z2, Z, 0)

    plt.xlim(-maxPlotLimit, maxPlotLimit)
    plt.ylim(-maxPlotLimit, maxPlotLimit)
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    '''
    plotLimit = max(np.max(z), np.abs(np.min(z)));
    plotLimit *= 1.2
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    ax = plt.gca()
    ax.set_xlim(-plotLimit,plotLimit)
    ax.set_ylim(-plotLimit,plotLimit)
    color=np.array(['y','b','r','c','g'])
    
    for i in range(len(array)):
        matrix, vector=phi(array[i,:],2)
        
        (wwidth, hheight, aangle) = ellipsoid_parameters_m1(matrix)
        ellipse = Ellipse(xy=(vector[0], vector[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                          edgecolor=color[i%5], fc='None', lw=1, alpha=0.5)
        ax.add_artist(ellipse)
    '''
 
def plot_best2(array,z,w):
    
    maxPlotLimit = max(np.max(z),np.abs(np.min(z)));
    maxPlotLimit *= 1.5

    delta = 0.1
    z1 = z2 = np.arange(-maxPlotLimit, maxPlotLimit, delta)
    Z1, Z2 = np.meshgrid(z1, z2)
    Z = np.ones_like(Z1)
    color=np.array(['y','b','r','c','g'])
    
    for r in range(array.shape[0]):
        matrix, vector=phi(array[r,:],2)
        
        for i in range(len(z1)):
            for j in range(len(z1)):
                z_i = [z1[i], z2[j]]
                Z[j][i] = (np.dot(z_i, np.dot(matrix, z_i)) + np.dot(vector, z_i)) - 1
        
    
        plt.contour(Z1,Z2,Z,0)
        
    plt.xlim(-maxPlotLimit, maxPlotLimit)
    plt.ylim(-maxPlotLimit, maxPlotLimit)
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    


if __name__ == "__main__":
    
    
    
    dim=2
    A = generate_rnd_PD_mx(dim)
    c = np.random.rand(dim)# need c from other programm!
    b= np.random.rand(dim)
    
    
    (z, w) = generate_rnd_points_m1(A, c, 200)
    
    # for model 1
    f= lambda A,c: evaluate_f_m1(z, w, A, c)
    gradf = lambda A,c: evaluate_grad_f_m1(z,w,A,c)
    
    # for model 2
    g= lambda A,b: evaluate_f_m2(z, w, A, b)
    gradg = lambda A,b: evaluate_grad_f_m2(z,w,A,b)
    
    
    (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
    limit = max(wwidth, hheight) / 2
    
    tolerance=limit
    print(tolerance)
    
    
    alpha=20
    rho=0.5
    c_1= 0.25
    # for inital values
    A_initial = generate_rnd_PD_mx(dim)
    c_initial = np.random.rand(dim)
    x_initial1=phi_inv(A_initial, c_initial)
    
    A_initial = generate_rnd_mx(dim)
    b_initial = generate_rnd_b_c(dim)
    x_initial2=phi_inv(A_initial, b_initial)
    
    print('Model 1')
    #STEEPEST DESCENT
    print('steepest descent')
    min1,conv1,v1=steepest_descent(f, gradf,x_initial1, alpha,rho, c_1, tolerance, dim) 
    print('')
    #GAUSS NEWTON
    print('Gauß-Newton')
    min2,conv2,v2=gauss_newton_m1(f, gradf,x_initial2, alpha,rho,c_1, tolerance, dim, z, w)
    
    
    print('')
    
    print('Model 2')
    #STEEPEST DESCENT
    print('steepest descent')
    min3,conv3,v3=steepest_descent(g, gradg,x_initial1, alpha,rho, c_1, tolerance, dim)   
    print('')
    #GAUSS NEWTON
    print('Gauß-Newton')
    min4,conv4,v4=gauss_newton_m2(g, gradg,x_initial2, alpha,rho,c_1, tolerance, dim, z, w)
    print('')
    
    
    '''
    #first figure
    plt.figure(1)
    plot_convergence(conv1, conv2, 'g', 'c','steepest descent','Gauß-Newton')
    string="Model 1, alpha=10"
    plt.title(string)
    
       
    
    plt.figure(2)
    plot_convergence(conv3, conv4, 'g', 'c','steepest descent','Gauß-Newton')
    string="Model 2, alpha=10"
    plt.title(string)
    
    
    plt.figure(3)
    plot_convergence(conv1, conv3, 'r', 'k','Model 1','Model 2')
    string="steepest descent, alpha=10"
    plt.title(string)
    
    
    plt.figure(4)
    plot_convergence(conv2, conv4, 'r', 'k','Model 1','Model 2')
    string="Gauß-Newton, alpha=10"
    plt.title(string)
    '''
    
    plt.figure(5)
    plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro')
    plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo')
    plot_levelsets(v1,z)
    (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
    ellipse = Ellipse(xy=(c[0], c[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                          edgecolor='k', fc='None', lw=2, ls='--')
    ax = plt.gca()
    ax.add_artist(ellipse)
    plt.title('Model 1, steepest descent')
    plt.show()
    
    plt.figure(6)
    plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro')
    plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo')
    plot_levelsets(v2,z)
    (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
    ellipse = Ellipse(xy=(c[0], c[1]), width=wwidth, height=hheight, angle=aangle * 180 / (np.pi),
                          edgecolor='k', fc='None', lw=2, ls='--')
    ax = plt.gca()
    ax.add_artist(ellipse)
    plt.title('Model 1, Gauß-Newton')
    plt.show()
    
    
    plt.figure(7)
    plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro')
    plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo')
    #plot_levelsets(v3,z)
    plot_best2(v3,z,w)
    plt.title('Model 2, steepest descent')
    plt.show()
    
    plt.figure(8)
    plt.plot(np.take(z[0,:],np.where(w==1)[0]),np.take(z[1,:],np.where(w==1)[0]),'ro')
    plt.plot(np.take(z[0,:],np.where(w==-1)[0]),np.take(z[1,:],np.where(w==-1)[0]),'bo')
    #plot_levelsets(v4,z)
    plot_best2(v4,z,w)
    plt.title('Model 2, Gauß-Newton')
    plt.show()
    
    