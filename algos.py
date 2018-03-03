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
     
    A_new,vec_new=phi(x+alpha*p, dim)
    
    while func(A_new,vec_new)>func(A, vec)+c_1*alpha*gradfunc(A,vec).dot(p):
        
        alpha= rho*alpha
        A_new,vec_new=phi(x+alpha*p, dim)
                
    
    return alpha



def steepest_descent(func, gradfunc,initial_data, initial_alpha,rho, c_1, tol, dim):
    k=0
    x= initial_data
    alpha=initial_alpha
    A,vec=phi(x,dim)
    p=(-1)*gradfunc(A,vec)
    
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
          (k, func(A,vec), np.linalg.norm(p,2), alpha))
    
    while  (np.linalg.norm(p,2)>tol/100): #and (alpha > tol*10**-3)
        
        #print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              #(k, func(A,c), np.linalg.norm(p,2), alpha))
              
        alpha= backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        x= x+ alpha*p
        A,vec=phi(x,dim)
        p=(-1)*gradfunc(A,vec)
        k += 1
        
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
          (k, func(A,vec), np.linalg.norm(p,2), alpha))
 
    return x


def gauss_newton(func, gradfunc,initial_data, initial_alpha,rho,
                 c_1, tol, dim, z, w):
    
    m = z.shape[1]
    k=0
    x= initial_data
    alpha=initial_alpha
    
    J=np.zeros((m,len(x)))
    r=np.zeros(m)
    A,vec=phi(x,dim)
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    
    while (np.linalg.norm(gradfunc(A,vec),2)>tol/100): #and alpha > tol/100000 
        A,vec=phi(x,dim)
        for i in range(m):
            z_i = z[:,i]
            w_i = w[i]
            J[i,:]=w_i*evaluate_grad_r_i_m1(z_i,A,vec)
            tail = w_i*2 * np.matmul(vec-z_i,A) #only for model 1
            
            J[i,int(dim*(dim+1)/2):] = tail
            r[i]=evaluate_r_i_m1(z_i,w_i,A,vec)
                
        matrix= np.matmul(np.linalg.inv(np.matmul(J.T,J)), J.T)
        p=- np.matmul(matrix,r)
        print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
        
        alpha=backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        x= x+ alpha*p
        k += 1
    
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,vec), np.linalg.norm(gradfunc(A,vec),2), alpha))
    return x


if __name__ == "__main__":
    

    dim=2
    A = generate_rnd_PD_mx(dim)
    c = np.array([1,1]) # need c from other programm!
    
    (z, w) = generate_rnd_points_m1(A, c, 200)
    
    # for model 1
    f= lambda A,c: evaluate_f_m1(z, w, A, c)
    gradf = lambda A,c: evaluate_grad_f_m1(z,w,A,c)
    
    # for model 2
    #g= lambda A,b: 
    #gradg = lambda A,b: 
    
    
    (wwidth, hheight, aangle) = ellipsoid_parameters_m1(A)
    limit = max(wwidth, hheight) / 2
    
    tolerance=limit
    print(tolerance)
    
    
    alpha=1
    rho=0.5
    c_1= 0.25
    # for inital values
    A_initial = generate_rnd_PD_mx(dim)
    c_initial = np.random.rand(dim)
    x_initial=phi_inv(A_initial, c_initial)
    
    #STEEPEST DESCENT
    steepest_descent(f, gradf,x_initial, alpha,rho, c_1, tolerance, dim)   
    
    #GAUSS NEWTON
    #gauss_newton(f, gradf,x_initial, alpha,rho,c_1, tolerance, dim, z, w)
    
    
    M=np.array([[1,2],[3,4]])
    (width, height, angle)=ellipsoid_parameters_m1(M)
    #print(width)
    #print(height)