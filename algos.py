# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 19:41:24 2018

@author: Anne Sur
"""

#import packages
import numpy as np
import matplotlib.pyplot as plt
from evaluate_f_gradf import *



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
    
    # grad is morre the derivate here
    # dot: scalar product of the derivate(1 x n matrix!) and p (1 x n matrix!)
    
    A_new,vec_new=phi(x+alpha*p, dim)
    while(func(A_new,vec_new)>func(A, vec)+c_1*alpha*gradfunc(A,vec).dot(p)):
        print('Hi2')
        alpha= rho*alpha
    
    return alpha



def steepest_descent(func, gradfunc,initial_data, initial_alpha,rho, c_1, tol, dim):
    k=0
    x= initial_data
    alpha=initial_alpha
    #np.linalg.norm(gradfunc(x))
    while ( alpha > tol):
        print('Hi1')
        A,vec=phi(x,dim)
        p=-gradfunc(A,vec)
        print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
              (k, func(A,c), np.linalg.norm(p,2), alpha))
        alpha= backtracking(func, gradfunc, x, A,vec, p, alpha, rho, c_1,dim)
        
        x= x+ alpha*p
        print('Hi3')
        k += 1
        
    print("Iter: %3d, f=%15.6e, ||grad f||=%15.6e, steplength=%15.6e" % \
          (k, func(A,vec), np.linalg.norm(p,2), alpha))
    
    return x



if __name__ == "__main__":
    
    dim=2
    A = generate_rnd_PD_mx(dim)
    c = np.random.rand(dim) # need c from other programm!
    (z, w) = generate_rnd_points(A, c, 200)
    
    # for model 1
    f= lambda A,c: evalute_f(z, w, A, c)
    gradf = lambda A,c: np.array([1,2,3,4,5])
    
    # for model 2
    #g= lambda A,b: 
    #gradg = lambda x: 
    
    
    
    (wwidth, hheight, aangle) = ellipsoid_parameters(A)
    limit = max(wwidth, hheight) / 2
    tolerance=limit/100
    alpha=0.9
    rho=0.5
    c_1= 10**-4
    # for inital values
    A_initial = generate_rnd_PD_mx(dim)
    c_initial = np.random.rand(dim)
    x_initial=np.zeros(int(0.5*(dim*(dim+1))+dim)) # to be added
    x_initial=phi_inv(A_initial, c_initial)
    h=evalute_f(z, w, A, c) # later on (z,w,phi(x))
    #print(h) # to check...
    #print(f(A,c))
    minimum=steepest_descent(f, gradf,x_initial, alpha,rho, c_1, tolerance, dim)   
    

