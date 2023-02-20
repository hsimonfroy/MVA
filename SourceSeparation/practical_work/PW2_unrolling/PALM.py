#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 16:56:48 2022

@author: ckervazo
"""

import numpy as np
import copy as cp
import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
#%%
def prox_l1(x,thrd=0):
    # Input : 
    # x : a torch tensor
    # thrd : a float
    zerolike = torch.zeros_like(x)
    return torch.maximum(zerolike,x-thrd) + torch.minimum(zerolike,x+thrd)

def prox_oblique(A):
    for ii in range(A.shape[1]):
        normeA = torch.sqrt(torch.sum(A[:,ii]**2))
        if normeA > 0 and normeA > 1.:
            A[:,ii] /= normeA
    return A


#%%
def PALM(set_loader, lamb = 0, itmax = 100,Ainit=1):    
    eps = 0.01
    S_est_list = []
    A_est_list = []
    itmax_list = []
    
    
    
    
    it=0

    for i, (X, A, S) in enumerate(set_loader): # For all mini-batches
        # X = Data set mini-batch
        # A = Ground truth A  mini-batch, used only for metric and sizes
        # S = Ground truth S  mini-batch, unused
        print("example %d",i)
        
        

        S_est_mb = torch.zeros([X.size()[0], S.size()[1],S.size()[2]], dtype=torch.double) # S_est_mb contains all the different samples of the estimated sources in a mini-batch
        A_est_mb = torch.zeros([X.size()[0], A.size()[1],A.size()[2]], dtype=torch.double) # A_est_mb contains all the different samples of A in a mini-batch
        it_max_mb = torch.zeros([X.size()[0]], dtype=torch.double)
    
        for j, (x, a, s) in enumerate(zip(X, A_est_mb, S_est_mb)):# For all samples in the current mini-batch

            S_est = s # A single source matrix
            A_est = Ainit # Initialization of A

            S_est_prev = S_est # For the stopping criterion
            A_est_prev = A_est # For the stopping criterion
            
            
            it = 0
            
            while(torch.norm(S_est-S_est_prev, p = 'fro') > 1e-6 or torch.norm(A_est-A_est_prev, p = 'fro') > 1e-6 or it < 2) and it < itmax: # PALM iterations
                if it>0:
                    S_est_prev = S_est
                    A_est_prev = A_est
                
                gamma = 1 / torch.linalg.norm(A_est.T @ A_est, ord=2)  # Lipschitz constant for S update
                S_est = prox_l1(S_est - gamma * A_est.T @ (A_est @ S_est - x) , lamb * gamma) # Proximal gradient step for S
                
                eta = 1 / torch.linalg.norm(S_est @ S_est.T, ord=2) # Lipschitz constant for A update
                A_est = prox_oblique( A_est + eta*(x - A_est @ S_est) @ S_est.T) # Proximal gradient step for A
                it += 1
            
            print('itmax %s'%it)
            S_est_mb[j] = S_est # Put the estimated S matrix inside of the mini-batch
            A_est_mb[j] = A_est # Put the estimated A matrix inside of the mini-batch
            it_max_mb[j] = it


        S_est_list.append(S_est_mb) # List containing all the mini-batches of estimated S matrices.
        A_est_list.append(A_est_mb) # List containing all the mini-batches of estimated A matrices.
        itmax_list.append(it_max_mb)
        
    return A_est_list,S_est_list,itmax_list
