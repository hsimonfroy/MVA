import numpy as np
import torch
import torch.nn as nn
from copy import deepcopy
# from fichiers_utils import prox_oblique
#%%
def prox_oblique(A):
    for ii in range(A.shape[1]):
        normeA = torch.sqrt(torch.sum(A[:,ii]**2))
        if normeA > 0 and normeA > 1.:
            A[:,ii] /= normeA
    return A

#%%
def ISTA_A(set_loader,S_LISTA_set,itmax):
    eps = 0.01
    A_est_list = []
    itmax_list = []
    
    
    it=0

    for X, A, S in set_loader:# For all mini-batches
        # X = Data set mini-batch
        # A = Ground truth A  mini-batch, used only for metric and sizes
        # S = Ground truth S  mini-batch, unused
        
        S_LISTA_mb = S_LISTA_set # A mini-batch of the sources estimated through LISTA
        

        A_est_mb = torch.zeros([X.size()[0], A.size()[1],A.size()[2]], dtype=torch.double) # In initialization of a mini-batch of A
        it_max_mb = torch.zeros([X.size()[0]], dtype=torch.double)
            
        for j, (x, a, s) in enumerate(zip(X, A, S_LISTA_mb)) :# For all samples in the current mini-batch
            S_LISTA = s # A single source matrix, which was estimated by LISTA
            A_est = a # A single mixing matrix to be estimated
            A_est_prev = a # For the stopping criterion, keep in memory the previous mixing matrix.
            
            L = np.linalg.norm(S_LISTA @ S_LISTA.T, ord=2) # Lispschitz constant
            
            it = 0 
            
            while(torch.norm(A_est-A_est_prev, p = 'fro') > 1e-6 or it < 2) and it < itmax: # ISTA iterations
                if it>0:
                    A_est_prev = A_est
                
                A_est = prox_oblique( A_est + (1/L)*(x - A_est @ S_LISTA) @ S_LISTA.T) # Proximal gradient step
                
                it += 1
            
            A_est_mb[j] = A_est # Put the estimated mixing matrix inside of the mini-batch
            it_max_mb[j] = it
            
        A_est_list.append(A_est_mb) # List containing all the mini-batches of estimated mixing matrices.
        itmax_list.append(it_max_mb)
        
    return A_est_list, itmax_list
