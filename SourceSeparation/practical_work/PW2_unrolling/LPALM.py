import torch
import torch.nn as nn
import numpy as np
from fichiers_utils import prox_theta,tile,prox

class LPALM(nn.Module):
    def __init__(self, T=10,S=None,A=None,theta_shared= False,  W_CP_S_shared= False, learn_L_A = True , L_A_shared= False):
        super(LPALM, self).__init__()
		# Inputs : S=None,A=None must be correctly initialized to compute good initial Lipschitz constants.
        # learn_L_S : to learn L_S (non LISTA-CP)
        #
        # LPALM uses LISTA-like updates for S. theta, W_X_S and W_S_S are learnt such that S = soft(S + W_CP_S.T(X - AS),theta)
        # W_CP_S_shared: when using LISTA_CP_S = True, the W_CP are shared among the layers
        #
        # LPALM only learns L_A for A update
        # L_A_shared: when learn_L_A = True, the L_A are shared among the layers
                
        torch.set_default_tensor_type(torch.DoubleTensor)
        self.T = T
        self.S = S
        self.A = A
        #######################################################     S
        self.theta_shared = theta_shared
		
        self.W_CP_S_shared = W_CP_S_shared
		
		#######################################################     A
        self.L_A_shared = L_A_shared
    
		##########
        self.alpha =0.00001
        self.L_S = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
        self.L_A = torch.tensor((1.001) * np.linalg.norm(self.S[0].numpy(),ord=2)**2)
		
		#############################################################################

		
        self.theta = (self.alpha / self.L_S).clone().detach()
        self.We = torch.div(torch.transpose(self.A[0],0,1), self.L_S)
        self.W_CP_S = torch.transpose(self.We,0,1)
        
        if not self.theta_shared:
            self.theta = self.theta.repeat(self.T,1)
        self.theta = nn.Parameter(self.theta, requires_grad = True)
        
        if not self.W_CP_S_shared:
            self.W_CP_S = self.W_CP_S.repeat(self.T,1,1)
        self.W_CP_S = nn.Parameter(self.W_CP_S, requires_grad = True)
			
				
#	############################################################################
		
        if not self.L_A_shared:
            self.L_A = self.L_A.repeat(self.T,1)
        self.L_A = nn.Parameter(self.L_A,requires_grad = True)
    
    
    
    
    
    # From here, you have things to fill
    def prox_l1(self, x,thrd=0):
        # Input : 
        # x : a torch tensor
        # thrd : a float
        zerolike = torch.zeros_like(x)
        return torch.maximum(zerolike,x-thrd) + torch.minimum(zerolike,x+thrd)

    def prox_oblique(self, A):
        Aret = torch.clone(A)
        for ii in range(A.shape[1]):
            normeA = torch.sqrt(torch.sum(A[:,ii]**2))
            if normeA > 0 and normeA > 1.:
                Aret[:,ii] = A[:,ii] / normeA
        return Aret
    
    def forward(self, X):
	
        b_size = X.shape[0]
        #Initialize S estimates
        S_pred = torch.zeros(b_size,self.A.shape[2], X.shape[2])
        S_pred = S_pred.type(torch.DoubleTensor)
        #Initialize A estimates
        A_pred = torch.ones([b_size, X.shape[1], self.S.shape[1]])
        n_ = torch.norm(A_pred, dim=1)
        A_pred = A_pred/tile(n_,0,A_pred.shape[1]).reshape(A_pred.shape[0],A_pred.shape[1],A_pred.shape[2])
        A_pred = A_pred.type(torch.DoubleTensor)

		
        for t in range(self.T):
			
            # Iteration on S
            if self.W_CP_S_shared:# Implement the gradient step of LPALM update for S, both when the parameters are shared and when they are not. Note that X is here a mini-batch (you can use the torch.bmm and repeat functions)
                S_pred = S_pred - self.W_CP_S.T @ (A_pred @ S_pred - X)
            else:
                S_pred = S_pred - self.W_CP_S[t].T @ (A_pred @ S_pred - X)
                
            if self.theta_shared:# Implement the proximal step of LPALM update for S, both when the parameters are shared and when they are not.
                S_pred = self.prox_l1( S_pred, self.theta)
            else:
                S_pred = self.prox_l1( S_pred, self.theta[t])
					
		#############################################################################
			# Iteration on A
            # Do the same with A: first implement the gradient step
            if self.L_A_shared:
                A_pred = A_pred + (1/self.L_A)*(X - A_pred @ S_pred) @ S_pred.transpose(-1,-2)
            else:
                A_pred = A_pred + (1/self.L_A[t])*(X - A_pred @ S_pred) @ S_pred.transpose(-1,-2)
            
			# The apply the proximal operator of the oblique constraint
            A_pred = self.prox_oblique( A_pred )
						
		#############################################################################

        return S_pred, A_pred
        
