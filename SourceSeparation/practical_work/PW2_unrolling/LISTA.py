import torch
import torch.nn as nn
import numpy as np
from fichiers_utils import prox_theta

class LISTA(nn.Module):
    def __init__(self, T=16, alpha=5, A=None,theta_shared=True ,We_shared = True ,G_shared = True):
        super(LISTA, self).__init__()
        self.T = T # Parameters : theta = alpha/L,# We and G : Z^{k+1}= We X + G Z^{k}
        self.alpha = alpha
        self.A = A
        self.theta_shared = theta_shared
        self.We_shared = We_shared
        self.G_shared = G_shared
        self.L = torch.tensor((1.001) * np.linalg.norm(self.A[0].numpy(),ord=2)**2)
        self.theta = (self.alpha / self.L).clone().detach()
        if not self.theta_shared:
            self.theta = self.theta.repeat(self.T,1)
        self.theta = nn.Parameter(self.theta,requires_grad= True)
        self.M = self.A.shape[1]
        self.N = self.A.shape[2]
        self.We = torch.div(torch.transpose(self.A[0],0,1), self.L)
        if not self.We_shared:
            self.We = self.We.repeat(self.T,1,1)
        self.We = nn.Parameter(self.We, requires_grad=True)
        # print('Type We')
        # print(self.We.type())
        self.G = torch.eye(self.A.shape[2]) - torch.matmul(torch.div(torch.transpose(self.A[0],0,1), self.L), self.A[0]) # Initialization I - 1/L A.T A
        if not self.G_shared:
            self.G = self.G.repeat(self.T,1,1)
        self.G = self.G.type(torch.FloatTensor)
        self.G = self.G.type('torch.DoubleTensor')
        self.G = nn.Parameter(self.G, requires_grad=True)
    
    def prox_l1(self, x,thrd=0):
        # Input : 
        # x : a torch tensor
        # thrd : a float
        zerolike = torch.zeros_like(x)
        return torch.maximum(zerolike,x-thrd) + torch.minimum(zerolike,x+thrd)
    
    def forward(self, X):
        b_size = X.shape[0]
        
        S = torch.zeros(b_size,self.N,X.shape[2]) # S initialization
        S = S.type('torch.DoubleTensor')
            
        for t in range(self.T):
            if self.We_shared:# Implement the W_e x X term of LISTA update, both when the parameters are shared and when they are not. Note that X is here a mini-batch (you can use the torch.bmm and repeat functions)
                p1 = self.We @ X
            else: # if the weights are not shared
                p1 = self.We[t] @ X
            if self.G_shared:# Do the same with the G x S term
                p2 = self.G @ S
            else:
                p2 = self.G[t] @ S
            p3 = p1+p2
            if self.theta_shared: # Apply the proximal operator of the L_1 norm
                S = self.prox_l1( p1 + p2, self.theta) # Shared parameters
            else:
                S = self.prox_l1( p1 + p2, self.theta[t]) # Non-shared parameters
                
        return S
