"""
Deep Learning on Graphs - ALTEGRAD - Jan 2022
"""

import scipy.sparse as sp
import numpy as np
import torch
import torch.nn as nn

def normalize_adjacency(A):
    ############## Task 1
    
    ##################
    Atld = A + sp.eye(A.shape[0])
    Dtld_inv = 1/(Atld.sum(axis=1))
    Dtld_inv = sp.diags(Dtld_inv.T, [0], shape=Atld.shape)
    A_normalized = Dtld_inv @ Atld
    ##################

    return A_normalized


def sparse_to_torch_sparse(M):
    """Converts a sparse SciPy matrix to a sparse PyTorch tensor"""
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col)).astype(np.int64))
    values = torch.from_numpy(M.data)
    shape = torch.Size(M.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def loss_function(z, adj, device):
    mse_loss = nn.MSELoss()

    ############## Task 3
    
    ##################
    A_hat = torch.sigmoid(z @ z.T)
    selector = torch.zeros_like(A_hat, dtype=bool)
    selector[tuple(adj._indices())] = True
    
    A_hat_0 = A_hat[~selector]
    A_hat_1 = A_hat[selector]
    len_0 = min(len(A_hat_1), len(A_hat_0))
    A_hat_0 = A_hat_0[torch.multinomial(torch.ones(len(A_hat_0)), len_0)]
    
    y = torch.cat((torch.ones(len_0), torch.zeros(len_0)))
    y_pred = torch.cat((A_hat_1, A_hat_0))
    
    ##################
    loss = mse_loss(y_pred, y)
    return loss
