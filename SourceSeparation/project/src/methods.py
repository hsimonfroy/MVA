import numpy as np
import Starlet2D as st2
import scipy.io as sci

from utils import prox_l1, prox_oblique, prox_positive, mad, grad_poisson_likelihood
import Starlet2D as st2

######## PALM-NMF algorithm ########
def PALM(X, n=3, itmax=1000, Ainit=None, Sinit=None): 
    if Ainit is None:
        A_est = np.random.rand(np.shape(X)[0],n)
    else:
        A_est = Ainit.copy()

    if Sinit is None:
        S_est = np.random.rand(n,np.shape(X)[1])
    else:
        S_est = Sinit.copy()
    S_est_prev = S_est.copy()
    
    it = 0
    criterions = [0]
    while(criterions[-1] > 1e-6 or it < 2) and it < itmax: # ISTA iterations
        if it > 0:
            S_est_prev = S_est
            
        # S update
        gamma = 1 / np.linalg.norm(A_est.T @ A_est, ord=2)
        S_est = S_est - gamma * A_est.T @ (A_est @ S_est - X)
        S_est = prox_positive(S_est)

        # A update
        eta = 1 / np.linalg.norm(S_est @ S_est.T, ord=2)
        A_est =  A_est - eta*(A_est @ S_est - X) @ S_est.T
        A_est = prox_positive(A_est)
        
        it += 1
        criterions.append(np.linalg.norm(S_est - S_est_prev))

    return A_est, S_est, criterions[1:]


######## ISTA algorithm with a stationary mixing matrix A ########
def ISTA_S(X, A, lamb=0, itmax=10000, Sinit=None):
    if Sinit is None:
        S_est = np.random.rand(A.shape[1], X.shape[1]) # A single estimated mixing matrix.
    else:
        S_est = Sinit.copy()
    S_est_prev = S_est.copy() # For the stopping criterion, keep in memory the previous source matrix.
            
    L = np.linalg.norm(A.T @ A, ord=2) 
    it = 0   
    criterions = [0]
    
    while(criterions[-1] > 1e-6 or it < 2) and it < itmax: # ISTA iterations
        if it > 0:
            S_est_prev = S_est
                                
        S_est = prox_positive(S_est - (1/L) * A.T @ (A @ S_est - X)) # Proximal gradient step
        it += 1
        criterions.append(np.linalg.norm(S_est - S_est_prev))
            
    return S_est, criterions[1:]


######## ISTA algorithm with a non-stationary mixing matrix A ########
def NSISTA_S(X, NSA, lamb=0, itmax=1000): 
    XT = X.T[:,:,None]
    S_est = np.random.rand(X.shape[1], NSA.shape[2], 1) # A single estimated mixing matrix.
    S_est_prev = S_est.copy() # For the stopping criterion, keep in memory the previous source matrix.
            
    Ls = np.linalg.norm(NSA.transpose(0,2,1) @ NSA, ord=2, axis=(1,2))[:,None,None]
    it = 0   
    criterions = [0]
    
    while(criterions[-1] > 1e-6 or it < 2) and it < itmax: # ISTA iterations
        if it > 0:
            S_est_prev = S_est
                                
        S_est = prox_positive(S_est - (1/Ls) * NSA.transpose(0,2,1) @ (NSA @ S_est - XT)) # Proximal gradient step
        it += 1
        criterions.append(np.linalg.norm(S_est - S_est_prev))
        
        if it%10==0:
            print(it)
            
    return S_est.squeeze().T, criterions[1:]


######## SPA algorithm to initialize the Minimum Volume NMF algorithm ########
def simpleSPA(Xin, r, optDisp = False):
    
    X = Xin.copy()
    K = np.zeros(r)
    R = np.zeros(r)
    m = X.shape[0]
    for ii in range(r):

        r = np.zeros(X.shape[1])
        for j in range(X.shape[1]):
            r[j] = np.linalg.norm(X[:, j])
        p = np.argmax(r)
        R[ii] = np.max(r)

        K[ii] = p
        X = (np.eye(X.shape[0]) - (X[:, p].reshape(m, 1) @ X[:, p].reshape(1, m))/np.linalg.norm(X[:, p])**2) @ X
        
        print('Max residual %s'%np.max(X))

    K = K.astype(int)
    if optDisp:

        plt.figure()
        plt.plot(A)
        plt.title('True A')
        plt.show()
    
    return K



######## Minimum Volume NMF algorithm ########
def simplexProx(X,epsilon = 0):
#     Given X,  computes its projection Y onto the simplex  

#       Delta = { x | x >= 0 and sum(x) <= 1 }, 

# that is, Y = argmin_z ||x-X||_2  such that z in S. 

# If X is a matrix, is projects its columns onto Delta to generate Y.


# ****** Input ****** 
# X       : m-by-r matrix
# epsilon : r-by-1 vector, generally positive and taken = 0

# ****** Output ****** 
# Y       : the projected matrix

# Code from the paper 
# P. De Handschutter, N. Gillis, A. Vandaele and X. Siebert, 
# "Near-Convex Archetypal Analysis", IEEE Signal Processing Letters 27 (1),
# pp. 81-85, 2020. 

    
    if np.isscalar(epsilon) == 1:
        epsilon = epsilon*np.ones(np.shape(X)[1]) 
    
    Y = np.zeros(np.shape(X))
    
    for ii in range(np.shape(X)[1]): # The prox is separable over the columns of X
        x = X[:,ii].copy() # We work on each column separately
        xsort = np.sort(x) # We need to sort the x value to apply the dichotomy
    
        index_min=0 # Index values for the dichotomy
        index_max=len(x)-1;
        
        # mu s.t. x_i > mu-epsilon, forall i
        mu_min=xsort[0]+epsilon[ii] # mu is the parameter required for the projection
    
        min_sum=np.sum(x)-len(x)*mu_min
        
        if min_sum < 1:# If the vector already satisfies the sum to at most one constraint
            mu=(np.sum(x)-1.)/float(len(x))
            y=np.maximum(-epsilon[ii]*np.ones(len(x)), x-mu) # Element-wise max
            Y[:,ii]=y;
        else:
            # Use dichotomy for finding the optimal mu value
            stop = False
            it = 0
            while stop == False:
                it += 1
                
                cur_ind = int(np.round((float(index_min)+float(index_max)+1e-6)/2.))
                mu=xsort[cur_ind]+epsilon[ii]
                y=np.maximum(-epsilon[ii]*np.ones(len(x)), x-mu)
                
                val_constr = np.sum(y)
                if val_constr < 1.:
                    index_max=cur_ind # Because the objective is decreasing with mu and indMax > indMin
                elif val_constr > 1.:
                    index_min=cur_ind
                    
                else: # We found the best mu
                    Y[:,ii]=y
                    stop = True
                    
                    
                if index_max == index_min + 1:# This is a stopping condition, as the constraint function is piecewise linear
                    stop = True
                    
                    
            mu_inf=xsort[index_min]+epsilon[ii];
            mu_sup=xsort[index_max]+epsilon[ii];
            constr_inf = np.sum(np.maximum(-epsilon[ii]*np.ones(len(x)),x-mu_inf))
            constr_sup = np.sum(np.maximum(-epsilon[ii]*np.ones(len(x)),x-mu_sup))
                
            slope=(constr_sup-constr_inf)/(mu_sup-mu_inf)
            mu_opt=(1.-constr_inf)/slope+mu_inf # Because the constraint function is piecewise linear
            
            # Compute the corresponding column of Y
            y=np.maximum(-epsilon[ii]*len(x), x-mu_opt)
            
            Y[:,ii]=y
             
    return Y




#%%
def FGM_MM_nonneg(A,C,W0=0,maxiter=500,proj=1):
    # Fast gradient method to solve nonnegative least squares.  
    # See Nesterov, Introductory Lectures on Convex Optimization: A Basic 
    # Course, Kluwer Academic Publisher, 2004. 
    
    # This code solves: 
    
    #     min_{x_i in R^r_+} sum_{i=1}^m ( x_i^T A x_i - 2 c_i^T x_i ), if proj == 1
    #     min_{x_i in S} sum_{i=1}^m ( x_i^T A x_i - 2 c_i^T x_i ), if proj == 2 (with S = simplex)
    # [W,e] = FGMfcnls(A,C,W0,maxiter) 
    
    # ****** Input ******
    # A      : Hessian for each row of W, positive definite
    # C      : linear term <C,W>
    # W0     : m-by-r initial matrix
    # maxiter: maximum numbre of iterations (default = 500). 
    # proj   : =1, nonnegative orthant
    #          =2, nonnegative orthant + sum-to-one constraints on columns
    #
    # ****** Output ******
    # W      : approximate solution of the problem stated above. 
    # e      : e(i) = error at the ith iteration

    if np.isscalar(W0):
        W0 = np.zeros(np.shape(C));

    L = np.linalg.norm(A,2)# Pas evident...
    e = np.zeros(maxiter)
    # Extrapolation parameter
    beta = (1.-np.sqrt(np.linalg.cond(A))) / (1. + np.sqrt(np.linalg.cond(A))); 
    
    # Project initialization onto the feasible set
    if proj == 1:
        W = np.maximum(W0,0)
    elif proj == 2:
        W = simplexProx(W0)
        
    
    Y = W # Initialization of the second sequence (for the acceleration)
    ii = 0;
    eps0 = 0.
    eps = 1.
    delta = 1e-6
    
    while ii < maxiter and eps >= delta*eps0:
        # print("FGM_MM_nonneg, it %s"%ii)
        # Previous iterate
        Wp = W
        
        # FGM Coefficients  
        # alpha(i+1) = ( sqrt(alpha(i)^4 + 4*alpha(i)^2 ) - alpha(i)^2) / (2); 
        # beta(i) = alpha(i)*(1-alpha(i))/(alpha(i)^2+alpha(i+1)); 
        
        # Projected gradient step from Y
        W = Y - (Y@A-C) / L 
        
        # Projection
        if proj == 1:
            W = np.maximum(W,0.)
        elif proj == 2:
            W = simplexProx( W )
            
        
        # Linear combination of iterates
        Y = W + beta*(W-Wp)
        
        # Error
        e[ii] = np.sum((W.T@W)*A) - 2.*np.sum(W*C)
        
        
        # Restart: fast gradient methods do not guarantee the objective
        # function to decrease, a good heursitic seems to restart whenever it
        # increases although the global convergence rate is lost! This could
        # be commented out. 
        
        if ii >= 2 and e[ii] > e[ii-1]:
            Y = W
            
        if ii == 1:
            eps0 = np.sqrt(np.sum((W-Wp)**2))
        
        eps = np.sqrt(np.sum((W-Wp)**2))
        
        ii += 1
    return W,e

#%%
def nnls_FPGM(X,W,delta=1e-6,inneriter=500,proj=0,alpha0=0.05,H = 0,options=0 ):

     # Computes an approximate solution of the following nonnegative least
     # squares problem (NNLS)
    
     #           min_{H >= 0} ||X-WH||_F^2
     
     # using a fast gradient method; 
     # See Nesterov, Introductory Lectures on Convex Optimization: A Basic 
     # Course, Kluwer Academic Publisher, 2004. 
     
     # Input / Output; see nnls_input_output.m  
     
     # + options.proj allows to use a contraints on the columns or rows of H so 
     #   that the entries in each column/row sum to at most one 
     #   options.proj = 0: no projection (default). 
     #   options.proj = 1: projection of the columns on {x|x>=0, sum(x) <= 1} 
     #   options.proj = 2: projection of the rows {x|x>=0, sum(x) = 1} 
          
     # + options.alpha0 is the FPGM  extrapolation parameter (default=0.05). If options.alpha0 = 0 --> no acceleration, PGM
    
     # Code modified from https://sites.google.com/site/nicolasgillis/code
    

   
    
    # If no initial matrices are provided, H is initialized as follows: 
    if np.isscalar(H):
        H = np.zeros((np.shape(W)[1],np.shape(X)[1]))

    
    # Hessian and Lipschitz constant 
    WtW = W.T@W
    L = np.linalg.norm(WtW,2)
    # Linear term 
    WtX = W.T@X

    alpha = np.zeros(inneriter + 1)
    beta = np.zeros(inneriter)
    alpha[0] = alpha0
    
    if options == 0: # Project onto the non-negative orthant
        H = np.maximum(H,0)
    elif options == 1: # Project columns of H onto the simplex and origin
        H = np.maximum(H,0) 
        K = np.where(np.sum(H,axis=0) > 1.)[0] 
        H[:,K] = simplexProx( H[:,K] ) 
    elif options == 2: # Project rows of H onto the simplex
        H = simplexProx(H.T)
        H = H.T 
    
    
    
    Y = H # Second sequence
    ii = 0
    # Stop if ||V^{k}-V^{k+1}||_F <= delta * ||V^{0}-V^{1}||_F
    eps0 = 0
    eps = 1  
    while ii < inneriter and eps >= delta*eps0:
        # Previous iterate
        Hp = H; 
        # FGM Coefficients; see Nesterov's book
        alpha[ii+1] = ( np.sqrt(alpha[ii]**4 + 4*alpha[ii]**2 ) - alpha[ii]**2) / 2.
        beta[ii] = alpha[ii]*(1.-alpha[ii])/(alpha[ii]**2+alpha[ii+1])
        # Projection step
        H = Y - (WtW@Y-WtX) / L
        
        if options == 0:
            H = np.maximum(H,0);
        elif options == 1:
            H = np.maximum(H,0) # Project columns of H onto the set {x|x>=0, sum(x) <= 1} 
            K = np.where(np.sum(H,axis=0) > 1.)[0]
            H[:,K] = simplexProx( H[:,K] )  
        elif options == 2:
            H = simplexProx(H.T) # Project rows of H onto the simplex
            H = H.T
        
        # Linear combination of iterates
        Y = H + beta[ii]*(H-Hp)
        if ii == 1:
            eps0 = np.linalg.norm(H-Hp,'fro')
        
        eps = np.linalg.norm(H-Hp,'fro')
        ii = ii + 1; 

    return H,WtW,WtX



def normalizeWH(W,H,sumtoone,X): # A TESTER
    # Normalization depending on the NMF model 
    if sumtoone == 1: # Normalize so that H^Te <= e entries in cols of H sum to at most 1
                      
        Hn = simplexProx( H );
        if np.linalg.norm(Hn - H) > 1e-3*np.linalg.norm(Hn):
           H = Hn
           # Reoptimize W, because this normalization is NOT w.l.o.g. 
           W,WtW_temp,WtX_temp = nnls_FPGM(X.T,H.T,H = W.T,inneriter=100)
           W = W.T
        
        H = Hn 
        
    elif sumtoone == 2: # Normalize so that He = e, entries in rows of H sum to 1
        scalH = np.sum(H,axis=1)
        H = np.diag( scalH**(-1) )@H
        W = W@np.diag( scalH )
        
    elif sumtoone == 3: # Normalize so that W^T e = e, entries in cols of W sum to 1
        scalW = np.sum(W,axis=0)
        H = np.diag( scalW )@H
        W = W@np.diag( scalW**(-1) )


    return W,H


#%%
def minvolNMF(X,r,W,H,lamb=0.1,delta=0.1,model=3,maxiter=100,inneriter=10,target = None):
    # W,H : initializations (W par SNPA, H par NNLS)

    # Normalization
    W,H = normalizeWH(W,H,model,X) # OM pour modele 2

    # Initializations
    normX2 = np.sum(X**2)
    normX = np.sqrt(normX2)
    
    WtW = W.T@W;
    WtX = W.T@X;
    
    err1 = np.zeros(maxiter)
    err2 = np.zeros(maxiter)
    e = np.zeros(maxiter)
    
    # Initial error and set of of lambda
    err1[0] = np.maximum(0,normX2-2.*np.sum(WtX*H)+np.sum(WtW*(H@H.T)))
    err2[0] = np.log(np.linalg.det(WtW + delta*np.eye(r)));  #OK
    
    lamb = lamb * np.maximum(1e-6,err1[0]) / (np.abs( err2[0] ))
    
    e[0] = err1[0] + lamb * err2[0] # OK


    # projection model for H
    if model == 1:
        proj = 1
    elif model == 2:
        proj = 2
    elif model == 3:
        proj = 0


    # Main loop 
    for ii in range(1,maxiter):
        if np.mod(ii,200) == 0:
            print(ii)

        #*** Update W ***
        XHt = X@H.T
        HHt = H@H.T

        Y = np.linalg.inv( ( W.T@W + delta*np.eye(r) ) )
        A = lamb*Y + HHt

        if model <= 2:
            W,irr = FGM_MM_nonneg(A,XHt,W,inneriter,proj=1)
        elif model == 3:
            W,irr = FGM_MM_nonneg(A,XHt,W,inneriter,proj=2)

        # *** Update H ***
        Hinit = H


        H,WtW,WtX = nnls_FPGM(X,W,H=Hinit,proj=proj,inneriter=inneriter,delta=delta)


        err1[ii] = np.maximum(0, normX2 - 2.*np.sum(WtX*H)  + np.sum(WtW*(H@H.T)))
        err2[ii] = np.log(np.linalg.det(WtW + delta*np.eye(r)))

        e[ii] = err1[ii] + lamb * err2[ii]

        # Tuning lambda to obtain options.target relative error
        if np.isscalar(target):
            if np.sqrt(err1[ii])/normX > target+0.001:
                lamb = lamb*0.95
            elif np.sqrt(err1[ii])/normX < target-0.001:
                lamb = lamb*1.05
          

    return W,H,e,err1,err2



######## FISTA step of the pGMCA algorithm ########

def FISTA_step(X, A, S, gamma, mu, max_iter = 100):
    l = 0
    A_prev = A.copy()
    A_est = A.copy()
    eps = 1e-6
    while (l < 1) or (l < max_iter and np.linalg.norm(A_est - A_prev)/np.linalg.norm(A_prev) > eps):
        A_prev = A_est.copy()
        A_est = prox_oblique(prox_positive(A_prev - gamma*grad_poisson_likelihood(X, A_prev@S, mu) @ S.T))
        l += 1
    return A_est

######## GFBS step of the pGMCA algorithm ########

def GFBS_step(X, Y, A, gamma, mu, mu0, t1, t2, lbd, max_iter = 100):
    mu1 = 1 - mu0
    n, t = Y.shape
    eps = 1e-5
    U0, U1 = Y.copy(), Y.copy()
    Y_prev, Y_est = Y.copy(), Y.copy()
    J = 2
    cs, ws = np.empty((n, t1, t2)), np.empty((n, t1, t2, J))
    
    criterions = [0]
    l = 0
    while (l < 1) or (l < max_iter and criterions[-1] > eps):
        Y_prev = Y_est.copy()

        # Gradient of the data fidelity term
        gradY = A.T @ grad_poisson_likelihood(X, A @ Y, mu)

        U_tmp = (2*Y_prev - U0 - gamma*gradY).reshape(n, t1, t2)
        for i in range(n):
            cs[i], ws[i] = st2.Starlet_Forward2D(U_tmp[i], J=J , boption=2)
        alpha = np.concatenate([cs.reshape(n, t), ws.reshape(n, J*t)], axis=1) 
        alpha = prox_l1(alpha, gamma*lbd[:,None])
        cs, ws = alpha[:,:t].reshape(n, t1, t2),  alpha[:,t:].reshape(n, t1, t2, J) 

        for i in range(n):
            U_tmp[i] = st2.Starlet_Backward2D(cs[i], ws[i])
        U0 += U_tmp.reshape(n, t) - Y_prev
        U1 += prox_positive(2*Y_prev - U1 - gamma*gradY) - Y_prev

        # update of Y
        Y_est = mu0*U0 + mu1*U1
        criterion = np.linalg.norm(Y_est - Y_prev)/np.linalg.norm(Y_prev)
        criterions.append(criterion)
        l += 1
        
    return Y_est, criterions[1:]


######## GMCA algorithm ########

def GMCA(X, A_0, max_iter = 100):
    m, n = A_0.shape
    A = A_0.copy()
    S = np.zeros((n, X.shape[1]))
    
    for k in range(max_iter):
        AinvX = np.linalg.pinv(A) @ X
        lbd = 3*mad(AinvX)
        S = prox_l1(AinvX, lbd[:,None])
        # S = prox_positive(AinvX)
        
        XSinv = X @ np.linalg.pinv(S)         
        A = prox_oblique(XSinv)
        # A = prox_positive(XSinv)
        
    return S, A


######## pGMCA algorithm ########

def pGMCA(Data, A_init, iter=100, iter_gmca=1000, iter_gfbs=1000, iter_fista=1000, plot = False):
    m, t1, t2 = Data.shape
    t = t1 * t2
    X = Data.reshape(m, t)
    n = A_init.shape[1]
    
    # initialization of A and S with classical GMCA algorithm
    S_est, A_est = GMCA(X, A_init, iter_gmca)
    print('Ok GMCA')
    # A_est = A_init.copy()
    print(f'Ainit : {A_est.shape}')
    print(f'Sinit : {S_est.shape}')

    # initialization of the hyper parameters
    mu = np.mean(X)/2
    mu0 = 0.5
    nu = 1e-3
    J = 2
    gradS = A_est.T @ grad_poisson_likelihood(X, A_est @ S_est, mu)
    cs, ws = np.empty((n, t1, t2)), np.empty((n, t1, t2, J))
    cs_grad, ws_grad = np.empty((n, t1, t2)), np.empty((n, t1, t2, J))
    for i in range(n):
        cs[i], ws[i] = st2.Starlet_Forward2D(S_est[i].reshape(t1, t2), J=J , boption=2)
        cs_grad[i], ws_grad[i] = st2.Starlet_Forward2D(gradS[i].reshape(t1, t2), J=J , boption=2)
    alpha = np.concatenate([cs.reshape(n, t), ws.reshape(n, J*t)], axis=1) 
    alpha_grad = np.concatenate([cs_grad.reshape(n, t), ws_grad.reshape(n, J*t)], axis=1) 
    print(alpha.shape, alpha_grad.shape)
    lbd = 1*mad(alpha_grad)
    Lambda = np.zeros_like(alpha)
    for i in range(n):
        for j in range(t):
            Lambda[i, j] = lbd[i] * nu/(nu + abs(alpha[i, j])/np.max(alpha[i]))
    print('Ok initialization parameters')


    k = 0
    criterions = []
    while k < iter:
        if (k+1)%1 == 0:
            print(f'iteration : {k+1}')
        # update of S
        gammaS = mu/np.linalg.norm(A_est.T @ A_est, ord=2)
        S_est, criterion = GFBS_step(X, S_est, A_est, gammaS, mu, mu0, t1, t2, lbd, iter_gfbs)
        criterions += criterion
        
        if plot:
            plt.figure()
            plt.plot(criterion)
            plt.show()

        # update of A
        gammaA = mu/np.linalg.norm(S_est @ S_est.T, ord=2)
        A_est = FISTA_step(X, A_est, S_est, gammaA, mu, iter_fista)

        k += 1

    return S_est, A_est, criterions