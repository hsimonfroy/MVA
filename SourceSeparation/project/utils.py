import numpy as np
import matplotlib.pyplot as plt

# Function to plot the mixing matrix and the sources
def plotAS(MixingMatrix, Sources, title=None, cmap="hot", square=True):
    if square:
        nrow, nline = 2, 2
        plt.figure(figsize = (10, 10)) 
    else: #line
        nrow, nline = 1, 4
        plt.figure(figsize = (16, 4))
        
    plt.subplot(nrow,nline,1)
    plt.plot(MixingMatrix)
    plt.gca().set_box_aspect(1)
    plt.title('Mixing Matrix')

    for i in range(3):
        plt.subplot(nrow,nline,i+2)
        plt.imshow(Sources[i], cmap=cmap)
        plt.title(f"Source {i+1}")

    if title is not None:
        plt.suptitle(title, fontsize=16)
    plt.show()

# Proximal operators
def prox_l1(x, thrd):
    return np.fmax(0, x - thrd) + np.fmin(0, x + thrd)

def prox_oblique(A):
    normeA = np.sqrt(np.sum(A**2, axis=0)) 
    for ii in range(len(normeA)):
        if normeA[ii] > 1:
            A[:,ii] /= normeA[ii]
    return A

def prox_positive(x):
    return np.fmax(x, 0)

# Median absolute deviation
def mad(z):
    return np.median(np.abs(z - np.median(z, axis=-1)[...,None]), axis=-1)/0.6735

# gradient of the Poisson log-likelihood
def grad_poisson_likelihood(X, Y, mu):
    return (1/(2*mu)) * (Y + mu) * (1 - (1 - 4*mu*(Y - X)/(Y + mu)**2)**0.5)



