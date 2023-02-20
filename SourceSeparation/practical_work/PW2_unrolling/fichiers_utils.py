import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scipy.stats import gennorm
from torch.utils.data import Dataset,Subset
from random import shuffle
from sklearn.model_selection import train_test_split
import pickle
import copy as cp
from munkres import Munkres


def length(x=0):

	l = np.max(np.shape(x))
	return l

def filter_1d(xin=0,h=0,boption=3):

	x = np.squeeze(cp.copy(xin));
	n = length(x);
	m = length(h);
	y = cp.copy(x);
	m2 = np.int(m/2.)
	z = np.zeros(1,m);
	for r in range(np.int(m2)):
		if boption == 1: # --- zero padding
			z = np.concatenate([np.zeros(m-r-m2-1),x[0:r+m2+1]],axis=0)
		if boption == 2: # --- periodicity
			z = np.concatenate([x[n-(m-(r+m2))+1:n],x[0:r+m2+1]],axis=0)
		if boption == 3: # --- mirror
			u = x[0:m-(r+m2)-1];
			u = u[::-1]
			z = np.concatenate([u,x[0:r+m2+1]],axis=0)
		y[r] = np.sum(z*h)
	a = np.arange(np.int(m2),np.int(n-m+m2),1)

	for r in a:
		y[r] = np.sum(h*x[r-m2:m+r-m2])
	a = np.arange(np.int(n-m+m2+1),n,1)

	for r in a:
		if boption == 1: # --- zero padding
			z = np.concatenate([x[r-m2:n],np.zeros(m - (n-r) - m2)],axis=0)
		if boption == 2: # --- periodicity
			z = np.concatenate([x[r-m2:n],x[0:m - (n-r) - m2]],axis=0)
		if boption == 3: # --- mirror	
			u = x[n - (m - (n-r) - m2 -1)-1:n]
			u = u[::-1]
			z = np.concatenate([x[r-m2:n],u],axis=0)
		y[r] = np.sum(z*h)

	return y


def Apply_H1(x=0,h=0,scale=1,boption=3):

	m = length(h)

	if scale > 1:
		p = (m-1)*np.power(2,(scale-1)) + 1
		g = np.zeros( p)
		z = np.linspace(0,m-1,m)*np.power(2,(scale-1))
		g[z.astype(int)] = h

	else:
		g = h

	y = filter_1d(x,g,boption)

	return y

def Starlet_Forward(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=2,boption=3):

	nx = np.shape(x)
	c = np.zeros((nx[0],nx[1]))
	w = np.zeros((nx[0],nx[1],J))
	c = cp.copy(x)
	cnew = cp.copy(x)

	for scale in range(J):
		for r in range(nx[0]):
			cnew[r,:] = Apply_H1(c[r,:],h,scale,boption)
		for r in range(nx[1]):
			cnew[:,r] = Apply_H1(cnew[:,r],h,scale,boption)
		w[:,:,scale] = c - cnew;
		c = cp.copy(cnew);
	return c,w
	
def Starlet_Inverse(c=0,w=0):

	x = c+np.sum(w,axis=2)

	return x

def invert_permutation(p):
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s
    
def perm_w(w):
	w = w.ravel()
	array = np.arange(len(w))
	np.random.shuffle(array)
	w = w[array]
	w_1d = w.reshape(1,w.size)
	return w_1d, array	
		
def get_w():
	with open('sources.pkl', 'rb') as f:
		sources = pickle.load(f)

	c1,w1 = Starlet_Forward(sources['sync'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	c2,w2 = Starlet_Forward(sources['therm'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	c3,w3 = Starlet_Forward(sources['fe1'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	c4,w4 = Starlet_Forward(sources['fe2'],h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
	
	perm = []
	
	w1_1d, array1 = perm_w(w1)
	perm.append(array1)
	w2_1d, array2 = perm_w(w2)
	perm.append(array2)
	w3_1d, array3 = perm_w(w3)
	perm.append(array3)
	w4_1d, array4 = perm_w(w4)
	perm.append(array4)
	
	c1_1d = c1.reshape(1,c1.size)
	c2_1d = c2.reshape(1,c2.size)
	c3_1d = c3.reshape(1,c3.size)
	c4_1d = c4.reshape(1,c4.size)
	w = np.concatenate((w1_1d,w2_1d,w3_1d,w4_1d))
	c = np.concatenate((c1_1d,c2_1d,c3_1d,c4_1d))
	return c,w,sources,perm

def load_data():
	spectres = np.load('spectra.npy', allow_pickle= True)
	spectres = spectres.reshape(1)[0]
	A1 = spectres['sync']
	A2 = spectres['therm'][:900]
	A3 = spectres['fe1']
	A4 = spectres['fe2']
		
	return A1,A2,A3,A4
	
def mixing_synthetic(beta = 0.3 , t =500, noise=True,normalize_S_lines = False):
	A1,A2,A3,A4 = load_data()
	
	X = np.zeros((A1.shape[0],A1.shape[1],t))
	A = np.zeros((A1.shape[0],A1.shape[1],4))
	S = np.zeros((A1.shape[0],4,t))
	for i in range(A1.shape[0]):
		A[i] = np.concatenate((A1[i].reshape(-1,1),A2[i].reshape(-1,1),A3[i].reshape(-1,1),A4[i].reshape(-1,1)) , axis = 1)
		S1 = gennorm.rvs(beta,size=(1,t))
		S2 = gennorm.rvs(beta,size=(1,t))
		S3 = gennorm.rvs(beta,size=(1,t)) 
		S4 = gennorm.rvs(beta,size=(1,t))
		S[i]=np.concatenate((S1,S2,S3,S4))
		if normalize_S_lines:
			for j in range(S.shape[1]):
				S[i][j,:] = S[i][j,:]/np.linalg.norm(S[i][j,:]) 
		X[i] = np.matmul(A[i],S[i])
		if noise:
			noise_level = 30
			N = np.random.randn(X[i].shape[0],X[i].shape[1])
			N = 10.**(-noise_level/20.)*np.linalg.norm(X[i])/np.linalg.norm(N)*N
			X[i] = X[i] + N
	
	return (X,A,S)


def mixing_realistic(beta1 = 0.28 ,beta2=0.28, beta3=0.28,beta4=0.28, t = 3000, noise=True,normalize_S_lines = False):
	A1,A2,A3,A4 = load_data()
	
	X = np.zeros((A1.shape[0],A1.shape[1],t))
	A = np.zeros((A1.shape[0],A1.shape[1],4))
	S = np.zeros((A1.shape[0],4,t))
	for i in range(A1.shape[0]):
		A[i] = np.concatenate((A1[i].reshape(-1,1),A2[i].reshape(-1,1),A3[i].reshape(-1,1),A4[i].reshape(-1,1)) , axis = 1)
		S1 = gennorm.rvs(beta1,loc=0, scale=2*(10**-6),size=(1,int(0.27*t)))
		S2 = gennorm.rvs(beta2,loc=0, scale=4*(10**-6),size=(1,int(0.15*t)))
		S3 = gennorm.rvs(beta3,loc=0, scale=5*(10**-7),size=(1,int(0.76*t)))
		S4 = gennorm.rvs(beta4,loc=0, scale=9*(10**-7),size=(1,int(0.35*t)))
		_1 = np.zeros((1,int(0.73*t)))
		_2 = np.zeros((1,int(0.85*t)))
		_3 = np.zeros((1,int(0.24*t)))
		_4 = np.zeros((1,int(0.65*t)))
		S1 = np.concatenate((S1,_1),axis=None).reshape(1,S1.size+_1.size)
		S2 = np.concatenate((S2,_2),axis=None).reshape(1,S2.size+_2.size)
		S3 = np.concatenate((S3,_3),axis=None).reshape(1,S3.size+_3.size)
		S4 = np.concatenate((S4,_4),axis=None).reshape(1,S4.size+_4.size)
		np.random.shuffle(np.transpose(S1))
		np.random.shuffle(np.transpose(S2))
		np.random.shuffle(np.transpose(S3))
		np.random.shuffle(np.transpose(S4))
		S[i]=np.concatenate((S1,S2,S3,S4))
		if normalize_S_lines:
			for j in range(S.shape[1]):
				S[i][j,:] = S[i][j,:]/np.linalg.norm(S[i][j,:]) 
		X[i] = np.matmul(A[i],S[i])
		if noise:
			noise_level = 30
			N = np.random.randn(X[i].shape[0],X[i].shape[1])
			N = 10.**(-noise_level/20.)*np.linalg.norm(X[i])/np.linalg.norm(N)*N
			X[i] = X[i] + N
	
	return (X,A,S)

def prox(x,lmda,L):
	theta = torch.maximum(lmda/L,torch.zeros_like(lmda/L))
	res = torch.sign(x) * torch.maximum(torch.abs(x) - lmda/L, torch.zeros_like(x))
	return res

def prox_l1(S_est,thrd): #numpy
  S_est[(abs(S_est) < thrd)] = 0
  indNZ = np.where(abs(S_est) > thrd)[0]
  S_est[indNZ] = S_est[indNZ] - thrd*np.sign(S_est[indNZ])
  return S_est

def prox_theta(x,theta): #torch
	theta = torch.maximum(theta,torch.zeros_like(theta))
	res = torch.sign(x) * torch.maximum(torch.abs(x) - theta , torch.zeros_like(x))
	#res = res.type(torch.FloatTensor) #comment when using LPALM
	return res

def norm_col(A):
    An = A.copy()
    type(An)
    for ii in range(np.shape(An)[1]):
        An[:,ii] = An[:,ii]/np.sqrt(np.sum(An[:,ii]**2));
    
    return An

def prox_oblique(A):
    for ii in range(np.shape(A)[1]):
        normeA = np.sqrt(np.sum(A[:,ii]**2))
        if normeA > 0 and normeA > 1.:
            A[:,ii] /= normeA
        
    return A

def correctPerm(W0_en,W_en):
    # [WPerm,Jperm,err] = correctPerm(W0,W)
    # Correct the permutation so that W becomes the closest to W0.
    
    W0 = W0_en.copy()
    W = W_en.copy()
    
    W0 = norm_col(W0)
    W = norm_col(W)

    costmat = -W0.T@W; # Avec Munkres, il faut bien un -
    
    m = Munkres()
    Jperm = m.compute(costmat.tolist())
    
    WPerm = np.zeros(np.shape(W0))
    indPerm = np.zeros(np.shape(W0_en)[1])
    
    for ii in range(W0_en.shape[1]):
        WPerm[:,ii] = W_en[:,Jperm[ii][1]]
        indPerm[ii] = Jperm[ii][1]
        
    return WPerm,indPerm.astype(int)


def NMSE(W0_en,W_en,S0_en,S_en):
    # W0 : true mixing matrix
    # W : estimated mixing matrix
    #
    # maxAngle : cosine of the maximum angle between the columns of W0 and W
    
    W0 = W0_en.copy()
    W = W_en.copy()
    
    S0 = S0_en.copy()
    S = S_en.copy()
    
    W,indPerm = correctPerm(W0,W);
    
    S = S[indPerm,:]
    
    nmse = np.sum((S-S0)**2)/(np.sum(S0**2))
    
    return nmse

def tile(a, dim, n_tile):
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index)
    
	
def train_val_dataset(dataset, val_split= 1/6):
	train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split,random_state=10, shuffle = True)
	datasets = {}
	datasets['train'] = Subset(dataset, train_idx)
	datasets['val'] = Subset(dataset, val_idx)
	return datasets

def train_val_smalldataset(dataset, train_size= 150,val_size=50):
    val_split = val_size/(val_size+train_size)
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split,random_state=10, shuffle = True)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx[0:train_size])
    datasets['val'] = Subset(dataset, val_idx[0:val_size])
    return datasets


def split_A(dataset):

	A0 = next(iter(dataset))[1]
	l = []
	for i,(X,A,S) in enumerate(dataset):
		crit = np.sum(np.multiply(A0[:,0],A[:,0])) / (np.linalg.norm(A0[:,0]) * np.linalg.norm(A[:,0])) + np.sum(np.multiply(A0[:,1],A[:,1])) / (np.linalg.norm(A0[:,1]) * np.linalg.norm(A[:,1])) + np.sum(np.multiply(A0[:,2],A[:,2])) / (np.linalg.norm(A0[:,2]) * np.linalg.norm(A[:,2])) + np.sum(np.multiply(A0[:,3],A[:,3])) / (np.linalg.norm(A0[:,3]) * np.linalg.norm(A[:,3]))
		l.append(crit)
	ind = sorted(range(len(l)), key=lambda k: l[k])
	val_idx = ind[:150]
	train_idx = ind[150:]
	datasets = {}
	datasets['train'] = Subset(dataset, train_idx)
	datasets['val'] = Subset(dataset, val_idx)
	
	return datasets
    

def split_A_light(dataset,num_val,num_train):

	A0 = next(iter(dataset))[1]
	l = []
	for i,(X,A,S) in enumerate(dataset):
		crit = np.sum(np.multiply(A0[:,0],A[:,0])) / (np.linalg.norm(A0[:,0]) * np.linalg.norm(A[:,0])) + np.sum(np.multiply(A0[:,1],A[:,1])) / (np.linalg.norm(A0[:,1]) * np.linalg.norm(A[:,1])) + np.sum(np.multiply(A0[:,2],A[:,2])) / (np.linalg.norm(A0[:,2]) * np.linalg.norm(A[:,2])) + np.sum(np.multiply(A0[:,3],A[:,3])) / (np.linalg.norm(A0[:,3]) * np.linalg.norm(A[:,3]))
		l.append(crit)
	ind = sorted(range(len(l)), key=lambda k: l[k])
	val_idx = ind[:num_val]
	train_idx = ind[-num_train:]
	datasets = {}
	datasets['train'] = Subset(dataset, train_idx)
	datasets['val'] = Subset(dataset, val_idx)
	
	return datasets

def apply_LPALM_realistic(val_loader):	
	c,w,sources,perm = get_w()
	normw = np.zeros(w.shape[0])
	normc = np.zeros(c.shape[0])
	for j in range(w.shape[0]):
		normw[j] = np.linalg.norm(w[j,:]) 
		w[j,:] = w[j,:]/normw[j]
		plt.imshow(w[j,:].reshape(346,346),cmap='jet')
		plt.colorbar()
		plt.show()
	for j in range(c.shape[0]):
		normc[j] = np.linalg.norm(c[j,:])
		c[j,:] = c[j,:]/normc[j] 
		plt.imshow(c[j,:].reshape(346,346),cmap='jet')
		plt.colorbar()
		plt.show()

	for i, (X,A,S) in enumerate(val_loader):
		if i==132:
			A_real = A[0]
			break

	X_real_tilde_w = torch.matmul(A_real,torch.from_numpy(w))
	X_real_tilde_c = torch.matmul(A_real,torch.from_numpy(c))

	X_real = Starlet_Inverse(X_real_tilde_c.numpy(),X_real_tilde_w.numpy().reshape(X_real_tilde_w.shape[0],X_real_tilde_w.shape[1],1))
	N = np.random.randn(X_real.shape[0],X_real.shape[1])
	noise_level=30
	N = 10.**(-noise_level/20.)*np.linalg.norm(X_real)/np.linalg.norm(N)*N
	X_real = X_real + N
	X_real_tilde_w = np.empty_like(X_real)
	X_real_tilde_c = np.empty_like(X_real)
	for i in range(X_real.shape[0]):
		c_ , w_ = Starlet_Forward(X_real[i].reshape(346,346),h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3)
		X_real_tilde_c[i,:] , X_real_tilde_w[i,:] = c_.reshape(c_.size) , w_.reshape(w_.size)
	X_real_tilde_c = torch.from_numpy(X_real_tilde_c)
	X_real_tilde_w = torch.from_numpy(X_real_tilde_w)
	criterion = nn.MSELoss()
	model = torch.load('LPALM97.pth')
	model.eval()
	with torch.no_grad():
		X_real_tilde_w = X_real_tilde_w.reshape(1,X_real_tilde_w.shape[0],X_real_tilde_w.shape[1])
		S_pred_tilde_w, A_pred, S_s, A_s = model(X_real_tilde_w)
		S_pred_tilde_c = torch.matmul(torch.pinverse(A_pred),X_real_tilde_c)
	for j in range(w.shape[0]): 
		S_pred_tilde_w[0,j,:] = S_pred_tilde_w[0,j,:]*normw[j]
		S_pred_tilde_w[0,j,:] = S_pred_tilde_w[0,j,:][invert_permutation(perm[j])]
	for j in range(c.shape[0]): 
		S_pred_tilde_c[0,j,:] = S_pred_tilde_c[0,j,:]*normc[j]
		
	S_pred =  Starlet_Inverse(S_pred_tilde_c.numpy().reshape(S_pred_tilde_c.shape[1],S_pred_tilde_c.shape[2]),S_pred_tilde_w.numpy().reshape(S_pred_tilde_w.shape[1],S_pred_tilde_w.shape[2],1))

	#for i in range(S_pred.shape[0]):
	#	im = plt.imshow(S_pred[i,:].reshape(346,346))
	#	im.set_cmap('jet')
	#	plt.colorbar(im)
	#	plt.show()
	#for i in range(A_pred.shape[2]):
	#	plt.plot(A_pred[0,:,i],label='predicted')
	#	plt.plot(A_real[:,i],label='ground truth')
	#	plt.legend()
	#	plt.show()

	S_pred = torch.from_numpy(S_pred)
	sources['sync'] = sources['sync'].reshape(1,346*346)
	sources['therm'] = sources['therm'].reshape(1,346*346)
	sources['fe1'] = sources['fe1'].reshape(1,346*346)
	sources['fe2'] = sources['fe2'].reshape(1,346*346)
	GT = np.concatenate((sources['sync'],sources['therm'],sources['fe1'],sources['fe2']))
	GT = torch.from_numpy(GT)
	NMSE_S = torch.numel(GT) * criterion(GT, S_pred) / torch.norm(GT)**2
	NMSE_A = torch.numel(A_real) * criterion(A_real, A_pred[0,:,:]) / torch.norm(A_real)**2
	print(NMSE_A,NMSE_S)
	return NMSE_S, NMSE_A
	
class My_dataset(Dataset):
	def __init__(self, beta=0.3 , t=1000, noise=True, SNR=30, normalize_S_lines=False,synthetic=True):
		if synthetic == True:
			self.X, self.A, self.S = mixing_synthetic(beta, t, noise, normalize_S_lines= normalize_S_lines)
		else:
			self.X, self.A, self.S = mixing_realistic(beta1,beta2,beta3,beta4,t,noise,normalize_S_lines)
	def __getitem__(self, item):
		return self.X[item], self.A[item], self.S[item]
	def __len__(self):
		return len(self.X)
		
def plot_train_test_loss(train_loss,test_loss):
	plt.plot(train_loss)
	plt.plot(test_loss)
	plt.xlabel('epochs')
	plt.ylabel('NMSE')
	plt.legend(["training loss", "validation loss"])
	plt.yscale('log')
	plt.show()
