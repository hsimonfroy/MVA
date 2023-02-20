################# DEFINES THE BSS CLASS

class BSS:
	def __init__(self, data,A0, NbrSources,nmax):
		import numpy as mp
		z = mp.shape(data)
		source = mp.random.randn(NbrSources,z[1])
		self.X = data
		self.S = source
		self.A = A0
		self.nmax = nmax
		self.ns = NbrSources 

################# DEFINES THE MEDIAN ABSOLUTE DEVIATION	

def mad(xin = 0):

	import numpy as np
	import scipy.linalg as lng 

	z = np.median(abs(xin - np.median(xin)))/0.6735
	
	return z
	
################# DEFINES THE MEDIAN ABSOLUTE DEVIATION	

def GenerateMixture(n=2,t=1024,m=2,p=0.02,SType=1,CdA = 1,noise_level = 120):

	import numpy as np
	import scipy.linalg as lng 
	
	A = np.random.randn(m,n)
	uA,sA,vA = np.linalg.svd(A)
	sA = np.linspace(1/CdA,1,n) 
	A = np.dot(uA[:,0:n],np.dot(np.diag(sA),vA[:,0:n].T))
 
	S = np.zeros((n,t))
	
	if SType == 0:
		#print('Generating Bernoulli-Gaussian sources')
			
		K = np.floor(p*t)
			
		for r in range(0,n):
			u = np.arange(0,t)
			np.random.shuffle(u)
			S[r,u[0:K]] = np.random.randn(K)
			
	elif SType == 1:
		#print('Gaussian sources')
		
		S = np.random.randn(n,t)
		
	elif SType == 2:
		#print('Uniform sources')

		S = np.random.rand(n,t) - 0.5
		
	elif SType == 4:
		#print('SPC sources')
		
		K = np.floor(p*t)
		Kc = np.floor(K*0.3) #--- 30% have exactly the same locations
			
		u = np.arange(0,t)
		np.random.shuffle(u)
		S[:,u[0:Kc]] = np.random.randn(n,Kc)
			
		for r in range(0,n):
			v = Kc + np.arange(0,t - Kc)
			df = K-Kc
			np.random.shuffle(v)
			v = v[0:df]
			v = v.astype(np.int64)
			S[r,u[v]] = np.random.randn(df)
		
	elif SType == 3:
		#print('Approx. sparse sources')
		
		S = np.random.randn(n,t)
		S = np.power(S,3)
		
	else:
		print('SType takes an unexpected value ...')
		
	
	X = np.dot(A,S)
	
	if noise_level < 120:
		#--- Add noise
		N = np.random.randn(m,t)
		N = np.power(10.,-noise_level/20.)*lng.norm(X)/lng.norm(N)*N
		X = X + N
	
	return X,A,S

################# CODE TO PERFORM A PCA

def Perform_PCA(X,n):

	import numpy as np
	import scipy.linalg as lng 
	import copy as cp
	
	Y = cp.copy(X)
	Yp = np.dot(np.diag(np.mean(Y,axis=1)),np.ones(np.shape(Y)))
	Y = Y - Yp

	Ry = np.dot(Y,Y.T)
	Dy, Uy = np.linalg.eig(Ry)
	
	Uy = Uy[:,0:n]
	Sy = np.dot(Uy.T,X)
	
	return Uy , Sy

################ END OF PCA

################# CODE TO PERFORM ILC

def Perform_ILC(X,colcmb):

	import numpy as np
	import scipy.linalg as lng 
	import copy as cp
	
	# - Remove the mean value
	
	Y = cp.copy(X)
	Yp = np.dot(np.diag(np.mean(Y,axis=1)),np.ones(np.shape(Y)))
	Y = Y - Yp

	iRy = lng.inv(np.dot(Y,Y.T))

	c = np.reshape(colcmb,(len(colcmb),1))
	
	w = 1./np.dot(c.T,np.dot(iRy,c)) * np.dot(c.T,iRy)
	
	s = np.dot(w,X)
	
	return s,w

################ END OF ILC

################# CODE TO PERFORM FastICA

def Perform_FastICA(Y,n,NL='exp'):

	import numpy as np
	import scipy.linalg as lng 
	import copy as cp
	from sklearn.decomposition import FastICA
	
	X = cp.copy(Y)
	X = X.T
	
	fpica = FastICA(fun=NL)

	S = fpica.fit(X).transform(X).T  # Get the estimated sources
	A = fpica.mixing_  # Get estimated mixing matrix

	return A , S

################ END OF FastICA

################# CODE TO PERFORM A INFOMAX

def Perform_InfoMax(X,n,NGType = 1):

	import numpy as np
	import scipy.linalg as lng 
	import copy as cp
	import nloptim as nlopt

	MaxNrIte = 5000
		
	Xp = cp.copy(X)

	scaleX = np.max([abs(np.max(X)),abs(np.min(X))])

	Xp = Xp/scaleX;

	# Whitening the data
	
	Yp = np.dot(np.diag(np.mean(Xp,axis=1)),np.ones(np.shape(Xp)))
	Xp = Xp - Yp

	Ry = np.dot(Xp,Xp.T)
	Dy, Uy = np.linalg.eig(Ry)

	Proj = np.dot(np.diag(Dy[0:ASize[1]]),Uy[:,0:ASize[1]].T)  
	iProj = np.dot(Uy[:,0:ASize[1]],np.diag(1/Dy[0:ASize[1]]))  
	Xp = np.dot(Proj,Xp)

	# initialize optimize parameters

	ucminf_opt = np.array([1 ,1e-6 ,1e-10, MaxNrIte])

	# initialize variables

	Xsize = np.shape(Xp);
	W = np.eye(Xsize[0]);

	# optimize
	
	par = nlopt.nlopt_params(Xp,Xsize[0],Xsize[0])

	Win = np.reshape(W,(Xsize[0]*Xsize[0],1))

	Wout = nlopt.ucminf( NGType , par , Win , ucminf_opt )
#	if NGType == 2 [W,info] = ucminf( 'ica_MLf_t_tanh' , par , W(:) , ucminf_opt );end

	Wout = np.reshape(Wout,(Xsize[0],Xsize[0]))

	Sout = np.dot(Wout,Xp)

	Wout = np.dot(iProj,Wout)

	return Wout,Sout
	
################ END OF INFOMAX
	
################# CODE TO COMPUTE THE MIXING MATRIX CRITERION (AND SOLVES THE PERMUTATION INDETERMINACY)	
	
def Eval_BSS(A0,S0,A,S):

	import numpy as np
	import scipy.linalg as lng 
		
	Diff = np.dot(lng.inv(np.dot(A.T,A)),np.dot(A.T,A0))
	
	z = np.shape(A)
	
	for ns in range(0,z[1]):
		Diff[ns,:] = abs(Diff[ns,:])/max(abs(Diff[ns,:]))
		
	Q = np.ones(z)
	Sq = np.ones(np.shape(S))
	
	for ns in range(0,z[1]):
		Q[:,np.nanargmax(Diff[ns,:])] = A[:,ns]
		Sq[np.nanargmax(Diff[ns,:]),:] = S[ns,:]
			
	Diff = np.dot(lng.inv(np.dot(Q.T,Q)),np.dot(Q.T,A0))
	
	for ns in range(0,z[1]):
		Diff[ns,:] = abs(Diff[ns,:])/max(abs(Diff[ns,:]))
		
	p = (np.sum(Diff) - z[1])/(z[1]*(z[1]-1))
	
	return p
	
################ END OF EVALUATION CRITERION

################# Data cube to matrix conversion

def Cube2Mat(X,sdim=0):
	
 	# sdim corresponds to the dimension observation
 	
	import numpy as np
	import scipy.linalg as lng 
	import copy as cp
	
	# - Remove the mean value
	
	Y = cp.copy(X)
	z = np.shape(Y)
	
	if sdim == 0:
		M = np.zeros((z[0],z[1]*z[2]))
		for r in range(0,z[0]):
			M[r,:] = np.reshape(Y[r,:,:],(1,z[1]*z[2]))
		
	if sdim == 1:
		M = np.zeros((z[1],z[0]*z[2]))
		for r in range(0,z[1]):
			M[r,:] = np.reshape(Y[:,r,:],(1,z[0]*z[2]))
	
	if sdim == 2:
		M = np.zeros((z[2],z[0]*z[1]))
		for r in range(0,z[2]):
			M[r,:] = np.reshape(Y[:,:,r],(1,z[0]*z[1]))
	
	return M

################ Cube2Mat

################# Matrix to data cube conversion

def Mat2Cube(M,nx,ny,sdim=0):
	
 	# sdim corresponds to the dimension observation
 	
	import numpy as np
	import scipy.linalg as lng 
	import copy as cp
	
	# - Remove the mean value
	
	Y = cp.copy(M)
	z = np.shape(Y)
	
	if sdim == 0:
		X = np.zeros((z[0],nx,ny))
		for r in range(0,z[0]):
			X[r,:,:] = np.reshape(Y[r,:],(nx,ny))
		
	if sdim == 1:
		X = np.zeros((nx,z[0],ny))
		for r in range(0,z[0]):
			X[:,r,:] = np.reshape(Y[r,:],(nx,ny))
	
	if sdim == 2:
		X = np.zeros((nx,ny,z[0]))
		for r in range(0,z[0]):
			X[:,:,r] = np.reshape(Y[r,:],(nx,ny))
	
	return X

################ SOBI

def sobi(X=0,n=0,p=5):

	import numpy as np
	import scipy.linalg as lng 
	import copy as cp
	import nloptim as nlopt

	nx = np.shape(X)
	m = nx[0]
	N = nx[1]

	if n==0:
		n=m

	Xp = cp.copy(X)

	# Whitening the data
	
	Yp = np.dot(np.diag(np.mean(Xp,axis=1)),np.ones(np.shape(Xp)))
	Xp = Xp - Yp

	Ry = np.dot(Xp,Xp.T)
	Dy, Uy = np.linalg.eig(Ry)

	Proj = np.dot(np.diag(Dy[0:n]),Uy[:,0:n].T)  
	iProj = np.dot(Uy[:,0:n],np.diag(1/Dy[0:n]))  
	Xp = np.dot(Proj,X)
	
	k=0
	pm=p*m
	
	M = np.zeros((m,p*m))
	
	for u in np.arange(0,pm,m):
		X1 = Xp[:,k:N]
		X2 =  np.transpose(Xp[:,0:N-k])
		Rxp=1/(N-k+1)*np.dot(X1,X2)
		M[:,u:u+m]=lng.norm(Rxp,'fro')*Rxp
		k=k+1

	#
	# Perform joint diagonalization
	#

	epsil=1/np.sqrt(N)/100 
	encore=1 
	V=np.eye(m)
	
	g = np.zeros((3,p),dtype = complex)
	
	while encore: 
		encore=0;
		for p in range(m-1):
			for q in range(p+1,m):
				qr = np.arange(q,pm,m)
				pr = np.arange(p,pm,m)
				g[0,:] = M[p,pr]-M[q,qr]
				g[1,:] = M[p,qr]+M[q,pr]
				g[2,:] = 1j*(M[q,pr]-M[p,qr])
				D,vcp = np.linalg.eig(np.real(np.dot(g,g.T)))
				D = np.array(D);
				K = D.argsort();
				K = K[::-1]
				angles=vcp[:,K[2]];
				angles=np.sign(angles[0])*angles;
				c=np.sqrt(0.5+angles[0]/2);
				sr=0.5*(angles[1]-np.complex(0,angles[2]))/c; 
				sc=np.conj(sr);
				oui = (abs(sr) > epsil) 
				encore=encore or oui ;
				if oui: 
					colp=M[:,pr];
					colq=M[:,qr];
					M[:,pr]=c*colp+sr*colq;
					M[:,qr]=c*colq-sc*colp;
					rowp=M[p,:];
					rowq=M[q,:];
					M[p,:]=c*rowp+sc*rowq;
					M[q,:]=c*rowq-sr*rowp;
					temp=V[:,p];
					V[:,p]=c*V[:,p]+sr*V[:,q];
					V[:,q]=c*V[:,q]-sc*temp;


	S = np.dot(V.T,Xp)

	H = np.dot(iProj,V); 

	return H,S

################# Useful codes

def length(x=0):

    import numpy as np
    l = np.max(np.shape(x))
    return l

################# 1D convolution	

def filter_1d(xin=0,h=0,boption=3):

    import numpy as np
    import scipy.linalg as lng
    import copy as cp    
    
    x = np.squeeze(cp.copy(xin));
    n = length(x);
    m = length(h);
    y = cp.copy(x);

    z = np.zeros(1,m);

    for r in range(np.int(np.floor(m/2))):
                
        if boption == 1: # --- zero padding
                        
            z = np.concatenate([np.zeros(m-r-np.floor(m/2)-1),x[0:r+np.floor(m/2)+1]],axis=0)
        
        if boption == 2: # --- periodicity
            
            z = np.concatenate([x[n-(m-(r+np.floor(m/2)))+1:n],x[0:r+np.floor(m/2)+1]],axis=0)
        
        if boption == 3: # --- mirror
            
            u = x[0:m-(r+np.floor(m/2))-1];
            u = u[::-1]
            z = np.concatenate([u,x[0:r+np.floor(m/2)+1]],axis=0)
                                     
        y[r] = np.sum(z*h)
        
        
    

    a = np.arange(np.int(np.floor(m/2)),np.int(n-m+np.floor(m/2)),1)

    for r in a:
        
        y[r] = np.sum(h*x[r-np.floor(m/2):m+r-np.floor(m/2)])
    

    a = np.arange(np.int(n-m+np.floor(m/2)+1),n,1)

    for r in a:
            
        if boption == 1: # --- zero padding
            
            z = np.concatenate([x[r-np.floor(m/2):n],np.zeros(m - (n-r) - np.floor(m/2))],axis=0)
        
        if boption == 2: # --- periodicity
            
            z = np.concatenate([x[r-np.floor(m/2):n],x[0:m - (n-r) - np.floor(m/2)]],axis=0)
        
        if boption == 3: # --- mirror
                        
            u = x[n - (m - (n-r) - np.floor(m/2) -1)-1:n]
            u = u[::-1]
            z = np.concatenate([x[r-np.floor(m/2):n],u],axis=0)
                    
        y[r] = np.sum(z*h)
    	
    return y
 
################# 1D convolution with the "a trous" algorithm	

def Apply_H1(x=0,h=0,scale=1,boption=3):

	import numpy as np
	import copy as cp
	
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
	

################# 2D "a trous" algorithm

def Starlet_Forward(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	import numpy as np
	import copy as cp
	
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
	
	################# 2D "a trous" algorithm

def Starlet_Inverse(c=0,w=0):

	import numpy as np
	
	x = c+np.sum(w,axis=2)

	return x
