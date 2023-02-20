# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 16:54:28 2017

@author: jbobin
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Dec 16 15:06:01 2016

@author: jbobin
"""

import numpy as np
from copy import deepcopy as dp

# Module that implements in 2D starlet transform

################# Useful codes

def length(x=0):

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
    
    m2 = np.int(np.floor(m/2))

    for r in range(m2):
                
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
    
    a = np.arange(np.int(n-m+m2+1)-1,n,1)

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
 
################# 1D convolution with the "a trous" algorithm	

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

################# 2D "a trous" algorithm

def Starlet_Forward2D(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	import copy as cp
 	
	nx = np.shape(x)
	c = np.zeros((nx[0],nx[1]),dtype=complex)
	w = np.zeros((nx[0],nx[1],J))

	c = cp.copy(x)
	cnew = cp.copy(x)
	
	for scale in range(J):
		
		for r in range(nx[0]):
			
			cnew[r,:] = Apply_H1(c[r,:],h,scale,boption)
			
		for r in range(nx[1]):
		
			cnew[:,r] = Apply_H1(cnew[:,r],h,scale,boption)
			
		w[:,:,scale] = c - cnew
		
		c = cp.copy(cnew);

	return c,w

########################

def Starlet_Forward1D(x=0,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3):

	import copy as cp
 	
	c = np.zeros((len(x),),dtype=complex)
	w = np.zeros((len(x),J))

	c = cp.copy(x)
	cnew = cp.copy(x)
	
	for scale in range(J):
     
         cnew = Apply_H1(c,h,scale,boption)
         
         w[:,scale] = c - cnew
         
         c = cp.copy(cnew);

	return c,w
	
########################

def Starlet_Backward1D(c,w):
    
    import numpy as np   
    
    return c + np.sum(w,axis=1)

########################
    
def Starlet_Backward2D(c,w):
    
    import numpy as np   
    
    return c + np.sum(w,axis=2)
    
########################

def mad(z):
    return np.median(abs(z - np.median(z)))/0.6735

########################

def Starlet_Filter1D(x=0,kmad=3,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3,L0=0,perscale=0):
    
    c,w = Starlet_Forward1D(x,h=h,J=J,boption=boption)
    
    for r in range(J):
    
        if perscale:
        	thrd = kmad*mad(w[:,r])
        else:
        	if r == 0:
        		thrd = kmad*mad(w[:,r]) # Estimate the threshold in the first scale only
        
        if L0:
        	w[:,r] = w[:,r]*(abs(w[:,r]) > thrd)
        else:
        	w[:,r] = (w[:,r] - thrd*np.sign(w[:,r]))*(abs(w[:,r]) > thrd)
    
    return Starlet_Backward1D(c,w)

########################
    
def Starlet_Filter2D(x=0,kmad=3,h=[0.0625,0.25,0.375,0.25,0.0625],J=1,boption=3,L0=0,perscale=0):
    
    c,w = Starlet_Forward2D(x,h=h,J=J,boption=boption)
    
    for r in range(J):
    
        if perscale:
        	thrd = kmad*mad(w[:,:,r])
        else:
        	if r == 0:
        		thrd = kmad*mad(w[:,:,r]) # Estimate the threshold in the first scale only
                  
        if L0:
        	w[:,:,r] = w[:,:,r]*(abs(w[:,:,r]) > thrd)
        else:
        	w[:,:,r] = (w[:,:,r] - thrd*np.sign(w[:,:,r]))*(abs(w[:,:,r]) > thrd)
    
    return Starlet_Backward2D(c,w)
