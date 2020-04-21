# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:30:19 2020

@author: Sima Soltani
"""

#compute convolution using numpy
import numpy as np
def conv1d(x,w,p=0,s=1):
    w_rot = np.array(w[::-1])
    x_padded = np.array(x)
    if p>0:
        zero_pad =np.zeros(shape=p)
        x_padded = np.concatenate([zero_pad,x_padded,zero_pad])
    res=[]
    for i in range(0,int(len(x)/s),s):
        res.append(np.sum(x_padded[i:i+w_rot.shape[0]]*w_rot))
    return np.array(res)

##testing
x=[1,3,2,4,5,6,1,3]
w=[1,0,3,1,2]

print('conv1d implementation:',conv1d(x,w,p=2,s=1))

print('convolution result:', np.convolve(x,w,mode='same'))
        

#implement conv2d and compare to the one in scipy.signal
import numpy as np
import scipy.signal

def conv2d (x,w,p=(0,0),s=(1,1)):
    W_rot = np.array(w)[::-1,::-1]
    X_origin = np.array(x)
    n1 = X_origin.shape[0]+2*p[0]
    n2 = X_origin.shape[1]+2*p[1]
    
    X_padded = np.zeros(shape=(n1,n2))
    X_padded[p[0]:p[0]+X_origin.shape[0],
             p[1]:p[1]+X_origin.shape[1]]=X_origin
    
    res = []
    for i in range(0,int((X_padded.shape[0]-W_rot.shape[0])/s[0])+1,s[0]):
        res.append([])
        for j in range(0,int((X_padded.shape[1]-W_rot.shape[1])/s[1]+1),s[1]):
            X_sub = X_padded[i:i+W_rot.shape[0],
                               j:j+W_rot.shape[1]]
            res[-1].append(np.sum(X_sub *W_rot))
    return (np.array(res))


X=[[1,3,2,4],[5,6,1,3],[1,2,0,2],[3,4,3,2]]
W = [[1,0,3],[1,2,1],[0,1,1]]
print('Con2d Implementation :\n', conv2d(X,W,p=(1,1),s=(1,1)))

print('Scipy conv2d results:\n',scipy.signal.convolve2d(X,W,mode='same'))
