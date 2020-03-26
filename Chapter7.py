# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 13:27:31 2020

@author: Sima Soltani
"""

from scipy.special import comb
import math
def ensemble_error(n_classifier,error):
    k_start = int (math.ceil(n_classifier/2.))
    probs =[comb(n_classifier,k)*error**k*
            (1-error)**(n_classifier-k)
            for k in range (k_start,n_classifier+1)]
    return sum(probs)
ensemble_error(n_classifier = 11, error = 0.25)

import numpy as np
import matplotlib.pyplot as plt
error_range = np.arange(0.0,1.01,0.01)
ens_errors = [ensemble_error(n_classifier = 11, error = error ) 
              for error in error_range]
plt.plot(error_range,ens_errors,
         label = 'Ensemble error',
         linewidth = 2)
plt.plot (error_range,error_range,
          linestyle = '--', label = 'Base error',
          linewidth = 2)

plt.xlabel('Base error')
plt.ylabel('base/ensemble wrror')
plt.legend(loc='best')
plt.grid(alpha = 0.5)
plt.show() 

np.argmax(np.bincount([0,0,1],weights = [0.2,0.2,0.6]))
ex = np.array([[0.9,.1],[0.8,.2],[.4,.6]])
p = np.average(ex,axis=0,weights = [.2,.2,.6])
np.argmax(p)
