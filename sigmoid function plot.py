# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 10:45:46 2020

@author: Sima Soltani
"""
#z = wT*x
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(z):
    return (1/(1+np.exp(-z)))

z = np.arange(-7,7,0.1)
phi_z = sigmoid(z)
plt.plot(z,phi_z)
plt.axvline(0.0,color='k')
plt.ylim(-0.1,1.1)
plt.xlabel('z')
plt.ylabel('$\phi (z) $')
#y axis ticks and guidline
plt.yticks([0.0,0.5,1.0])
ax = plt.gca()
ax.yaxis.grid(True)
plt.tight_layout()
plt.show()
