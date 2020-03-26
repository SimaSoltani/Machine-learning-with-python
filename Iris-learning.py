# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 18:48:09 2020

@author: Sima Soltani
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from Perceptron import Perceptron
s = os.path.join('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

print ('URL:', s)
df = pd.read_csv(s, header = None, encoding = 'utf-8')

df.tail()
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values

# plot data
plt.scatter(X[:50,0],X[:50,1],
            color = 'red',marker = 'o',label = 'setosa')
plt.scatter(X[50:100,0],X[50:100,1],
            color = 'blue', marker = 'x', label = 'versicolor')
plt.xlabel ('Sepal length [cm]')
plt.ylabel ('Petal length [cm]')
plt.legend(loc = 'upper left')
plt.show()

ppn = Perceptron(eta = 0.1, n_iter = 10)
ppn.fit(X,y)

plt.plot(range(1,len(ppn.errors_)+1),
         ppn.errors_,marker = 'o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')
plt.show()

from matplotlib.colors import ListedColormap

def plot_decision_regions (X,y,classifier,resolution  = 0.02):
    #setup marker generator and color map
    markers = ('s','x','o','^','v')
    colors = ('red','blue','lightgreen','grey','cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    #plot the decision surface
    x1_min,x1_max = X[:,0].min() -1, X[:,0].max()+1
    x2_min,x2_max = X[:,1].min() -1 , X[:,1].max() +1
    xx1,xx2 = np.meshgrid(np.arange(x1_min,x1_max,resolution),
                           np.arange(x2_min,x2_max,resolution))
    z  = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    
    plt.contourf(xx1,xx2,z,alpha = 0.3,cmap=cmap)
    plt.xlim(xx1.min(),xx1.max())
    plt.ylim(xx2.min(),xx2.max())
    
    #plot class examples
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x = X[y == cl,0],
                    y = X[y == cl,1],
                    alpha = 0.8,
                    c = colors[idx],
                    marker = markers[idx],
                    label = cl,
                    edgecolor = 'black')


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('patal length [cm]')
plt.legend(loc='upper left')
plt.show()