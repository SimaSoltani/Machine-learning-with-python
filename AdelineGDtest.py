# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 21:15:07 2020

@author: Sima Soltani
"""

import os
import matplotlib.pyplot as plt
from AdalineGD import AdalineGD
from AdalineSGD import AdalineSGD
#from Iris_learning import plot_decision_regions
import pandas as pd
import numpy as np
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


s = os.path.join('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')
df = pd.read_csv(s, header = None, encoding = 'utf-8')

df.tail()
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa',-1,1)
X = df.iloc[0:100,[0,2]].values


fig,ax = plt.subplots(nrows=1, ncols = 2,figsize=(10,4))
ada1 = AdalineGD(n_iter=10,eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),
           np.log10(ada1.cost_),marker='o')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('log(Sum-squared-error')
ax[0].set_title('Adeline-Learning rate 0.01')

ada2 = AdalineGD(n_iter=10,eta= 0.0001).fit(X,y)
ax[1].plot(range(1,len(ada2.cost_)+1),
           np.log(ada2.cost_),marker='o')
ax[1].set_xlabel ('Epoch')
ax[1].set_ylabel('Log(sum_squared-errot)')
ax[1].set_title('Adeline-Learning rate 0.0001')
plt.show()

X_std = np.copy(X)
X_std[:,0]=(X[:,0]-X[:,0].mean())/X[:,0].std()
X_std[:,1]=(X[:,1]-X[:,1].mean())/X[:,1].std()

ada_gd=AdalineGD(n_iter=15, eta = 0.01)
ada_gd.fit(X_std,y)

plot_decision_regions(X_std,y,classifier = ada_gd)
plt.title('Adaline-Gradient Descent')
plt.xlabel('Sepal length [standardized]')
plt.ylabel('petal length [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada_gd.cost_)+1),
         ada_gd.cost_,marker='o')
plt.xlabel('Epoches')
plt.ylabel('Sum_squared_error')
plt.tight_layout()
plt.show()
ada_sgd = AdalineSGD(n_iter = 15,eta = 0.01,random_state = 1)
ada_sgd.fit(X_std,y)

plot_decision_regions(X_std,y,classifier = ada_sgd)
plt.title('Adaline- Stochastic Gradiant Descent')
plt.xlabel('sepal length [standardized]')
plt.ylabel('petal length[Standardized]')
plt.legend(loc = 'upper left')
plt.tight_layout()
plt.show()
plt.plot(range(1,len(ada_sgd.cost_)+1),ada_sgd.cost_,marker = 'o')
plt.xlabel('Epoch')
plt.ylabel('Average Cost')
plt.tight_layout()
plt.show()
