# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 15:13:30 2020

@author: Sima Soltani
"""

import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/rasbt/'
                 'python-machine-learning-book-3rd-edition'
                 '/master/ch10/housing.data.txt',
                 header = None,
                 sep = '\s+')
df.columns = ['CRIM','ZN','INDUS','CHAS','NOX','RM',
               'AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT','MEDV']
df.head()

import matplotlib.pyplot as plt
from mlxtend.plotting import scatterplotmatrix
cols = ['LSTAT','INDUS','NOX','RM','MEDV']

scatterplotmatrix(df[cols].values,figsize=(10,8),
                  names = cols,alpha = 0.5)
plt.tight_layout()
plt.show()

from mlxtend.plotting import heatmap
import numpy as np
cm = np.corrcoef(df[cols].values.T)
hm = heatmap(cm,
             row_names=cols,
             column_names=cols)
plt.show()

X=df[['RM']].values
y = df['MEDV'].values
from sklearn.preprocessing import StandardScaler
from LinearRegressionGD import LinearRegressionGD
sc_x = StandardScaler()
sc_y = StandardScaler()
X_std = sc_x.fit_transform(X)
y_std = sc_y.fit_transform(y[:,np.newaxis]).flatten()
lr = LinearRegressionGD()
lr.fit(X_std,y_std)
plt.plot(range(1,lr.n_iter+1),lr.cost_)
plt.ylabel('SSE')
plt.xlabel('Epoch')
plt.show()

def lin_regplot(X,y,model):
    plt.scatter(X,y,c='steelblue',edgecolor='white',s=70)
    plt.plot(X,model.predict(X),color='black',lw=2)
    return None


lin_regplot(X_std,y_std,lr)
plt.xlabel('Average number of rooms [RM] (Standardized)')
plt.ylabel('Price in $1000s [MEDV] (Standardized)')
plt.show()

from sklearn.linear_model import LinearRegression
slr = LinearRegression()
slr.fit(X,y)
y_pred = slr.predict(X)
slr.coef_[0]
slr.intercept_

lin_regplot(X,y,slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000 [MEDV]')
plt.show()
