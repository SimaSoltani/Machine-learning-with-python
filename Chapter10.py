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

#RANSAC
from sklearn.linear_model import RANSACRegressor
ransac = RANSACRegressor(LinearRegression(),
                         max_trials =100,
                         min_samples = 50,
                         loss = 'absolute_loss',
                         residual_threshold = 5.0,
                         random_state=0)
ransac.fit(X,y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)
line_X=np.arange(3,10,1)
line_y_ransac = ransac.predict(line_X[:,np.newaxis])
plt.scatter(X[inlier_mask],y[inlier_mask],
            c='steelblue',edgecolor = 'white',
            marker = 'o',label='Inliers')
plt.scatter(X[outlier_mask],y[outlier_mask],
            c='limegreen',edgecolor = 'white',
            marker = 's',label='Outliers')
plt.plot(line_X,line_y_ransac,color='black',lw = 2)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='best')
plt.show()
print('the slope : %0.2f'%ransac.estimator_.coef_[0])
print('Intercept %.3f'%ransac.estimator_.intercept_)

from sklearn.model_selection import train_test_split
X=df.iloc[:,:-1].values
y = df['MEDV'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size = 0.3,
                                                 random_state = 0)
slr = LinearRegression()
slr.fit(X_train,y_train)
y_train_predict =slr.predict(X_train)
y_test_predict = slr.predict(X_test)

# prot residual plot
plt.scatter(y_train_predict,y_train_predict-y_train,
            c='steelblue',marker = 'o',edgecolor = 'white',
            label = 'Training data'
            )
plt.scatter(y_test_predict,y_test_predict-y_test,
            c='limegreen',marker = 's',edgecolor = 'white',
            label = 'Test data')
plt.xlabel('Predicted values')
plt.ylabel('Risiduals')
plt.legend(loc='best')
plt.hlines(y = 0,xmin=-10,xmax = 50,color = 'black',lw = 2)
plt.xlim([-10,50])
plt.show()

from sklearn.metrics import mean_squared_error
print('MSE train: %.3f , test: %.3f'%(mean_squared_error(y_train,y_train_predict),mean_squared_error(y_test,y_test_predict)))

from sklearn.metrics import r2_score
print('R^2 train: %.3f, test: %.3f'%(
    r2_score(y_train,y_train_predict),r2_score(y_test,y_test_predict)))


#Plynomial Features
from sklearn.preprocessing import PolynomialFeatures
X=np.array([258.0,270.0,294.0,320.0,342.0,
            368.0,396.0,446.0,480.0,586.0])[:,np.newaxis]
y = np.array([236.4,234.4,252.8,298.6,314.2,
              342.2,360.8,368.0,391.2,390.8])
lr = LinearRegression()
pr = LinearRegression()
quadratic = PolynomialFeatures(degree =2)
X_quad = quadratic.fit_transform(X)

lr.fit(X,y)
X_fit = np.arange(250,600,10)[:,np.newaxis]
y_lin_fit=lr.predict(X_fit)
pr.fit(X_quad,y)
y_quad_fit = pr.predict(quadratic.fit_transform(X_fit))

plt.scatter(X,y,label='Training points')
plt.plot(X_fit,y_lin_fit,
         label = 'Linear fit',linestyle='--')
plt.plot(X_fit,y_quad_fit,label = 'Quadratic fit')
plt.xlabel('Explanatory variable')
plt.ylabel('predicted or known target values')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

y_lin_pred=lr.predict(X)
y_quad_pred = pr.predict(X_quad)
print('Training MSe linear: %.3f, quadratic:%.3f' %(mean_squared_error(y,y_lin_pred),
                                                    mean_squared_error(y,y_quad_pred)))

print('Training R^2 linear: %.3f, quadratic: %.3f'%(
    r2_score(y,y_lin_pred),r2_score(y,y_quad_pred)))

X=df[['LSTAT']].values
y=df['MEDV'].values

regr=LinearRegression()
#creat quadratic features
quadratic = PolynomialFeatures(degree = 2)
cubic = PolynomialFeatures(degree = 3)
X_quad = quadratic.fit_transform(X)
X_cubic = cubic.fit_transform(X)

#fit features
X_fit = np.arange(X.min(),X.max(),1)[:,np.newaxis]

regr = regr.fit(X,y)
y_lin_fit=regr.predict(X_fit)
linear_r2 = r2_score(y,regr.predict(X))

regr = regr.fit(X_quad,y)
y_quad_fit = regr.predict(quadratic.fit_transform(X_fit))
quadratic_r2 = r2_score(y,regr.predict(X_quad))

regr = regr.fit(X_cubic,y)
y_cubic_fit = regr.predict(cubic.fit_transform(X_fit))
cubic_r2 = r2_score(y,regr.predict(X_cubic))

#plot results
plt.scatter(X,y,label='Training points', color ='lightgray')
plt.plot(X_fit,y_lin_fit,
         label = 'Linear (d=1), $R^2=%.2f$' %linear_r2,
         color = 'blue',
         lw = 2,
         linestyle =':')

plt.plot(X_fit, y_quad_fit,
         label = 'Quadratic (d=2), $R^2=%.2f$'% quadratic_r2,
         color = 'red',
         lw=2,
         linestyle ='-')

plt.plot(X_fit,y_cubic_fit,
         label='Cubic (d=3), $R^2=%.2f$' %cubic_r2,
         color ='green',
         lw=2,
         linestyle='--')

plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('Price in $1000s [MEDV]')
plt.legend(loc='best')
plt.show()


#transform features
X_log = np.log(X)
y_sqrt = np.sqrt(y)

#fit features
X_fit = np.arange(X_log.min()-1,
                  X_log.max()+1,1)[:, np.newaxis]
regr = regr.fit(X_log,y_sqrt)
y_lin_fit = regr.predict(X_fit)
linear_r2 = r2_score(y_sqrt,regr.predict(X_log))

# plot results
plt.scatter(X_log,y_sqrt,
            label = 'Training points',
            color ='lightgray')
plt.plot(X_fit,y_lin_fit,
         label = 'Linear (d=1), $R^2=%.2f$' % linear_r2,
         color = 'blue',
         lw = 2)

plt.xlabel('log(% lower status of the population [LSTAT])')
plt.ylabel('$\sqrt{Price \; in \; \$1000s \; [MEDV]}$')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()

from sklearn.tree import DecisionTreeRegressor
X=df[['LSTAT']].values
y = df['MEDV'].values
tree=DecisionTreeRegressor(max_depth=3)
tree.fit(X,y)
sort_idx=X.flatten().argsort()
lin_regplot(X[sort_idx],y[sort_idx],tree)
plt.xlabel('% lower status of the population [LSTAT]')
plt.ylabel('price in $1000s[MEDV]')
plt.show()

X= df.iloc[:,:-1].values
y = df['MEDV'].values
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size = 0.4,
                                                 random_state=1)

from sklearn.ensemble import RandomForestRegressor
forest = RandomForestRegressor(n_estimators = 1000,
                               criterion ='mse',
                               random_state = 1,
                               n_jobs = -1)
forest.fit(X_train,y_train)
y_train_predict = forest.predict(X_train)
y_test_predict = forest.predict(X_test)

print('MSE train: %.3f, test: %.3f'%(
      mean_squared_error(y_train,y_train_predict),
      mean_squared_error(y_test,y_test_predict)))
print (' R^2 train : %.3f, test: %.3f' %(
    r2_score(y_train,y_train_predict),
    r2_score(y_test,y_test_predict)))

#risidual errors
plt.scatter(y_train_predict,
            y_train_predict-y_train,
            c='steelblue',
            edgecolor='white',
            marker ='o',
            s=35,
            alpha =0.9,
            label = 'Training data')
plt.scatter(y_test_predict,
            y_test_predict-y_test,
            c='limegreen',
            edgecolor='white',
            marker = 'o',
            s=35,
            alpha = .9,
            label = 'Test data')

plt.xlabel('Predicted values')
plt.ylabel ('Residuals')
plt.legend(loc ='best')
plt.hlines(y=0, xmin=-10, xmax=50, lw=2, color='black')
plt.xlim([-10,50])
plt.tight_layout()
plt.show()