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


from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

iris = datasets.load_iris()
X,y = iris.data[50:,[1,2]],iris.target[50:]

le = LabelEncoder()
y = le.fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.5,
                                                 stratify = y,
                                                 random_state = 1)

from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import numpy as np

clf1 = LogisticRegression(penalty='l2',solver='lbfgs',C=0.001,random_state=1)
clf2 = DecisionTreeClassifier(max_depth = 1,criterion='entropy',random_state = 0)
clf3 = KNeighborsClassifier(n_neighbors=1,p=2,metric='minkowski')

pip1 = Pipeline([['sc',StandardScaler()],
                ['clf',clf1]])
pip3 = Pipeline([['sc',StandardScaler()],
                ['clf',clf3]])

clf_labels = ['Logistic regression','Decision Tree','KNN']
print('10-fold cross validation:\n')
for clf,label in zip([pip1,clf2,pip3],clf_labels):
    scores = cross_val_score(estimator = clf,
                             X=X_train,
                             y=y_train,
                             cv = 10,
                             scoring='roc_auc')
    print("ROC AUC:%0.2f (+/- %.2f)[%s]"%(scores.mean(),scores.std(),label))
    
from MajorityVoteClassifier import MajorityVoteClassifier

mv_clf = MajorityVoteClassifier(classifiers = [pip1,clf2,pip3])
clf_labels+=['Majority voting']
all_clf = [pip1,clf2,pip3,mv_clf]
for clf,label in zip(all_clf,clf_labels):
    scores = cross_val_score(estimator = clf,
                             X=X_train,
                             y=y_train,
                             cv = 10,
                             scoring = 'roc_auc')
    print("ROC AUC : %.2f(+/-%.2f [%s])" %(scores.mean(),scores.std(),label))
    
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
color = ['black','orange','blue','green']
linestyles = [':','--','-.','-']
for clf,label,clr,ls in zip(all_clf,clf_labels,color,linestyles):
    # assuming the label of the positive class is 1
    y_pred = clf.fit(X_train,y_train).predict_proba(X_test)[:,1]
    fpr,tpr,threshold = roc_curve(y_true=y_test,y_score=y_pred)
    roc_auc = auc(x=fpr,y=tpr)
    plt.plot(fpr,tpr,color=clr,linestyle=ls,
             label = '%s (auc=%0.2f)'%(label,roc_auc))
plt.legend(loc='best')
plt.plot([0,1],[0,1],
         linestyle ='--',color = 'grey',linewidth = 2)
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.grid(alpha = 0.5)
plt.xlabel('False positive rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.show()

sc = StandardScaler()
X_train_std = sc.fit_transform(X_train)
from itertools import product
x_min = X_train_std[:,0].min()-1
x_max = X_train_std[:,0].max()+1
y_min = X_train_std[:,1].min()-1
y_max = X_train_std[:,1].max()+1
xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))
f,axarr = plt.subplots(nrows=2,ncols=2,
                       sharex='col',
                       sharey='row',
                       figsize=(7,5))
for idx,clf,tt in zip (product([0,1],[0,1]),
                       all_clf,clf_labels):
    clf.fit(X_train_std,y_train)
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx[0],idx[1]].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx[0],idx[1]].scatter(X_train_std[y_train==0,0],
                                 X_train_std[y_train==0,1],
                                 c='blue',
                                 marker = '^',
                                 s=50)
    axarr[idx[0],idx[1]].scatter(X_train_std[y_train==1,0],
                                 X_train_std[y_train==1,1],
                                 c='green',
                                 marker = 'o',
                                 s=50)
    axarr[idx[0],idx[1]].set_title(tt)
plt.text(-3.5,-5.,s='Sepalwidth [standardized]',
         ha='center',va='center',fontsize=12) 
plt.text(-12.5,4.5,
         s='Petal length[tsandardized]',
         ha='center',va='center',fontsize = 12,rotation =90)
plt.show()   

mv_clf.get_params()


# grid search parameters of Logistic regression and decision tree
from sklearn.model_selection import GridSearchCV
params = {'decisiontreeclassifier__max_depth':[1,2],
          'pipeline-1__clf__C':[0.001,0.01,1,10,100]}
grid= GridSearchCV(estimator=mv_clf,
                   param_grid=params,
                   cv=10,
                   iid=False,
                   scoring = 'roc_auc')
grid.fit(X_train,y_train)

for r,_ in enumerate(grid.cv_results_['mean_test_score']):
    print("%.3f+/-%.2f %r"
          %(grid.cv_results_['mean_test_score'][r],
            grid.cv_results_['std_test_score'][r],
            grid.cv_results_['params'][r]))
    
print ('Best parameters: %s'%grid.best_params_)
print('Accuracy: %.2f'%grid.best_score_)

from sklearn.ensemble import StackingClassifier
estimators = [('dt',clf2),('kn',pip3)]
clf4 = StackingClassifier(estimators = estimators,final_estimator=pip1)
clf4.fit(X_train,y_train).score(X_test,y_test)

clf4.get_params()

# bagging
import pandas as pd
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/'
                      'machine-learning-databases/wine/wine.data',
                      header = None)
df_wine.columns = ['Class label','Alcohol',
                   'Malic Acid','Ash',
                   'Alcalinity of ash','Magnesium',
                   'Total Phenoles','Flavanoids',
                   'Nonflaveoid phenold',
                   'Prantocyanins',
                   'color intensity',
                   'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

#drop 1 class 
df_wine = df_wine[df_wine['Class label']!=1]
y = df_wine['Class label'].values
X= df_wine[['Alcohol','OD280/OD315 of diluted wines']].values
le = LabelEncoder()
le.fit(y)
y = le.transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state = 1,
                                                 test_size=0.2,
                                                 stratify = y,
                                                )

from sklearn.ensemble import BaggingClassifier
tree = DecisionTreeClassifier(criterion = 'entropy',
                              random_state=1,
                              max_depth=None)
bag = BaggingClassifier(base_estimator=tree,
                        n_estimators=500,
                        max_samples=1.0,
                        max_features=1.0,
                        bootstrap=True,
                        bootstrap_features=False,
                        n_jobs = 1,
                        random_state = 1)

from sklearn.metrics import accuracy_score
tree.fit(X_train,y_train)
y_train_pred = tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train,y_train_pred)
tree_test = accuracy_score(y_test,y_test_pred)
print('Bagging train/test accuracies %.3f/%.3f'%(tree_train,tree_test))

bag.fit(X_train,y_train)
y_train_pred = bag.predict(X_train)
y_test_pred = bag.predict(X_test)
bag_train = accuracy_score(y_train,y_train_pred)
bag_test = accuracy_score(y_test,y_test_pred)
print('Bagging train/ test accuracy %.3f/%.3f'%(bag_train,bag_test))

#decision regions
x_min = X_train[:,0].min()-1
x_max = X_train[:,0].max()+1
y_min = X_train[:,1].min()-1
y_max = X_train[:,1].max()+1

xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))
f,axarr = plt.subplots(nrows = 1,
                       ncols=2,
                       sharex = 'col',
                       sharey = 'row',
                       figsize = (8,3))
for idx,clf,tt in zip([0,1],
                      [tree,bag],
                      ['Decision Tree','Bagging']):
    clf.fit(X_train,y_train)
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0,0],
                       X_train[y_train==0,1],
                       c='blue',marker = '^')
    axarr[idx].scatter(X_train[y_train==1,0],
                       X_train[y_train==1,1],
                       c='green',marker ='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol',fontsize=12)
plt.tight_layout()
plt.text(0,-0.2,
         s='OD280/OD315 of diluted wines',
         ha = 'center',
         va = 'center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()

# adaboost
from sklearn.ensemble import AdaBoostClassifier
tree = DecisionTreeClassifier(criterion ='entropy',
                              random_state =1,
                              max_depth = 1)
ada = AdaBoostClassifier(base_estimator=tree,
                         n_estimators = 500,
                         learning_rate=0.1,
                         random_state=1)

tree =tree.fit(X_train,y_train)
y_train_pred=tree.predict(X_train)
y_test_pred = tree.predict(X_test)
tree_train = accuracy_score(y_train,y_train_pred)
tree_test = accuracy_score(y_test,y_test_pred)
print('Decision tree Accuracy : %.3f/%.3f'%(tree_train,tree_test))

ada = ada.fit(X_train,y_train)
y_train_pred = ada.predict(X_train)
y_test_pred = ada.predict(X_test)
ada_train = accuracy_score(y_train,y_train_pred)
ada_test = accuracy_score(y_test,y_test_pred)
print('AdaBoost train/test accuracy : %.3f/%.3f'%(ada_train,ada_test))

#decision regions
x_min = X_train[:,0].min()-1
x_max = X_train[:,0].max()+1
y_min = X_train[:,1].min()-1
y_max = X_train[:,1].max()+1

xx,yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                    np.arange(y_min,y_max,0.1))
f,axarr = plt.subplots(nrows = 1,
                       ncols=2,
                       sharex = 'col',
                       sharey = 'row',
                       figsize = (8,3))
for idx,clf,tt in zip([0,1],
                      [tree,ada],
                      ['Decision Tree','AdaBoost']):
    clf.fit(X_train,y_train)
    Z = clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    axarr[idx].contourf(xx,yy,Z,alpha=0.3)
    axarr[idx].scatter(X_train[y_train==0,0],
                       X_train[y_train==0,1],
                       c='blue',marker = '^')
    axarr[idx].scatter(X_train[y_train==1,0],
                       X_train[y_train==1,1],
                       c='green',marker ='o')
    axarr[idx].set_title(tt)

axarr[0].set_ylabel('Alcohol',fontsize=12)
plt.tight_layout()
plt.text(0,-0.2,
         s='OD280/OD315 of diluted wines',
         ha = 'center',
         va = 'center',
         fontsize=12,
         transform=axarr[1].transAxes)
plt.show()