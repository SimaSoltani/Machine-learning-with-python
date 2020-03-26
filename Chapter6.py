# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 14:33:52 2020

@author: Sima Soltani
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
df  = pd.read_csv('https://archive.ics.uci.edu/ml/'
                  'machine-learning-databases'
                  '/breast-cancer-wisconsin/wdbc.data',
                  header = None)

# LabelEncoderto transform the class labels from their original string representation to integer
X = df.loc[:,2:].values
y = df.loc[:,1].values
le = LabelEncoder()
y = le.fit_transform(y)
le.classes_

# seperate train test data
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,stratify = y, random_state = 1,test_size = 0.20)


# create a piipeline for scaling transformation and algorithm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
pipe_lr = make_pipeline(StandardScaler(),PCA(n_components = 2), 
                        LogisticRegression(random_state = 1,solver = 'lbfgs'))
pipe_lr.fit(X_train,y_train)
y_pred = pipe_lr.predict(X_test)
print('Test Accuracy : %.3f' %pipe_lr.score(X_test,y_test))


from sklearn.model_selection import StratifiedKFold

kfold = StratifiedKFold(n_splits = 10).split(X_train,y_train)
scores = []
for k , (train,test) in enumerate(kfold):
    pipe_lr.fit(X_train[train],y_train[train])
    score = pipe_lr.score(X_train[test],y_train[test])
    scores.append(score)
    print('Fold : %2d, Class dist.: %s,Acc: %0.3f' %(k+1, np.bincount(y_train[train]),score))
    
    
#use scikitlearn
from sklearn.model_selection import cross_val_score # comment is that it is stratified cv
scores = cross_val_score(estimator=pipe_lr,X=X_train, y= y_train, cv = 10,n_jobs=1)


# learning curve in sklearn

import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve

pipe_lr=make_pipeline(StandardScaler(),
                      LogisticRegression(random_state=1,
                                         solver = 'lbfgs',
                                         penalty = 'l2',
                                         max_iter = 10000))

train_sizes,train_scores,test_scores=learning_curve(estimator=pipe_lr,X=X_train, y=y_train,train_sizes=np.linspace(0.1,1,10),
                                                    cv=10,
                                                    n_jobs=1)
train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(train_sizes,train_mean,color = 'blue',marker = 'o',
         markersize = 5, label = 'Training accuracy')

plt.fill_between(train_sizes,train_mean+train_std,train_mean-train_std, alpha = 0.15, color = 'blue')

plt.plot(train_sizes,test_mean, color = 'green', linestyle='--',
         marker ='s', markersize = 5, label = 'Validation Accuracy')
plt.fill_between(train_sizes,test_mean+test_std,test_mean-test_std, alpha =0.15, color = 'green')
plt.grid()
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.03])
plt.tight_layout()
plt.show()


from sklearn.model_selection import validation_curve
param_range = [0.001,0.01,0.1,1.0,10.0,100.0]
train_scores,test_scores = validation_curve(
    estimator=pipe_lr,
    X=X_train,
    y=y_train,
    param_name ='logisticregression__C',
    param_range=param_range,
    cv=10)

train_mean = np.mean(train_scores,axis=1)
train_std = np.std(train_scores,axis=1)
test_mean=np.mean(test_scores,axis=1)
test_std = np.std(test_scores,axis=1)
plt.plot(param_range,train_mean,color = 'blue',marker = 'o',
         markersize = 5, label = 'Training accuracy')

plt.fill_between(param_range,train_mean+train_std,train_mean-train_std, alpha = 0.15, color = 'blue')

plt.plot(param_range,test_mean, color = 'green', linestyle='--',
         marker ='s', markersize = 5, label = 'Validation Accuracy')
plt.fill_between(param_range,test_mean+test_std,test_mean-test_std, alpha =0.15, color = 'green')
plt.grid()
plt.xscale('log')
plt.xlabel('Number of training examples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8,1.03])
plt.tight_layout()
plt.show()

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


pipe_svc = make_pipeline(StandardScaler(),
                         SVC(random_state = 1))
param_range = [0.0001,0.001,0.01,0.1,1.0,10.0,100.0,1000.0]
param_grid = [{'svc__C':param_range,
               'svc__kernel':['linear']},
              {'svc__C':param_range,
               'svc__gamma':param_range,
               'svc__kernel':['rbf']}]
gs = GridSearchCV(estimator= pipe_svc,
                  param_grid=param_grid,
                  scoring = 'accuracy',
                  cv=10,
                  refit = True,
                  n_jobs = 1)
gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

clf = gs.best_estimator_
clf.fit(X_train,y_train)
print('Test Accuracy : %.3f' %clf.score(X_test,y_test))

#the above 3 line can be replaced with :
print('Test Accuracy: %.3f' %gs.score(X_test,y_test))

#nested cross validation 
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid=param_grid,
                  scoring='accuracy',
                  cv=2)
scores=cross_val_score(gs,X_train,y_train,scoring='accuracy',cv=5)
print('CV accuracy:%.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

#using nested cross validatiobn for both parameter and algorithm selection
from sklearn.tree import DecisionTreeClassifier
gs = GridSearchCV(estimator=DecisionTreeClassifier(random_state=0),
                  param_grid=[{'max_depth':[1,2,3,4,5,6,None]}],
                  scoring = 'accuracy',
                  cv=2)
scores=cross_val_score(gs,X_train,y_train,scoring='accuracy',cv=5)
print('CV accuracy:%.3f +/- %.3f'%(np.mean(scores),np.std(scores)))

from sklearn.metrics import confusion_matrix
pipe_svc.fit(X_train,y_train)
y_predict = pipe_svc.predict(X_test)
confmat = confusion_matrix(y_true = y_test, y_pred=y_predict)
print(confmat)

from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
disp = plot_confusion_matrix(pipe_svc, X_test, y_test,cmap = plt.cm.Blues,display_labels=le.classes_)


from sklearn.metrics import make_scorer,f1_score

c_gamma_range = [0.01,0.1,1.0,10.0]
param_grid =[{'svc__C':c_gamma_range,'svc__kernel':['linear']},
             {'svc__C':c_gamma_range,'svc__kernel':['rbf'],'svc__gamma':c_gamma_range}]
scorer = make_scorer(f1_score,pos_label=0)
gs = GridSearchCV(estimator=pipe_svc,
                  param_grid = param_grid,
                  scoring=scorer,cv=10)
gs = gs.fit(X_train,y_train)
print(gs.best_score_)
print(gs.best_params_)

from sklearn.metrics import roc_curve,auc
from numpy import interp

pipe_lr = make_pipeline(StandardScaler(),
                        PCA(n_components=2),
                        LogisticRegression(random_state = 1,
                                           penalty = 'l2',
                                           solver ='lbfgs',
                                           C=100.0))
X_train2 = X_train[:,[4,14]]
cv = list(StratifiedKFold(n_splits=3).split(X_train2,y_train))

fig = plt.figure(figsize=(7,5))
mean_tpr = 0.0
mean_fpr = np.linspace(0,1,100)
all_tpr = []

for i ,(train,test)in enumerate(cv):
    probas=pipe_lr.fit(X_train2[train],
                       y_train[train]).predict_proba(X_train2[test])
    fpr,tpr,threshold = roc_curve(y_train[test],probas[:,1],pos_label=1)
    mean_tpr +=interp(mean_fpr,fpr,tpr)
    mean_tpr[0]=0.0
    roc_auc=auc(fpr,tpr)
    plt.plot(fpr,
             tpr,
             label = 'ROC fold %d (area = %.2f)'
             %(i+1,roc_auc))
plt.plot([0,1],
         [0,1],
         linestyle='--',
         color=(.6,.6,.6),
         label ='Random guessing')

mean_tpr /=len(cv)
mean_tpr[-1]=1.0
mean_auc = auc(mean_fpr,mean_tpr)
plt.plot(mean_fpr,mean_tpr,'k--',
         label ='Mean ROC (area = %.2f)' %mean_auc,lw=2)

plt.plot([0,0,1],[0,1,1],
         linestyle = ':',
         color = 'black',
         label = 'Perfect performance')

plt.xlim([-.05,1.05])
plt.ylim([-.05,1.05])
plt.xlabel('False Positive rate')
plt.ylabel('True Positive rate')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()


X_imb = np.vstack((X[y==0],X[y==1][:40]))
y_imb = np.hstack((y[y==0],y[y==1][:40]))


#upsampling the minority class
from sklearn.utils import resample
print ('Number of class 1 example before:',
       X_imb[y_imb==1].shape[0])

X_upsampled,y_upsampled = resample(X_imb[y_imb==1],
    y_imb[y_imb==1],
    replace = True,
    n_samples=X_imb[y_imb==0].shape[0],random_state = 123)

print('Number of class 1 examples after :', X_upsampled.shape[0])
