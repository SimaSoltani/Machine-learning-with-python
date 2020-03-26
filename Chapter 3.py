# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 09:28:16 2020

@author: Sima Soltani
"""


from sklearn import datasets
import numpy as np
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

def plot_decision_regions (X,y,classifier,test_idx =None,resolution  = 0.02):
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
    
    for idx,cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl,0],y=X[y==cl,1],
                    alpha = 0.8, c=colors[idx],
                    marker = markers[idx],label = cl,
                    edgecolor='black')
        #highlight test examples
    if test_idx:
        #plot all examples
        X_test,y_test = X[test_idx,:],y[test_idx]
        
        plt.scatter(X_test[:,0],X_test[:,1],
                    c='',edgecolor='black',alpha =1.0,
                    linewidth=1, marker='o',
                    s=100, label = 'test set')

        
iris = datasets.load_iris()
X = iris.data[:,[2,3]]
y = iris.target

print('Class labels:', np.unique(y))

from sklearn.model_selection import train_test_split
#using a fix random_state ensurs that our results are reproducble. 
#stratify means that the train-test-split method returs training and test subsets
# that have the same proportions of class labelsas the input set  
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3, 
                                                 random_state = 1, stratify =y)

#using bincount function, which counts the number of occurance of each value in
# an array, to verify that this is indeed the case

print('Labels counts in y:', np.bincount(y))
print('Labels counts in train', np.bincount(y_train))


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)


from sklearn.linear_model import Perceptron

ppn = Perceptron(eta0 = 0.01,random_state = 1)
ppn.fit(X_train_std,y_train)
y_pred = ppn.predict(X_test_std)
print('Misclassified example: %d' %(y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_pred))
print(ppn.score(X_test_std,y_test))

X_combined_std = np.vstack ((X_train_std,X_test_std))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X= X_combined_std, y=y_combined,
                      classifier = ppn,
                      test_idx =range(105,150))

plt.xlabel('petal length [Standardized]')
plt.ylabel('petal width [standardized')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

from LogisticRegressionGD import LogisticsRegressionGD
X_train_01_subset = X_train[(y_train == 0)|(y_train==1)]
y_train_01_subset = y_train[(y_train ==0)| (y_train ==1)]

lrgd = LogisticsRegressionGD(eta=0.5,n_iter=1000,random_state=1)
lrgd.fit(X_train_01_subset,y_train_01_subset)

plot_decision_regions(X=X_train_01_subset,y=y_train_01_subset,classifier = lrgd)

plt.xlabel('petal length[standardised]')
plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=100.0,random_state=1,solver = 'lbfgs',
                        multi_class= 'ovr')
lr.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,
                      classifier=lr,
                      test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
lr.score(X_test_std,y_test)
lr.predict_proba(X_test_std)
lr.predict(X_test_std)


from sklearn.svm import SVC
svm = SVC(C=1.0,kernel='linear',random_state=1)
svm.fit(X_train_std,y_train)
plot_decision_regions(X=X_combined_std,y=y_combined,
                      classifier=svm,
                      test_idx=range(105,150))
plt.xlabel('petal length[standardised]')
plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


### Kernel SVM
#create the synthetic data

np.random.seed(1)
X_xor= np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:,0]>0,
                       X_xor[:,1]>0)
y_xor = np.where(y_xor,1,-1)
plt.scatter(X_xor[y_xor==1,0],
            X_xor[y_xor==1,1],
            c='b',marker='x',
            label='1')
plt.scatter(X_xor[y_xor==-1,0],
            X_xor[y_xor==-1,1],
            c='r',marker='s',
            label='-1')
plt.xlim([-3,3])
plt.ylim([-3,3])
plt.legend(loc='best')
plt.tight_layout()
plt.show()

svm = SVC(C=10.0,kernel='rbf',random_state=1,gamma=0.10)
svm.fit(X_xor,y_xor)
plot_decision_regions(X=X_xor,y=y_xor,
                      classifier=svm)
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

svm = SVC(C=1.0,kernel='rbf',random_state=1,gamma=100)#change gamma between 0.1,0.2,100
svm.fit(X_train_std,y_train)
plot_decision_regions(X=X_combined_std,y=y_combined,
                      classifier=svm,
                      test_idx=range(105,150))

plt.xlabel('petal length[standardised]')
plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

# Decision tree
from sklearn.tree import DecisionTreeClassifier
tree_model = DecisionTreeClassifier(criterion='gini',
                                    max_depth=4,
                                    random_state=1)
tree_model.fit(X_train,y_train)
X_combined=np.vstack((X_train,X_test))
y_combined = np.hstack((y_train,y_test))
plot_decision_regions(X_combined,y_combined,
                      classifier=tree_model,
                      test_idx=range(105,150))
plt.xlabel('petal length[cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#visualize decision tree
from sklearn import tree
tree.plot_tree(tree_model)
plt.show()

#RandomForest
from sklearn.ensemble import RandomForestClassifier
forest=RandomForestClassifier(criterion = 'gini',
                              n_estimators=25,
                              random_state=1,
                              n_jobs=2)
forest.fit(X_train,y_train)
plot_decision_regions(X_combined,y_combined,
                      classifier=forest,
                      test_idx=range(105,150))

plt.xlabel('petal length[cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()


#KNN
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=5,
                         p=2,
                         metric='minkowski')
knn.fit(X_train_std,y_train)
plot_decision_regions(X_combined_std,y_combined,
                      classifier=knn,
                      test_idx=range(105,150))
plt.xlabel('petal length[standardised]')
plt.ylabel('petal width [standardized]')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show()
