# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 14:03:54 2020

@author: Sima Soltani
"""
import pandas as pd
import numpy as np

df_wine=pd.read_csv('https://archive.ics.uci.edu/'
                    'ml/machine-learning-databases/'
                    'wine/wine.data',header=None)


df_wine.columns = ['Class label','Alcohol',
                   'Malic Acid','Ash',
                   'Alcalinity of ash','Magnesium',
                   'Total Phenoles','Flavanoids',
                   'Nonflaveoid phenold',
                   'Prantocyanins',
                   'color intensity',
                   'Hue', 'OD280/OD315 of diluted wines',
                   'Proline']

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X,y = df_wine.iloc[:,1:].values,df_wine.iloc[:,0].values
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size=0.3,
                                                  stratify=y,
                                                  random_state=0)
sc=StandardScaler()
X_train_std = sc.fit_transform(X_train)
X_test_std = sc.transform(X_test)

# calculate the covariance of the features
cov_mat =np.cov(X_train_std.T)
# calculate the eign_values and eigen vectors. We can use eigh fuction for symetric metrices which returns real values
eigen_vals,eigen_vecs = np.linalg.eig(cov_mat)

#plot the varience explained ratio
tot = sum(eigen_vals)
var_exp = [(i/tot) for i in sorted(eigen_vals, reverse = True)]

cum_var_exp = np.cumsum(var_exp)
import matplotlib.pylab as plt
plt.bar(range(1,14),var_exp,alpha =0.5, align = 'center',
         label = 'Individual explained variance')

plt.step(range(1,14), cum_var_exp,where ='mid',
         label = 'Cumulative explained variance ')
plt.ylabel('Explained varieance ratio')
plt.xlabel('Principal component index')
plt.legend(loc = 'best')
plt.tight_layout()
plt.show()


# make a list of (eigenvalue, eigenvector) tuples

eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i])
                for i in range (len(eigen_vals))]
#sort the (eigenvalue, eigenvector) tuples from high to low
eigen_pairs.sort(key=lambda k:k[0],reverse=True)

# select the two top eigen pairs that have 60 percent of the variance
w = np.hstack((eigen_pairs[0][1][:,np.newaxis],
               eigen_pairs[1][1][:,np.newaxis]))

# transform X_train using the selected eigen vectors
X_train_pca= X_train_std.dot(w)


# visulaize the data set
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip(np.unique(y_train),colors,markers):
    plt.scatter(X_train_pca[y_train==l,0],
                X_train_pca[y_train==l,1],
                c=c,label=l,marker=m)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# Use scikit-learn for PCA
from matplotlib.colors import ListedColormap
from plot_decision_regions_script import plot_decision_regions 

from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
#initializing the PCA transformer and logistic regression estimator:
pca = PCA(n_components = 2)
lr = LogisticRegression(multi_class='ovr',
                        random_state =1,
                        solver = 'lbfgs')
# dimentionallity reduction :
X_train_pca = pca.fit_transform(X_train_std)
X_test_pca = pca.transform(X_test_std)
#fitting the logistic regression model on the reduced dataset
lr.fit(X_train_pca,y_train)
plot_decision_regions(X_train_pca,y_train,classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

plot_decision_regions(X_test_pca,y_test,classifier = lr)
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

pca= PCA(n_components=None)#all principle components are kept
X_train_pca = pca.fit_transform(X_train_std)
pca.explained_variance_ratio_


# calculate the LDA step by step 
#1- Standardize the data
#2- For eaxh class calculate the d-dimention mean vector
np.set_printoptions(precision = 4)
mean_vecs = []
for label in range (1,4):
    mean_vecs.append(np.mean(
        X_train_std[y_train==label],axis=0))
    print('MV %s: %s\n' %(label,mean_vecs[label-1]))
  
# 3- Construct the between class scatter matrix, Sb, and the within class scatter matrix, sw.
#first within-class scatter matrix 
d = 13 # number of dimentions
s_w = np.zeros((d,d))
for label,mv in zip(range(1,4),mean_vecs):
    class_scatter=np.zeros((d,d))
for row in X_train_std[y_train==label]:
    row,mv = row.reshape(d,1),mv.reshape(d,1)
    class_scatter+=(row-mv).dot((row-mv).T)
    s_w +=class_scatter
print('Within-class scatter matrix: %sx%s' %(
    s_w.shape[0],s_w.shape[1]))

# Distribution of classes
print('Class label distribution : %s'
      % np.bincount(y_train)[1:])

d = 13 # number of dimentions
s_w = np.zeros((d,d))

for label,mv in zip(range(1,4),mean_vecs):
    
    class_scatter=np.cov(X_train_std[y_train ==label].T)
    s_w +=class_scatter
print('Within-class scatter matrix: %sx%s' %(
    s_w.shape[0],s_w.shape[1]))

mean_overall = np.mean(X_train_std,axis = 0)

d = 13
S_B = np.zeros((d,d))

for i,mean_vec in enumerate(mean_vecs):
    n = X_train_std[y_train==i+1,:].shape[0]
    mean_vec = mean_vec.reshape(d,1) #make column vector
    mean_overall=mean_overall.reshape(d,1)
    S_B+=n*(mean_vec-mean_overall).dot((mean_vec-mean_overall).T)
print ('Between class scatter matrix:  %sX%s' % (S_B.shape[0],S_B.shape[1]))



#4- Compute the eigenvectora and corresponding eigenvalues of the matrix S_W*-1.S_B
eigen_vals,eigen_vecs =\
    np.linalg.eig(np.linalg.inv(s_w).dot(S_B))
# 5- sort the eigenvalues in descending order
eigen_pairs = [(np.abs(eigen_vals[i]),eigen_vecs[:,i])
               for i in range(len(eigen_vals))]
eigen_pairs = sorted(eigen_pairs, key = lambda k:k[0],reverse= True)
print('Eigen values in descending order: \n')
for eigen_val in eigen_pairs:
    print(eigen_val[0])


# plot the linear discriminants 
tot = sum(eigen_vals.real)
discr = [(i/tot) for i in sorted(eigen_vals.real,reverse = True)]
cum_discr = np.cumsum(discr)
plt.bar(range(1,14),discr,alpha = 0.5,align = 'center',label = 'Individual "Discrimanatability"')
plt.step(range(1,14),cum_discr, where='mid',
          label = 'Cumulaive "Discriminability"')
plt.ylabel('"Discreminability" ratio')
plt.xlabel('Linear Disriminants')
plt.ylim([-0.1,1.1])
plt.legend(loc='best')
plt.tight_layout()
plt.show()
 # 6- choose the k-eigenvectors that correspond to the k largest eigenvalues 
 #to construct a dxk-dimentional transformation matrix, W; the eigen vectors are the columns of this matrix
 
w = np.hstack((eigen_pairs[0][1][:,np.newaxis].real,
               eigen_pairs[1][1][:,np.newaxis].real))

print('Matrix W:\n',w)
 #7- Project the example onto thenew feature subspace using the transformation matrix W

X_train_lda = X_train_std.dot(w)
colors = ['r','b','g']
markers = ['s','x','o']
for l,c,m in zip (np.unique(y_train),colors,markers):
    plt.scatter(X_train_lda[y_train==l,0],
               X_train_lda[y_train==l,1]*(-1),
               c=c,label=l,marker=m)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
lda = LDA(n_components=2)
X_train_lda = lda.fit_transform(X_train_std,y_train)
lr = LogisticRegression(multi_class='ovr',
                        random_state=1,
                        solver='lbfgs')
lr= lr.fit(X_train_lda,y_train)
plot_decision_regions(X_train_lda,y_train,classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

X_test_lda = lda.transform(X_test_std)
plot_decision_regions(X_test_lda,y_test,classifier=lr)
plt.xlabel('LD 1')
plt.ylabel('LD 2')
plt.legend(loc ='best')
plt.tight_layout()
plt.show()


from RBF_Kernel_PCA import rbf_kernel_pca

from sklearn.datasets import make_moons
X,y=make_moons(n_samples = 100,random_state=123)
plt.scatter(X[y==0,0],X[y==0,1],
            color = 'red',marker = '^',alpha=0.5)
plt.scatter(X[y==1,0],X[y==1,1],
            color = 'blue', marker = 'o', alpha = 0.5)

plt.tight_layout()
plt.show()

#using PCA
from sklearn.decomposition import PCA

scikit_pca = PCA(n_components=2)
X_spca = scikit_pca.fit_transform(X)
fig, ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(X_spca[y==0,0],X_spca[y==0,1],color = 'red',marker = '^',alpha=0.5)
ax[0].scatter(X_spca[y==1,0],X_spca[y==1,1],color = 'blue',marker = 'o',alpha=0.5)

ax[1].scatter(X_spca[y==0,0],np.zeros((50,1))+0.02,color = 'red',marker = '^',alpha=0.5)
ax[1].scatter(X_spca[y==1,0],np.zeros((50,1))-0.02,color = 'blue',marker = 'o',alpha=0.5)



#Using Kernel PCA
X_kpca = rbf_kernel_pca(X,gamma=15,n_components=2)
fig, ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],color = 'red',marker = '^',alpha=0.5)
ax[0].scatter(X_kpca[y==1,0],X_kpca[y==1,1],color = 'blue',marker = 'o',alpha=0.5)

ax[1].scatter(X_kpca[y==0,0],np.zeros((50,1))+0.02,color = 'red',marker = '^',alpha=0.5)
ax[1].scatter(X_kpca[y==1,0],np.zeros((50,1))-0.02,color = 'blue',marker = 'o',alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

from sklearn.datasets import make_circles
X,y = make_circles(n_samples=1000,
                   random_state = 123,
                   factor = 0.2, noise = 0.1)
plt.scatter(X[y==0,0], X[y==0,1],
            color = 'red',marker='^',alpha = 0.5)
plt.scatter(X[y==1,0],X[y==1,1],
            color = 'blue', marker ='o', alpha = 0.5)
plt.tight_layout()
plt.show()

scikit_pca = PCA(n_components=2)
X_spca =scikit_pca.fit_transform(X)
fig,ax = plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(X_spca[y==0,0],X_spca[y==0,1], color = 'red', marker = '^',alpha = 0.5)
ax[0].scatter(X_spca[y==1,0],X_spca[y==1,1], color = 'blue', marker = 'o',alpha = 0.5)

ax[1].scatter(X_spca[y==0,0],np.zeros((500,1))+0.02,color ='red', marker = '^',alpha = 0.5)
ax[1].scatter(X_spca[y==1,0],np.zeros((500,1))-0.02,color='blue', marker ='o', alpha = 0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

X_kpca = rbf_kernel_pca(X,gamma=15,n_components=2)
fig, ax=plt.subplots(nrows=1,ncols=2,figsize=(7,3))
ax[0].scatter(X_kpca[y==0,0],X_kpca[y==0,1],color = 'red',marker = '^',alpha=0.5)
ax[0].scatter(X_kpca[y==1,0],X_kpca[y==1,1],color = 'blue',marker = 'o',alpha=0.5)

ax[1].scatter(X_kpca[y==0,0],np.zeros((500,1))+0.02,color = 'red',marker = '^',alpha=0.5)
ax[1].scatter(X_kpca[y==1,0],np.zeros((500,1))-0.02,color = 'blue',marker = 'o',alpha=0.5)

ax[0].set_xlabel('PC1')
ax[0].set_ylabel('PC2')
ax[1].set_ylim([-1,1])
ax[1].set_yticks([])
ax[1].set_xlabel('PC1')
plt.tight_layout()
plt.show()

#create a new half-moon dataset and project it into one dimentional sub_space 
X,y = make_moons(n_samples=100,random_state = 123)
alphas,lambdas =rbf_kernel_pca(X,gamma=15,n_components = 1)


from RBF_Kernel_PCA import project_x

x_new = X[25]
x_reproje = project_x(x_new,X,gamma = 15, alphas=alphas, lambdas=lambdas)



# using scikitlearn
from sklearn.decomposition import KernelPCA
X,y = make_moons(n_samples=100,random_state = 123)
scikit_kpca = KernelPCA(n_components=2,
                        kernel = 'rbf',
                        gamma = 15)
X_skernpca= scikit_kpca.fit_transform(X)
plt.scatter(X_skernpca[y==0,0], X_skernpca[y==0,1],
            color = 'red',marker='^',alpha = 0.5)
plt.scatter(X_skernpca[y==1,0],X_skernpca[y==1,1],
            color = 'blue', marker ='o', alpha = 0.5)
plt.tight_layout()
plt.show()