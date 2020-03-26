# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 12:21:05 2020

@author: Sima Soltani
"""

from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np

def rbf_kernel_pca (X,gamma,n_components):
    """
    RBF kernel PCA

    Parameters
    ----------
    X : {Numpy narray}, shape = [n_example, n_feature]
        
    gamma : float
        Tuning parameter for RBF kernel.
    n_components : int
        Number of principle components to return.

    Returns
    -------
   alphas{numpy n_array}, shape = [n_examples, K_features]
    projectd dataset
    lambdas: list
    Eigenvalues

    """
    # Calculate pairwise squared Euclidean distances
    # in the MxN dimensional dataset.
    sq_dists = pdist(X, 'sqeuclidean')

    # Convert pairwise distances into a square matrix.
    mat_sq_dists = squareform(sq_dists)

    # Compute the symmetric kernel matrix.
    K = exp(-gamma * mat_sq_dists)

    # Center the kernel matrix.
    N = K.shape[0]
    one_n = np.ones((N, N)) / N
    K = K - one_n.dot(K) - K.dot(one_n) + one_n.dot(K).dot(one_n)

    # Obtaining eigenpairs from the centered kernel matrix
    # scipy.linalg.eigh returns them in ascending order
    eigvals, eigvecs = eigh(K)
    eigvals, eigvecs = eigvals[::-1], eigvecs[:, ::-1]

    # Collect the top k eigenvectors (projected examples)
    alphas= np.column_stack([eigvecs[:,i]
                           for i in range (n_components)])
    
    #colect the corresponding eigenvalues
    lambdas = [eigvals[i] for i in range (n_components)]
    return alphas, lambdas
    
def project_x(x_new, X,gamma,alphas,lambdas):
   pair_dist = np.array([np.sum((x_new -row)**2) for row in X])
   k = np.exp(-gamma*pair_dist)
   return k.dot(alphas/lambdas)