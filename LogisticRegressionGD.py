# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 11:04:57 2020

@author: Sima Soltani
"""

import numpy as np
class LogisticsRegressionGD(object):
    """Logistic regression classifier using gradient descent
    
    Parmeters
    
    eta:float
     learning rate (between 0.0 and 1.0)
    n_iter : int
     Passes over the training dataset
    random_state : int
      Random nnumber generator seed for random weight initialization 
    
    Attributes
    ----------
    w_ 1d-array
        Weights after fitting
    cost_ : list
        Sum-of-squares cost function value in each epoch
    
    """
    
    def __init__(self, eta = 0.01, n_iter =50, random_state=1) :
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        
    def fit (self, X, y):
        """
        

        Parameters
        ----------
        X : {array-like},shape =[n_examples,n-features]
            
            Traininng vectors, where n-examples
            is number of examples 
            and n-features is the number of features.
        y : array-like,shape = [n_examples]
            target values.

        Returns
        -------
        self:object.

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0,scale=0.01,
                              size = 1+X.shape[1])
        self.cost_ = []
        
        for i in range (self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y-output)
            self.w_[1:]+= self.eta * X.T.dot(errors)
            self.w_[0] += self.eta *errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self
    
    def net_input(self,X):
        """ Calculate net input"""
        return np.dot(X,self.w_[1:])+self.w_[0]
    
    def activation(self,z):
        """ Compute logistic sigmoid activation """
        return 1./(1. +np.exp(-np.clip(z,-250,250)))
    
    def predict(self,X):
        """ Return class label after unit step"""
        
        return np.where(self.activation(self.net_input(X))>=0.5,1,0)
        