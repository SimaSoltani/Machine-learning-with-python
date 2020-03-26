# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 22:31:22 2020

@author: Sima Soltani
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 20:51:23 2020

@author: Sima Soltani
"""

import numpy as np
class AdalineSGD(object):
    """ADAptive LInear NEuron classifier
    
    Parmeters
    
    eta:float
     learning rate (between 0.0 and 1.0)
    n_iter : int
     Passes over the training dataset
    random_state : int
      Random nnumber generator seed for random weight initialization 
    shuffle: bool (default : True)
     shiffles training data every epoch if True to prevent cycle
    Attributes
    ----------
    w_ 1d-array
        Weights after fitting
    cost_ : list
        Sum-of-squares cost function value in each epoch
    
    """
    
    def __init__(self, eta = 0.01, n_iter =50, random_state=1, shuffle=True) :
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle
        
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
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        
        for i in range (self.n_iter):
            if self.shuffle:
                X,y =self._shuffle(X,y)
            cost = []
            for xi,target in zip(X,y):
                cost.append(self._update_weights(xi,target))
            avg_cost = sum(cost)/len(y)
            self.cost_.append(avg_cost)
            
        return self
    
    
    def partial_fit(self,X,y):
        """fit training data without reinitializing the weights"""
        if not self.w_initialized:
            self.w_initialize_weights(X.shape[1])
        if y.ravel().shape[0]>1:
            for xi,target in zip (X,y):
                self._update_weights(xi,target)
        else:
            self._update_weights(X,y)
        return self
    def _shuffle(self,X,y):
        """shuffle training data"""
        r=self.rgen.permutation(len(y))
        return X[r],y[r]
    
    def _initialize_weights(self,m):
        """initialize weights to small random numbers"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc = 0.0,scale =0.01,size = 1+m)
        self.w_initialized = True
 
    
    def _update_weights(self,xi,target):
        """Apply Adaline learning rule to update the weighst"""
        output = self.activation(self.net_input(xi))
        error=target - output
        self.w_[1:] += self.eta*xi.dot(error)
        self.w_[0]+= self.eta*error
        cost = 0.5 * error**2
        return cost
    def net_input(self,X):
        """ Calculate net input"""
        return np.dot(X,self.w_[1:])+self.w_[0]
    
    def activation(self,X):
        """ Compute linear activation """
        return X
    
    def predict(self,X):
        """ Return class label after unit step"""
        return np.where(self.activation(self.net_input(X))>=0.0,1,-1)
        