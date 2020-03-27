# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:19:42 2020

@author: Sima Soltani
"""

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class MajorityVoteClassifier(BaseEstimator,ClassifierMixin):
    """ A majority vote ensemble classifier
    
    parameters
    ----------
    classifiers: array-like , shape =[n_classifiers]
     Different classififers for the ensembles 
    
    vote: str, {'classlabel','probability'}
     Default:'classlabel'
     if 'classlabel' the prediction is based on the argmax of class labels. 
     Else if 'probability', the argmaxof the sum of probabilities is used 
     to predict the class label (recommended for calibrated classifiers)
     
    weights : arry-like , shape = [n_classifiers]
     optional, default: None
     if a list of 'int' or 'float' values are provided, the classifiesrs 
     are weighted by importance; uses uniform weights if 'weights =None'
    """
    def __init__(self, classifiers, vote = 'classlabel',weights=None):
        self.classifiers = classifiers
        self.named_classifiers={key:value 
                                for key,value in 
                                _name_estimators(classifiers)}
        self.vote = vote
        self.weights = weights
    
    def fit(self,X,y):
        """
        Fit classifiers

        Parameters
        ----------
        X : {array-like,sparse matrix},
        shape =[n_example,n_features]
            Matrix of training examples.
        y : array-like shape =[n_examples]
            Vector of target class labels.

        Returns
        -------
        self: object

        """
        if self.vote not in ('probability','classlabel'):
            raise ValueError("vote must be 'probability' or"
                             "'classlabel' ; got (vote = %r)"%self.vote)
        if self.weights and \
        len(self.weights)!=len(self.classifiers):
            raise ValueError("Number of classifiers and weights"
                             "must be equal; got %d weights,"
                             "%d classifiers")
        # Use LabelEncoder to ensure class labels starts with 0, which is 
        # important for np.argmax call in self.predict
        self.lablenc_ = LabelEncoder()
        self.lablenc_.fit(y)
        self.classes_ = self.lablenc_.classes_
        self.classifiers_=[]
        for clf in self.classifiers:
            fitted_clf = clone(clf).fit(X,self.lablenc_.transform(y))
            self.classifiers_.append(fitted_clf)
        return self
    
    def predict (self,X):
        """
        Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix},
        Shape = [n_examples, n_features]
        Matrix of training examples.
        
        
        Returns
        -------
        maj_vote: array-like, shape=[n_examples]
        Predicted class labels.

        """
        if self.vote == 'probability':
            maj_vote = np.argmax(self.predict_proba(X),axis = 1) 
        else:
            #'classlabel vote
            #colect results from clf.predict calls
            predictions = np.asarray([clf.predict(X) for 
                                      clf in self.classifiers_]).T
            maj_vote = np.apply_along_axis(lambda x: np.argmax
                                           (np.bincount(x,
                                                        weights = self.weights)),
                                           axis = 1,
                                           arr = predictions)
        maj_vote = self.lablenc_.inverse_transform(maj_vote)
        return maj_vote
        
    def predict_proba(self, X):
        """
        Predict class probabilities for X


        Parameters
        ----------
        X :{array-like, sparse matrix}
        shape = [n_examples,n_features]
        Training vectors, where 
        n_examples is the number of examples and
        n)features is the number of features
            

        Returns
        -------
        avg_prob: array-like,
        shape = [n_example, n_classes]
        weighted average probability for class per example

        """
        probas=np.asarray([clf.predict_proba(X)
                           for clf in self.classifiers_])
        avg_proba = np.average(probas,axis = 0,
                               weights = self.weights)
        return avg_proba
    
    def get_params(self, deep = True):
        """
        Get classifier parameter names for GridSearch

        
        """
        if not deep:
            return super(MajorityVoteClassifier,self).get_params(deep = False)
        else:
            out = self.named_classifiers.copy()
            for name, step in self.named_classifiers.items():
                for key,value in step.get_params(
                        deep = True).items():
                    out['%s__%s'%(name,key)] = value
            return out