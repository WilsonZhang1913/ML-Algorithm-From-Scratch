#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 22 11:09:16 2021

@author: wilsonzhang
"""
import numpy as np 
from numpy import linalg
    
class Linear_Regression:
    """Linear Regression 

    Example usage:
        > clf = Linear_Regression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-7,
                 theta_0=None, optimizer = 'OLS', verbose=True,
                 intercept = True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.optimizer = optimizer 
        self.verbose = verbose
        self.intercept = intercept
        
    def fit(self, X, y):
        
        if self.intercept:
            X = self.add_intercept(X)
        
        y = y.reshape(X.shape[0], -1)
        
        if self.optimizer =='OLS':
            coef_, _residues, rank_, singular_ = linalg.lstsq(X, y)
            self.theta = np.ravel(coef_)
        
        elif self.optimizer == 'BGD':
            
            counter = 0
            
            theta_diff = np.matrix(np.ones(X.shape[1])).T
            
            self.theta = np.zeros(shape=[X.shape[1],1])

            while np.linalg.norm(theta_diff) > self.eps and counter <=self.max_iter:
            
                counter += 1
            
                gradient = (1/X.shape[0])*self.gradient(X, y)
                theta_diff = self.step_size*gradient 
            
                self.theta -= theta_diff
                #print(gradient)
                
            #print(counter, gradient)
                
        elif self.optimizer == 'Newton':
            
            counter = 0
            
            theta_diff = np.matrix(np.ones(X.shape[1])).T
            
            self.theta = np.zeros(shape=[X.shape[1],1])

            while np.linalg.norm(theta_diff) > self.eps and counter <=self.max_iter:
            
                counter += 1
            
                gradient = self.gradient(X, y)
                hessian = self.hessian(X, y)
            
                theta_diff = (linalg.inv(hessian).dot(gradient)) 
            
                self.theta -= theta_diff
            print(counter, gradient)
            
    def predict(self, x):
        
        """Return predicted given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).
        
        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = self.add_intercept(x)
        
        pred_y =np.dot(x,self.theta)
        
        return(np.ravel(pred_y))
            
        
        
    
    def add_intercept(self, x):
        
        """
        Add intercept to matrix x.

        Args:
            x: 2D NumPy array.

        Returns:
            New matrix same as x with 1's in the 0th column.
        """

        new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = x
        return new_x
    
        
    def gradient(self, x, y):
        """
        Calculate gradient of square loss fucntion 1/2(y_hat - y)^2

        Args:
            x: 2D NumPy array.
            y: 2D Numpy array

        Returns:
            the gradient of the loss function 
        """           
        h_x = x.dot(self.theta)
        #print(h_x.shape)
        #print(y.shape)
        #print(h_x-y)
        
        return x.T.dot(h_x -y)
    
    def hessian(self, x, y):
        
        """
        Calculate hessian of square loss fucntion 1/2(y_hat - y)^2

        Args:
            x: 2D NumPy array.
            y: 2D Numpy array

        Returns:
            the hessian of the loss function 
        """     
        return x.T.dot(x)
    

class Logistic_Regression:
    """Linear Regression 

    Example usage:
        > clf = Linear_Regression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-7,
                 theta_0=None, optimizer = 'BGD', verbose=True,
                 intercept = True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.optimizer = optimizer 
        self.verbose = verbose
        self.intercept = intercept
        
    def fit(self, X, y):
        
        if self.intercept:
            X = self.add_intercept(X)
        
        y = y.reshape(X.shape[0], -1)
        
        
        if self.optimizer == 'BGD':
            
            counter = 0
            
            theta_diff = np.matrix(np.ones(X.shape[1])).T
            
            self.theta = np.zeros(shape=[X.shape[1],1])

            while np.linalg.norm(theta_diff) > self.eps and counter <=self.max_iter:
            
                counter += 1
            
                gradient = (1/X.shape[0])*self.gradient(X, y)
                theta_diff = self.step_size*gradient 
            
                self.theta -= theta_diff
                
                #print(gradient)
                
            print(counter, gradient)
                
        elif self.optimizer == 'Newton':
            
            counter = 0
            
            theta_diff = np.matrix(np.ones(X.shape[1])).T
            
            self.theta = np.zeros(shape=[X.shape[1],1])

            while np.linalg.norm(theta_diff) > self.eps and counter <=self.max_iter:
            
                counter += 1
            
                gradient = self.gradient(X, y)
                hessian = self.hessian(X, y)
            
                theta_diff = (linalg.inv(hessian).dot(gradient)) 
            
                self.theta -= theta_diff
            print(counter, gradient)
            
    def predict(self, x):
        
        """Return predicted given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).
        
        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        x = self.add_intercept(x)
        
        pred_y =1/(1+np.exp(-1*np.dot(x,self.theta)))
        
        
        return(np.ravel(pred_y))
            
        
        
    
    def add_intercept(self, x):
        
        """
        Add intercept to matrix x.

        Args:
            x: 2D NumPy array.

        Returns:
            New matrix same as x with 1's in the 0th column.
        """

        new_x = np.zeros((x.shape[0], x.shape[1] + 1), dtype=x.dtype)
        new_x[:, 0] = 1
        new_x[:, 1:] = x
        return new_x
    
        
    def gradient(self, x, y):
        """
        Calculate gradient of log loss  (binary cost entropy) function 
        
        -(y*log(y_hat) + (1-y)*log(1-y_hat))

        Args:
            x: 2D NumPy array.
            y: 2D Numpy array

        Returns:
            the gradient of the loss function 
        """           
        h_x =1/(1+np.exp(-1*np.dot(x,self.theta)))
        
        return x.T.dot(h_x -y)
    
    def hessian(self, x, y):
        
        """
        Calculate hessian of log loss fucntion -(y*log(y_hat) + (1-y)*log(1-y_hat))
        hessian = (X.T)SX
        where S = diag(y_hat*(1-y_hat))
        Args:
            x: 2D NumPy array.
            y: 2D Numpy array

        Returns:
            the hessian of the loss function 
        """ 
        h_x =np.ravel(1/(1+np.exp(-1*np.dot(x,self.theta))))
        
        S = np.multiply(h_x, (1-h_x))
        S = np.diag(S)
        
        return x.T.dot(S).dot(x)


    
        
            
            
            