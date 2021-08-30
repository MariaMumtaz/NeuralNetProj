#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np


### Loss L1
def L1(yhat, y):
    
    loss = np.sum(abs(y-yhat))
    
    return loss


### Loss L2
def L2(yhat, y):
    
    loss = np.dot( (y-yhat),(y-yhat) )
    
    return loss


### Loss Mean Square Error
def mse(yhat, y):
    
    loss = np.mean(np.power(y-yhat, 2))
    
    return loss


def mse_prime(yhat, y):
    
    loss_prime = (2 * (y-yhat)) /y.size
    
    return loss_prime


### Loss cross entopy cost
def compute_cross_entropy_cost(yhat, y):
    
    m = y.shape[1]
    
    cost = np.squeeze( (1./m) * (-np.dot(y,np.log(yhat).T) - np.dot(1-y, np.log(1-yhat).T)) )
    
    return cost




