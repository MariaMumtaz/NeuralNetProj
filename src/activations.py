#!/usr/bin/env python
# coding: utf-8

import math
import numpy as np


### Sigmoid
def sigmoid(Z):
    
    A = 1.0/(1.0 + np.exp(-Z))  # Activation
    C = Z                       # Cache
    
    return A , C 


def sigmoid_prime(dA, C): ## Derivative and Cache
    
    Z = C                 ## Cache
    
    S, _  = sigmoid(dA)
    ds = S * (1-S)
    
    dZ = dA * S
    
    return dZ


### RELU
def relu(Z, alpha=0.00001):
    C = Z                       # Cache
    A = np.maximum(alpha,Z)     # Activation
    
    return A , C 


def relu_prime(dA, C):
    
    Z = C                      ## Cache
    
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    dZ[Z >  0] = 1
    
    return dZ


### tanH
def tanh(Z):
    
    C = Z                       # Cache
    A = np.tanh(Z)              # Activation
    
    return A , C


def tanh_prime(dA, C):
    
    Z = C                      ## Cache
    
    dZ = 1-np.tanh(dA)**2
    
    return dZ

### Identity
def identity(Z):
    
    C = Z                       # Cache
    A = Z                       # Activation
    
    return A , C


def tanh_prime(dA, C):
    
    Z = C                      ## Cache
    
    dZ = dA
    
    return dZ


### SoftMax
def softmax(Z, axis=1):
    
    A = np.exp(x) / np.sum(np.exp(x), axis = axis, keepdims = True)
    C = Z
    
    return A , C

