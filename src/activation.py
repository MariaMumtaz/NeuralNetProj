#!/usr/bin/env python
# coding: utf-8

import numpy as np


### Sigmoid
def sigmoid(Z):
    
    Z[Z > 709] =  709 #prevent np.exp overflow
    Z[Z <-709] =  0   #prevent np.exp overflow
    
    A = np.where(Z >= 0, 
                    1. / (1. + np.exp(-Z)), 
                    np.exp(Z) / (1. + np.exp(Z))) # Activation
    C = Z                       # Cache
    
    return A , C 


def sigmoid_prime(dA, C): ## Derivative and Cache
    
    Z = C                 ## Cache
    
    S = 1. / (1. +np.exp(-Z))
    ds = S * (1-S)
    
    dZ = dA * ds
    
    return dZ


### RELU
def relu(Z, alpha=0.0000000001):
    C = Z                       # Cache
    A = np.maximum(alpha*Z,Z)     # Activation
    
    return A , C 


def relu_prime(dA, C):
    
    Z = C                      ## Cache
    
    dZ = np.array(dA, copy=True)
    
    dZ[Z <= 0] = 0
    #dZ = dZ * 2
    
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


def identity_prime(dA, C):
    
    Z = C                      ## Cache
    
    dZ = dA
    
    return dZ


### SoftMax
def softmax(Z, axis=0):
    
    C = Z
    expZ = np.exp(Z - np.max(Z))
    A = expZ / expZ.sum(axis = axis, keepdims = True)
    
    return A , C

def softmax_prime(dA, C):
    
    #Z = C                      ## Cache
    
    #s = dA.reshape(-1,1)
    #dZ = np.diagflat(s) - np.dot(s, s.T)
    
    Z = C                      ## Cache
    dZ = dA
    
    return dZ
    
