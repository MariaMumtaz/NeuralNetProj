import numpy as np

from nn_layer import NN_Layer
from fc_unit import FullyConnectedUnit

from config import LEARNING_RATE, SEED, ROOT_PATH

class FullyConnectedLayer(NN_Layer):
    
    def __init__(self, num_in, num_out, activation_fn): # num Input , num_output
        
        self.ins  = num_in
        self.outs = num_out
        self.nn_unit = FullyConnectedUnit(num_in,num_out,activation_fn)
                
    def forward(self, X):
        
        self.A, (Z_cache, A_cache) = self.nn_unit.forward(X)
        
        return self.A, (Z_cache, A_cache)

    def backward(self, dA, cache):
        
        dA, dW, dB = self.nn_unit.backward(dA, cache)
        
        return dA, dW, dB
    
    def update(self, dW, dB, lr=0.01):
        self.nn_unit.W = self.nn_unit.W - (lr * dW)
        self.nn_unit.B = self.nn_unit.B - (lr * dB)
        
    def init(self):
        self.nn_unit.init()