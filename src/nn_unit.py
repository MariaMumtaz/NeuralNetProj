import numpy as np
from util import activation_map

class NN_Unit:
    def __init__(self, activation_fn='Identity'):
        
        self.n_X = None  # Input_size
        self.n_A = None  # Output_size
        
        self.activation_f = activation_map[activation_fn][0]
        self.activation_b = activation_map[activation_fn][1]
        
    def forward(self, X):
        raise NotImplementedError
        
    def backward(self, dA, cache):
        raise NotImplementedError
        
    def init(self):
        raise NotImplementedError