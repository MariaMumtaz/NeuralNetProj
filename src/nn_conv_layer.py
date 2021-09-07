import numpy as np

from nn_layer import NN_Layer
from nn_conv_unit import Conv2DUnit

from config import LEARNING_RATE, SEED

## TODO: Complete Con 2D unit
class Conv2DLayer(NN_Layer):
    
    def __init__(self, num_in, num_out, activation_fn): # num Input , num_output
        
        self.ins  = num_in
        self.outs = num_out
        self.nn_unit = Conv2DUnit()
        
    ## TODO:        
    def forward(self, X):        
        return X
        
    def backward(self, dA, cache):        
        return dA, 1.0, 1.0
    
    def update(self, dW, dB, lr=0.01):
        pass
        
    def init(self):
        pass
        