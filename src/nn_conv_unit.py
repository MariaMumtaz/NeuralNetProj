import numpy as np

from nn_unit import NN_Unit

## TODO: Complete Con 2D unit
class Conv2DUnit(NN_Unit):
    
    def __init__(self, n_X, n_A, activation_fn): # num Input , num_output
        super(FullyConnectedUnit, self).__init__(activation_fn)
        pass        
        
    ## TODO:        
    def forward(self, X):
        return X , (X (X, 1.0, 1.0))

    ## TODO:
    def backward(self, dA, cache):
        return dA, 1.0, 1.0
        
    def init(self):
        pass
        
    