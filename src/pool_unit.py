import numpy as np

from nn_unit import NN_Unit
from config import SEED
np.random.seed(SEED)

class MaxPool2dUnit(NN_Unit):
    
    # n_A is here pool_filter_size
    def __init__(self, n_X, n_Y, n_A, stride=2, activation_fn='Max'): # num Input , num_output
        super(FullyConnectedUnit, self).__init__(activation_fn)
        
        self.n_X = n_X
        self.n_X = n_Y
        self.n_A = n_A
   
   ## TODO     
    def forward(self, X):
        
        pass
    
    def backward(self, dA, cache):
        
        pass
    
    def init(self):
        
        pass
   