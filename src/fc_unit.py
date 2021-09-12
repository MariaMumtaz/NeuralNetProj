import numpy as np

from nn_unit import NN_Unit
from config import SEED
np.random.seed(SEED)

class FullyConnectedUnit(NN_Unit):
    
    def __init__(self, n_X, n_A, activation_fn): # num Input , num_output
        super(FullyConnectedUnit, self).__init__(activation_fn)
        
        self.n_X = n_X
        self.n_A = n_A
        
        self.W = np.random.randn(n_A, n_X) * 0.01
        self.B = np.zeros((n_A, 1))        
                
    def forward(self, X):
        
        self.X = X
        
        self.Z = self.W.dot(self.X) + self.B
        
        Z_cache = (self.X, self.W, self.B)
        
        self.A, A_cache = self.activation_f(self.Z)
        return self.A, (Z_cache, A_cache)

    def backward(self, dA, cache):
        
        Z_cache, A_cache = cache
        
        dZ = self.activation_b(dA, A_cache)
        
        A , W , B = Z_cache        
        
        m = A.shape[1]
        
        dW = 1./m * np.dot(dZ,A.T)
        dB = 1./m * np.sum(dZ, axis = 1, keepdims = True)
        dA = np.dot(W.T,dZ)
        
        return dA, dW, dB
    
    def init(self):
        
        np.random.seed(SEED)
        self.W = np.random.randn(self.n_A, self.n_X) / np.sqrt(self.n_X)
        self.B = np.zeros((self.n_A, 1))
    