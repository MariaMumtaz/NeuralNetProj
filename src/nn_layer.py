from config import LEARNING_RATE

class NN_Layer:
    def __init__(self):
        self.nn_unit = None
        
    def forward(self, X):
        raise NotImplementedError
        
    def backward(self, dA, cache):
        raise NotImplementedError
        
    def update(self, grads, lr=LEARNING_RATE):
        raise NotImplementedError
        
    def init(self):
        raise NotImplementedError