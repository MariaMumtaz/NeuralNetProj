import numpy as np
import time

from nn_layer import NN_Layer

from config import LEARNING_RATE, SEED, DYNAMIC_EPOCS, MODEL_DIR, EARLY_STOP
from loss import *
from activation import softmax

np.random.seed(SEED)

class NN_Network:

    def __init__(self, l_rate=LEARNING_RATE):
        
        self.layers = []
        self.l_rate = l_rate
        self.costs = []
        
    def add_layer(self, layer):
        num_out, num_in = layer.outs, layer.ins
        self.layers.append(layer)
        layer.init()
                
    def train(self, X, Y, epochs= 100, dynamic_learning=False):
        begin = time.process_time()
        print(f'Training started')
        print(f'Learning rate = {self.l_rate}')
        
        cost = float('+inf')
        start = time.process_time()
        
        for e in range(0,epochs):
            
            A = X
            caches = []
            for idx, l in enumerate(self.layers):
                A, cache = l.forward(A)
                caches.append(cache)
            
            
            #pred = np.expand_dims(np.max(A, axis=0),0)
            #Yhat = np.expand_dims(np.argmax(Y, axis=1),0)
            
            lcost = cost                                           
            cost = np.mean(cross_entropy_cost(A, Y.T))
            #cost = -np.mean(Y * np.log(A.T+ 1e-8))
            
            dA = cross_entropy_cost_prime(A, Y.T)
                
            cnt = len(self.layers)
            #Y = Y.reshape(A.shape)
            
            for i in reversed(range(0, cnt)):
                dA, dW, dB = self.layers[i].backward(dA, caches[i])
                self.layers[i].update(dW, dB, self.l_rate)
            
            if cost != float('+inf') and not (np.isnan(cost)):
                self.costs.append(cost)
            
            if e%100 == 0 and e != 0:
                print(f"Cost at epoch {e} is {cost:<.5f}, Iteration time : {time.process_time() - start:<.3f}")
                start = time.process_time()
                
            if e % DYNAMIC_EPOCS == 0 and e != 0 and dynamic_learning:
                self.l_rate = self.l_rate / 2
                print(f"Learning rate reduced to {self.l_rate}")
            
            # Early Stop
            if lcost - cost <= EARLY_STOP and lcost > cost:
                print(f"No significant improvement in loss, stopping early at epoch {e}")
                break
                
        print(f"Cost at the end of training is {cost:<.5f}, Time elspsed : {time.process_time() - begin:<.3f}")
        
        return self.costs
        
    ## FixMe: Use Num Classes
    def predict(self, X):
        A = X
        for idx, l in enumerate(self.layers):
            A, cache = l.forward(A)
        
        pred = np.expand_dims(np.argmax(A, axis=0),0)
        
        return pred
    
    ## FixMe: Use Num Classes
    def evaluate(self, X, Y):
        Yhat = self.predict(X).astype(int)
        Y    = np.expand_dims(np.argmax(Y, axis=1),0)
        res = np.sum((Yhat.astype(int) == Y)/Y.shape[1])
        print(f"Accuracy: {res:<.3f}")
        return Yhat, Y