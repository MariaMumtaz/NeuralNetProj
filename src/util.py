#!/usr/bin/env python
# coding: utf-8

#setting up path for sys, which is in the same directory
import sys
sys.path.insert(0, f'.')

#setting up path for config, residing in src
import config
sys.path.insert(0, f'{config.ROOT_PATH}src')

import numpy as np
from activation import identity, identity_prime, sigmoid ,sigmoid_prime, relu, relu_prime, tanh, tanh_prime, softmax ,softmax_prime
from sklearn.metrics import roc_curve, auc, roc_auc_score #for convolution layer, not implemented yet

np.random.seed(config.SEED)

activation_map = {
        'Identity':(identity,identity_prime),
        'Sigmoid' :(sigmoid ,sigmoid_prime ),
        'Relu'    :(relu    ,relu_prime    ),
        'Tanh'    :(tanh    ,tanh_prime    ),
        'Softmax' :(softmax ,softmax_prime )
    }

#convolution, not implemented yet
def calculate_metric(scores, truth):
    """
    Given arrays for prediction and ground truth for each class,
    calculate area under the curve [Macro] for each class and combined
    to be used in model evaluation
    """
    AUROCs = []
    for i in range(NUM_CLASSES):
        AUROCs.append(roc_auc_score(truth[:, i], scores[:, i]))
    macro_auroc = np.mean(np.array(AUROCs))
    return macro_auroc, AUROCs

def conv_output_volume(W, F, S, P):
    """
    Given the input volume size $W$, the kernel/filter size $F$,
    the stride $S$, and the amount of zero padding $P$ used on the border,
    calculate the output volume size.
    """
    return int((W - F + 2 * P) / S) + 1

def maxpool_output_volume(W, F, S):
    """
    Given the input volume size $W$, the kernel/filter size $F$,
    the stride $S$, and the amount of zero padding $P$ used on the border,
    calculate the output volume size.
    """
    return int(math.ceil((W - F + 1) / S))
    

# Tools for handling model cache
import pickle
import py7zr
import os
import time


def dump_model(model, name):
    with open(f'model/{name}_fc5.pickle', 'wb') as handle:
        pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with py7zr.SevenZipFile(f'model/{name}_fc5.pickle.7z', "w") as zip:
        zip.write(f'model/{name}_fc5.pickle')
        os.remove(f'model/{name}_fc5.pickle')
        
def load_model(name):
    if os.path.exists(f'model/{name}_fc5.pickle.7z'):
        with py7zr.SevenZipFile(f'model/{name}_fc5.pickle.7z', "r") as zip:
            zip.extractall()
    with open(f'model/{name}_fc5.pickle', 'rb') as handle:
        return pickle.load(handle)
