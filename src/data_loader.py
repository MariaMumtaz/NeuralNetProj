#!/usr/bin/env python
# coding: utf-8

import sys
import config
import numpy as np
import h5py
from config import ROOT_PATH, DATA_DIR, FILENAME

sys.path.insert(0, f'{config.ROOT_PATH}src')

def load_data(FILENAME=FILENAME):
    
    train_set_x, train_set_y, test_set_x, test_set_y, classes = None,None,None,None,None
    
    with h5py.File(f'{ROOT_PATH}{DATA_DIR}train_{FILENAME}.h5', "r") as train_dataset:
        train_set_x = np.array(train_dataset["train_set_x"][:]) # your train set features
        train_set_y = np.array(train_dataset["train_set_y"][:]) # your train set labels
        train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))  
        classes = np.array(train_dataset["list_classes"][:]) # the list of classes        

    with h5py.File(f'{ROOT_PATH}{DATA_DIR}test_{FILENAME}.h5', "r") as test_dataset:
        test_set_x = np.array(test_dataset["test_set_x"][:]) # your test set features
        test_set_y = np.array(test_dataset["test_set_y"][:]) # your test set labels
        test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
        
    return train_set_x, train_set_y, test_set_x, test_set_y, classes

## Cat vs No-Cat
def load_cvnc_data():
    
    return load_data('catvnoncat')

## MINST Hand Written Digits
def load_minst_data():
    
    return load_data('mnist')
    
def normalize_image_data(set_x):
    return set_x.reshape(set_x.shape[0], -1).T/255.
    
def one_hot_data(set_y, classes):
    return np.squeeze(np.eye(len(classes))[set_y.reshape(-1)]).astype(int)
    
    