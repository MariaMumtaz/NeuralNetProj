#!/usr/bin/env python
# coding: utf-8

# Paths
ROOT_PATH='./'

import sys
sys.path.insert(0, f'{ROOT_PATH}src')

SRC_DIR='src/'
DATA_DIR='data/'
MODEL_DIR='model/'

#setting up seed, number of classes, epochs, learning rate, filename
SEED=1
NUM_CLASSES=10
NUM_EPOCHS = 2500
LEARNING_RATE = 1.0000e-03
DYNAMIC_EPOCS = 500
EARLY_STOP = 1.0000e-08
USE_PRETRAINED=True

FILENAME='mnist'

