#!/usr/bin/env python
# coding: utf-8

sys.path.insert(0, f'src/')

import config
from data_loader import load_data, load_cvnc_data, load_minst_data, normalize_image_data

def main():

    train_x, train_y, test_x, test_y, classes = load_data()
    

if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()