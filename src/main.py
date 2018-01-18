#!/usr/bin/env python

import sys
import descriptors as d
import classification as c
import evaluation as e
import time
import numpy as np
import gc

def main(*kargs, **kwargs):
    #(X, y, g, di) = d.get_descriptors(['tiny-clean'],[('MFCC',{})])
    (X, y, g, di) = d.get_descriptors(['tiny-noisy'],[('MFCC',{})])    
    #(X1, y1, g1, di1) = d.get_descriptors(['tiny-noisy'],[('MFCC',{'keep': '1'})])    
    # (X, y, g, di) = d.get_descriptors(['tiny-noisy'],[('MFCC',{"multi":False})])    
    #(X, y, g, di) = d.get_descriptors(['tiny-clean','tiny-noisy','tiny-noises'],[('MFCC',{"multi":False})])
    
    #(X, y, g, di) = d.get_descriptors(['clean'],[('MFCC',{})])
    #(X, y, g, di) = d.get_descriptors(['noisy'],[('MFCC',{})])
    #X, y, g, di) = d.get_descriptors(['noisy'],[('MFCC',{"multi":False})])
    #(X, y, g, di) = d.get_descriptors(['noisy'],[('MFCC',{"keep": '3'})])
    # prfs = e.cv(X,y,g,2,di, seed = 0)
    # print(prfs)
    # del(X,y,g,di)
    # (X, y, g, di) = d.get_descriptors(['noisy'],[('MFCC',{"multi":False,"keep": '2'})])
    # prfs = e.cv(X,y,g,2,di, seed = 0)
    # print(prfs)
    # del(X,y,g,di)
    # (X, y, g, di) = d.get_descriptors(['noisy'],[('MFCC',{"multi":False,"keep": '3'})])
    # prfs = e.cv(X,y,g,2,di, seed = 0)
    # print(prfs)
    # del(X,y,g,di)
    # (X, y, g, di) = d.get_descriptors(['noisy'],[('MFCC',{"multi":False})])
    # prfs = e.cv(X,y,g,2,di, seed = 0)
    # print(prfs)
    # del(X,y,g,di)

    # (X, y, g, di) = d.get_descriptors(['noisy'],[('MFCC',{"multi":False})])
    #(X, y, g, di) = d.get_descriptors(['clean','noisy','noises'],[('MFCC',{"multi":False})])
    # prfs = e.cv(X,y,g,3,di, test_loc = True, seed = 0)
    #prfs = e.cv(X,y,g,2,di, seed = 0, look_back = 1)
    # prfs = e.cv(X,y,g,2,di, method = "LSTM", seed = 0, look_back = 1)

    #prfs = e.cv(X, y, g, 2, di, test_loc=True, seed = 0)
    # prfs = e.cv(X,y,g,2,di, method = "LSTM", test_loc = True, seed = 0, look_back = 1)
    
    #prfs = e.cv(X, y, g, 2, di, test_loc=False, seed = 0)
    #prfs = e.cv(X, y, g, 2, di, test_loc=False, seed = 0, smoothing=True)
    #prfs2 = e.cv(X,y,g,2,di, method = "LSTM", seed = 0, look_back = 1, smoothing=True)
    #prfs = e.cv(X, y, g, 2, di, test_loc=True, seed = 0)
    #prfs = e.cv(X,y,g,2,di, method = "LSTM", seed = 0, look_back = 1)

    # prfs = e.cv(X,y,g,2,di, seed = 0)
    # prfs1 = e.cv(X1,y1,g1,2,di1, seed = 0)
    # prfs = e.cv(X,y,g,2,di, method = "LSTM", seed = 0, look_back = 1)

    # prfs = e.cv(X, y, g, 2, di, test_loc=False, seed = 0)
    # prfs1 = e.cv(X, y, g, 2, di, test_loc=False, seed = 0)
    # prfs = e.cv(X, y, g, 2, di, test_loc=False, seed = 0, smoothing=True)
    # prfs2 = e.cv(X,y,g,2,di, method = "LSTM", seed = 0, look_back = 1, smoothing=True)
    # prfs = e.cv(X, y, g, 2, di, test_loc=True, seed = 0)

    # prfs = e.cv(X,y,g,2,di, method = "LSTM", test_loc=True, seed = 0, look_back = 1, predict_method="naive-2calc")
    prfs = e.cv(X,y,g,2,di, method = "LSTM", test_loc=True, seed = 0, look_back = 1, predict_method="naive", mbss_pooling='sum')
    print(prfs)

    gc.collect()

if __name__ == "__main__":
    main(sys.argv)
