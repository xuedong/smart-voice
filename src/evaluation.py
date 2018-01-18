#!/usr/bin/env python

from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support

import numpy as np
import descriptors as d
import utils as u
import classification as c
import utils
import matlab.engine



def create_dataset(sequence, classes, group, look_back=1):
    assert len(sequence)==len(classes)
    data_seq, data_class = [], []
    groups = []
    dim = sequence[0].shape[0]
    for i in range(1,look_back+1):
        a = []
        for j in range(look_back-i+1):
            a.append(np.zeros((dim,)))
        a = a + sequence[0:i]
        data_seq.append(a)
        data_class.append(classes[i])
        groups.append(group)
    for i in range(len(sequence)-look_back):
        a = sequence[i:(i+look_back+1)]
        data_seq.append(a)
        data_class.append(classes[i + look_back])
        groups.append(group)
    return data_seq, data_class, groups

def cv(X, y, groups, n_splits, di, method="RandomForest", smooth = False,test_loc=False, seed = None, smoothing=False, mbss_pooling='max', **kwargs):
    # Convert lists into np.array
    #X = np.array(X)
    #y = np.array(y)
    #groups = np.array(groups)
    
    #ireproducibility
    if seed != None:
        np.random.seed(seed )
    
    future = 0
    eng = 0
    if test_loc:
        print('Starting Matlab (asynchronously).')
        future = matlab.engine.start_matlab(async=True)

    if "predict_method" in kwargs.keys():
        predict_method = kwargs["predict_method"]
    else:
        predict_method = "naive"

    if "nscr" in kwargs.keys():
        nsrc = kwargs["nsrc"]
    else:
        nsrc = 3
        
    if "nscr_no" in kwargs.keys():
        nsrc_no = kwargs["nsrc_no"]
    else:
        nsrc_no = 2

    # Group K-foldi
    if method == "LSTM" or "look_back" in kwargs.keys():
        if "look_back" in kwargs.keys():
            look_back = kwargs["look_back"]
        else:
            look_back = 1
        X_seq, X_seq_groups = d.convert_to_sequences(X,groups)
        y_seq, y_seq_groups = d.convert_to_sequences(y,groups)
        split_X = []
        split_y = []
        split_groups = []
        assert len(X_seq)==len(y_seq) & len(X_seq)==len(X_seq_groups)
        for i in range(len(X_seq)):
            nX, ny, ng = create_dataset(X_seq[i],y_seq[i],X_seq_groups[i], look_back=look_back)
            split_X = split_X + nX
            split_y = split_y + ny
            split_groups = split_groups + ng
            del(nX, ny, ng)
        del(X_seq, X_seq_groups, y_seq, y_seq_groups)
        split_X = np.array(split_X)
        split_y = np.array(split_y)
        split_groups = np.array(split_groups)
        if not method == "LSTM":
            split_X = split_X.reshape((split_X.shape[0],split_X.shape[1]*split_X.shape[2]))
    else:
        split_X, split_y, split_groups = X, y, groups
    gkf = GroupKFold(n_splits=n_splits)
    test_predicted_classes =  np.array([])
    test_classes = np.array([])
    test_predicted_dir_classes = np.array([])
    test_dir_classes = np.array([])
    test_dir_found = np.array([])
    i = 1
    print("Starting crossvalidation")
    for train, test in gkf.split(split_X, split_y, groups=split_groups):
        print("Starting", i,"th crossvalidation")
        i += 1
        test_classes = np.append(test_classes, split_y[test])
        train_values = split_X[train]
        train_classes = split_y[train]
        train_group = split_groups[train]
        test_values = split_X[test]
        test_group = split_groups[test]
        test_p = c.predict(train_values, train_classes, train_group, test_values, test_group, seed, method = method, **kwargs)
        if smoothing:
            test_p = utils.low_pass_gaussian(test_p)
        test_predicted_classes = np.append(test_predicted_classes,test_p)
        if test_loc: # Calculate locations
            for gr in np.unique(test_group):
                if eng == 0:
                    print('\nInitializing Matlab engine.')
                    eng = future.result() # get Matlab engine
                    eng.addpath('src')
                print('\n === Processing group '+str(gr)+', i.e. file '+str(di[gr])+' ===')
                fname = di[gr]
                idx = np.where(test_group == gr)
                test_y_gr = test_p[idx] 
                # for i in test_y_gr:
                    # print('%d ' % i, end="")
                # print('\n')
                le = float((test_y_gr.shape[0]+1)*16/1000) # length in secs
                # print(le)
                test_y_smooth = utils.low_pass_gaussian(test_y_gr)
                mbss_times = utils.seq_to_mbss_times(test_y_smooth)
                # print(test_y_times)
                from_sec_no = mbss_times[0][0]
                to_sec_no = mbss_times[0][1]
                from_sec = mbss_times[1][0]
                to_sec = mbss_times[1][1]
                direction_speaker_true = utils.get_speaker_angles(fname)
                direction_noises_true = utils.get_noises_angles(fname)
                direction_all_true = np.vstack([direction_speaker_true, direction_noises_true])
                dirs_speaking = utils.get_mbss(eng, fname, from_sec, to_sec, nsrc, mbss_pooling)
                if predict_method == 'naive':
                    print('Calculating sources from '+str(from_sec)+' to '+str(to_sec)+' (speaking).')
                    direction_predicted = dirs_speaking
                    dirs_all = dirs_speaking
                elif predict_method == 'naive-2calc' or predict_method == 'naive-2calc-combine-1st':
                    print('Calculating sources from '+str(from_sec)+' to '+str(to_sec)+' (speaking).')
                    print('Calculating sources from '+str(from_sec_no)+' to '+str(to_sec_no)+' (not speaking).')
                    dirs_pred_no = utils.get_mbss(eng, fname, from_sec_no, to_sec_no, nsrc_no, mbss_pooling)
                    direction_predicted = (dirs_speaking, dirs_pred_no)
                    dirs_all = np.vstack([dirs_speaking, dirs_pred_no])
                else:
                    raise NameError('Unknown error!')
                found_sources = utils.found_sources(direction_all_true,dirs_all, method = predict_method, **kwargs)
                # Classes for prediction, and directions keeped
                (direction_predicted_classes, keep_pred) = utils.predict_from_direction(direction_predicted, method = predict_method, **kwargs)
                # True classes
                direction_classes_true = np.append(np.zeros((len(direction_speaker_true),), dtype=int),np.ones((len(direction_noises_true),), dtype=int))
                # Associate found / true classes
                direction_associate_classes = utils.associate_classes(direction_all_true, direction_classes_true, keep_pred, direction_predicted_classes)
                test_predicted_dir_classes = np.append(test_predicted_dir_classes, direction_associate_classes)
                test_dir_classes = np.append(test_dir_classes, direction_classes_true)
                test_dir_found = np.append(test_dir_found, found_sources)

    test_dir_found_true = np.zeros(test_dir_found.shape)
    prfs = {}
    prfs["global"] = precision_recall_fscore_support(test_classes, test_predicted_classes, average = None)
    prfs["macro"] = precision_recall_fscore_support(test_classes, test_predicted_classes, average = 'macro')
    if test_loc:
        prfs["found_loc"] = precision_recall_fscore_support(test_dir_found_true, test_dir_found, average = None)
        prfs["global_loc"] = precision_recall_fscore_support(test_dir_classes, test_predicted_dir_classes, average = None)
        prfs["macro_loc"] = precision_recall_fscore_support(test_dir_classes, test_predicted_dir_classes, average = 'macro')
    return prfs
