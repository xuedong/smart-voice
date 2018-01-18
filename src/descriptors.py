import os
import math
import csv
import numpy as np
import sys  
import codecs
import pickle
import hashlib

# Directories
DIR_ROOT_AUDIO = 'data/audio/'
DIR_CLEAN = DIR_ROOT_AUDIO+'clean'
DIR_NOISY = DIR_ROOT_AUDIO+'noisy'
DIR_NOISES = DIR_ROOT_AUDIO+'noises'
DIR_TINY_CLEAN = DIR_ROOT_AUDIO+'clean-tiny'
DIR_TINY_NOISY = DIR_ROOT_AUDIO+'noisy-tiny'
DIR_TINY_NOISES = DIR_ROOT_AUDIO+'noises-tiny'
DIRS = {'clean': DIR_CLEAN, 'noisy': DIR_NOISY, 'noises': DIR_NOISES,
        'tiny-clean': DIR_TINY_CLEAN, 'tiny-noisy': DIR_TINY_NOISY,
        'tiny-noises': DIR_TINY_NOISES}

DESCS_TYPES = ['MFCC']

def get_descriptors(cats, descs, use_cache=True):
    # Get descriptors for categories in cat ('clean', 'noisy' or 'noises').
    # Inputs:
    # - [cats] is a list of string in ['clean','noisy','noises'], e.g. ['clean'].
    #   Don't include 'clean' if you're using 8-channels descriptors!
    # - [descs] is a list of (desc_type,desc_args) couples, e.g. [('MFCC',{}),('TDOA',args)]
    # Output:
    # - [X] is a mxn array, n being the size of concatenated descriptors,
    #   and m the number of training data (each line corresponds to a 32ms window of 
    #   one of the files in [cats])
    # - [y] is the true class (speaking / not speaking)
    # - [groups] is a vector of length m of integers indicating which lines of X
    #   have been extracted from the same file (for group cross-validation).
    # - [dic_groups] is a dictionary 'group numper (int)' > 'audio file name (string)'

    args = str(cats)+str(descs)
    hash = hashlib.md5(args.encode())
    cache_path = 'cache/'+str(hash.hexdigest())+'.npy'
    di_path = 'cache/'+str(hash.hexdigest())+'_di.npy'
    # fname_args = 'cache/'+str(hash.hexdigest())+'_args.txt'
    if use_cache and os.path.isfile(cache_path):
        print('Loading descriptors from cache')
        cache = np.load(cache_path)
        X = cache[0]
        y = cache[1]
        groups = cache[2]
        dic_groups = pickle.load(open(di_path,'rb'))
    else:
        print('Fetching descriptors (no cache).')
        X = []
        y = []
        groups = []
        group = 0
        dic_groups = {}
        for cat in cats: # Concatenate X for all categories
            cat_dir = ''
            try:
                cat_dir = DIRS[cat]
            except KeyError:
                print('descriptors.get_descriptors(): Category "'+cat+'" is not a correct category.')
            # Go through all files and concatenates descriptors
            for dir_name, subdir_list, file_list in os.walk(cat_dir):
                # print('Going through directory: %s' % dir_name)
                if cat == 'clean' or cat == 'tiny-clean':
                    nb_files = int(math.ceil(len(file_list)/2))
                else:
                    nb_files = int(math.ceil(len(file_list))/9)
                prog = 1
                # Find all audio file of category [cat]
                for fname in file_list:
                    file_name = dir_name+'/'+fname
                    ext = file_name[-4:]
                    if ext == '.wav': # process only audio files
                        sys.stdout.write("\rProcessing file {}/{}".format(prog,nb_files))
                        prog = prog +1
                        group = group+1 # group for this file
                        dic_groups[group] = file_name
                        X_file = []
                        nb_in_group = 0
                        for (d_type, d_args) in descs:
                            if d_type == 'MFCC':
                                # print('File: %s' % file_name)
                                if cat=='clean' or cat=='tiny-clean':
                                    multi = False
                                else:
                                    if "multi" in d_args.keys():
                                        multi = d_args["multi"]
                                    else:
                                        multi = True
                                X_MFCC = get_MFCC(file_name, multi, **d_args)
                                X_file.append(X_MFCC)
                                nb_in_group = X_MFCC.shape[0]
                                # TODO: Other desc types
                        groups.extend([group]*nb_in_group)
                        X.extend(X_file)
                        # Get transcription if needed
                        if cat=='clean' or cat=='noisy' or cat=='tiny-clean' or cat=='tiny-noisy':
                            y_file = get_transcription(file_name, nb_in_group)
                        else:
                            if labels_type == 'biclass':
                                y_file = [0] * nb_in_group
                            else:
                                raise Exception("Unknown labels type.")
                        y.extend(y_file)
        print("\n")
        X = np.vstack(X)
        y = np.array(y)
        groups = np.array(groups)
        assert X.shape[0] == groups.size
        assert X.shape[0] == y.size
        to_cache = np.array([X,y,groups,[]],dtype=object)
        to_cache.shape
        np.save(cache_path, to_cache)
        with open(di_path, 'wb') as outfile:
            pickle.dump(dic_groups, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    print('Fetched descriptors, matrix size is %s.' % str(X.shape))
    return (X, y, groups, dic_groups)

def get_MFCC(path, multi_chan, **kwargs):
    MFCC_fname = path[0:-4]+'_Descriptors_Channel_1.txt'
    X = [np.loadtxt(MFCC_fname)]
    if 'keep' in kwargs and kwargs['keep'] == '1':
        X[0] = X[0][:,0:24]
    if 'keep' in kwargs and kwargs['keep'] == '2':
        X[0] = X[0][:,0:48]
    if 'keep' in kwargs and kwargs['keep'] == '3':
        X[0] = X[0][:,0:72]
    if multi_chan: # If this is a 8-channels audio file
        for ch in range(2,8):
            MFCC_fname = path[0:-4]+'_Descriptors_Channel_'+str(ch)+'.txt'
            X.append(np.loadtxt(MFCC_fname))
            if 'keep' in kwargs and kwargs['keep'] == '1':
                X[ch-1] = X[ch-1][:,0:24]
    X = np.hstack(X)
    return X

def get_transcription(path, tot_size):
    # Get the truth values from the transcription files for audio file [path].
    y = []
    trans_path = path[0:-4]+'.txt'
    trans_path = trans_path.replace('audio', 'transcriptions')
    test_speech = lambda s: 0 if s=='[$NO_SPEECH]' else 1
    change_times = [0.]
    cur_state = 1
    start_state = 0
    cur_idx = 0 # current index
    st = 0
    # Detect 'changing' frames.
    with codecs.open(trans_path,'r','utf-8') as trans:
        for line in csv.reader(trans, delimiter="\t"):
            deb = float(line[0])
            end = float(line[1])
            lbl = line[2]
            state = test_speech(lbl)
            if st == 0: # first line
                start_state = state
                cur_state = state
                change_times.append(end)
                cur_idx = 1
            st = st+1
            if cur_state == state:
                change_times[cur_idx] = end
            else:
                change_times.append(end)
                cur_state = state
                cur_idx = cur_idx + 1
    change_ids = [math.floor(t*1000/16) for t in change_times]
    ic = 0
    sc = start_state
    for i in change_ids[1:]:
        y.extend([sc]*(i-ic))
        sc = (sc+1)%2
        ic = i
    len_y = len(y)
    # Just be sure that we didn't cut one frame too early/late because of overlapping (yeah… can surely do something cleaner).
    if len_y>tot_size:
        y = y[:-(len_y-tot_size)]
    elif len_y<tot_size:
        pad = y[len_y-1] * (tot_size-len_y)
        y = y+pad
    y = np.array(y)
    return y 

def pause():
        programPause = input("Press the <ENTER> key to continue...")

def convert_to_sequences(values, group):
    l = []
    actual_g = group[0]
    actual_sequence = []
    new_groups = [actual_g]
    for i in range(len(group)):
        if group[i] == actual_g:
            actual_sequence.append(values[i])
        else:
            actual_g = group[i]
            new_groups.append(actual_g)
            l.append(actual_sequence)
            actual_sequence = [values[i]]
    l.append(actual_sequence)
    l = np.array(l)
    return l, new_groups

