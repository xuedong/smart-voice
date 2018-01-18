#!/usr/bin/env python

import descriptors as d

import sys
import os

from sklearn import svm,tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.layers.wrappers import TimeDistributed

import hashlib
from sklearn.externals import joblib
import pickle
from keras.models import model_from_json

def classify_lstm(training_sequences, training_sequences_classes, top_words = 5000, max_review_length = 500, embedding_vector_length = 32, num_cells = 100, nb_epoch = 3, batch_size = 64, look_back = 1, **kwargs):
        model = Sequential()
        model.add(LSTM(num_cells, input_dim = training_sequences.shape[2] ))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        #print(model.summary())
        model.fit(training_sequences, training_sequences_classes, nb_epoch=nb_epoch, batch_size=batch_size)
        return model

def classifier(method, **kwargs):
    if method == "SVM":
        classifier = svm.SVC(kernel='linear',probability=True,verbose=True)
    elif method == "DecisionTree":
        classifier = tree.DecisionTreeClassifier()
    elif method == 'RandomForest':
        classifier = RandomForestClassifier(n_jobs = 8)
    elif method == 'SGD':
        classifier = SGDClassifier(loss="log", n_jobs = 4)
    elif method == 'NaiveBayes':
        classifier = GaussianNB()
    elif method == 'KNN':
        classifier = KNeighborsClassifier(n_jobs = 4)
    classifier.set_params(**kwargs)
    return classifier

# predicts cfa values for [test] using method [method].
def predict(training_values, training_classes, training_group, test_values, test_group, seed, method="RandomForest", use_cache = True, **kwargs):
    args = str(training_group)+str(training_values.shape)+method+str(seed)
    hash = hashlib.md5(args.encode())
    cache_path = 'cache/'+str(hash.hexdigest())+'.pkl'
    if use_cache and os.path.isfile(cache_path):
        print('Loading model from cache',method)
        if method == "LSTM": 
            c_json = pickle.load(open(cache_path,'rb'))
            c = model_from_json(c_json)
            c.load_weights(cache_path+"_weights",by_name = False)
        else:
            c = pickle.load(open(cache_path,'rb'))
    else:
        print('Computing model', method)
        if method == "LSTM":
            c = classify_lstm(training_values, training_classes, **kwargs)
            with open(cache_path, 'wb') as outfile:
                pickle.dump(c.to_json(), outfile, protocol=pickle.HIGHEST_PROTOCOL)
            c.save_weights(cache_path+"_weights")
        else:
            c = classifier(method)
            c.fit(training_values, training_classes)
            with open(cache_path, 'wb') as outfile:
                pickle.dump(c, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    if method == "LSTM":
        return c.predict_classes(test_values)
    else:
        return c.predict(test_values)

