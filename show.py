# -*-coding:utf-8-*-

import numpy as np
from collections import defaultdict
import re
from keras.utils.np_utils import to_categorical
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sys
from selftools import *
import os
from sklearn.utils import class_weight
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras import initializers
from config import *
from selfmodels import *
config = Config()


if __name__ == "__main__":

    train_facts, train_laws, train_accus = load_data("small_data_train_cuted")
    test_facts, test_laws, test_accus = load_data("small_data_test_cuted")

    # accu_class_weights = class_weight.compute_class_weight('balanced',
    #                                                        np.unique(train_facts),
    #                                                        train_accus)
    # law_class_weights = class_weight.compute_class_weight('balanced',
    #                                                       np.unique(train_facts),
    # train_laws)
    filepath = "han.hdf5"
    # checkpoint = ModelCheckpoint(filepath=os.path.join(data_path, "generated/" + filepath), monitor='val_score', mode='max', verbose=1, save_best_only='True')
    print("model fitting - Hierachical attention network")


    print train_facts.shape
    print train_laws.shape
    print train_accus.shape

    print test_facts.shape
    print test_laws.shape
    print test_accus.shape

    for i in range(100):
        print (test_facts[i],test_laws[i],test_accus[i])
