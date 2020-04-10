
# -*-coding:utf-8-*-

import numpy as np
from collections import defaultdict
import re
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import sys
from selftools import *
import os
from sklearn.utils import class_weight
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten, concatenate
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Activation
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K

from keras import initializers
from config import *
from final_selfmodel import *
config = Config()


class Metrics(Callback):
    def cal_metric(self, name, predict, target):
        
        micro_precesion = precision_score(target, predict, average="micro")
        macro_precesion = precision_score(target, predict, average="macro")
        micro_recall = recall_score(target, predict, average="micro")
        macro_recall = recall_score(target, predict, average="macro")
        micro_f1 = f1_score(target, predict, average="micro")
        macro_f1 = f1_score(target, predict, average="macro")
        acc = accuracy_score(target, predict)
        score = macro_f1
        
        print("\n" + name + ": \n")
        print("\n acc: %f mico precesion: %f ,macro precesion: %f \n micro recall: %f ,macro recall: %f \n micro f1: %f ,macro f1: %f \n score %f \n" % (acc, micro_precesion, macro_precesion, micro_recall, macro_recall, micro_f1, macro_f1, score))
        
        return score
    
    def on_train_begin(self, logs={}):
        
        self.scores = []
    
    def on_epoch_end(self, epoch, logs={}):
        
        accu_pred, law_pred, term_pred = self.model.predict(self.validation_data[0])
        accu_np = np.asarray(accu_pred)
        accu_max = accu_np.max(axis=1).reshape(-1, 1)
        accu_pred = np.floor(accu_np/accu_max)
        
        law_np = np.asarray(law_pred)
        law_max = law_np.max(axis=1).reshape(-1, 1)
        law_pred = np.floor(law_np/law_max)
        
        
        term_np = np.asarray(term_pred)
        term_max = term_np.max(axis = 1).reshape(-1, 1)
        term_pred = np.floor(term_np/term_max)
        
        
        accu_target = self.validation_data[1]
        law_target = self.validation_data[2]
        term_target = self.validation_data[3]
        
        accu_score = self.cal_metric("accu", accu_pred, accu_target)
        law_score = self.cal_metric("law", law_pred, law_target)
        term_score = self.cal_metric("term", term_pred, term_target)
        score = accu_score + law_score + term_score
        print("\n Final socre  : " + str(score))
        
        if self.scores == [] or score > max(self.scores):
            filepath = "build_CNN.h5"
            print("now is saving ..")
            self.model.save(os.path.join(data_path, "generated/" + filepath))
        
        self.scores.append(score)
        
        return



if __name__ == "__main__":
    metrics = Metrics()
    model = build_CNN()
    model.compile(loss =
                  {
                  'accu_preds': 'categorical_crossentropy',
                  'law_preds': 'categorical_crossentropy',
                  'term_preds': 'categorical_crossentropy',
                  },
                  optimizer='adam',
                  )
    print(model.metrics_names)
    train_facts, train_laws, train_accus = load_data("new_bigdata_train_cuted")
    test_facts, test_laws, test_accus = load_data("new_bigdata_test_cuted")
    train_singlefacts = np.load(os.path.join(data_path, "generated/new_bigdata_train_cuted_singlefact.npy"))
    test_singlefacts= np.load(os.path.join(data_path, "generated/new_bigdata_test_cuted_singlefact.npy"))
    train_terms = np.load(os.path.join(data_path, "generated/new_bigdata_train_cuted_term.npy"))
    test_terms = np.load(os.path.join(data_path, "generated/new_bigdata_test_cuted_term.npy"))

    callback_lists = [metrics]
    print("model fitting - Hierachical attention network")



    model.summary()
    model.fit(train_singlefacts, {
            'accu_preds': to_categorical(train_accus, num_classes=130),
            'law_preds': to_categorical(train_laws, num_classes=118),
            'term_preds': to_categorical(train_terms, num_classes=11),
            },
            validation_data=(test_singlefacts, {
                             'accu_preds': to_categorical(test_accus, num_classes=130),
                             'law_preds': to_categorical(test_laws, num_classes=118),
                             'term_preds': to_categorical(test_terms, num_classes=11),
                             }),
            epochs=config.epochs,
            batch_size=config.batch_size,
            callbacks=callback_lists)

