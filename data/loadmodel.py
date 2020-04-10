# -*--coding:utf-8*-

from config import Config
from keras.engine.topology import Layer, InputSpec
import numpy as np
from keras.layers import Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed
from keras.layers import Dense, Input, Flatten, Merge, Lambda, Activation
from keras.layers import average, merge, concatenate, subtract, multiply
from keras import initializers
from keras import backend as K
from keras.models import Model
config = Config()

