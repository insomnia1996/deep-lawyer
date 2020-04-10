# -*--coding:utf-8*-

from config import Config
from keras.engine.topology import Layer, InputSpec
import numpy as np
from keras.layers import Embedding, Dropout, LSTM, GRU, Bidirectional, TimeDistributed, Average
from keras.layers import Dense, Input, Flatten, Merge, Lambda, Activation, Conv2D, MaxPooling2D, Conv1D, MaxPooling1D, Reshape, Dropout, Average
from keras.layers import average, merge, concatenate, subtract, multiply, dot, add, maximum
from keras import initializers
from keras import backend as K
from keras.models import Model
config = Config()


class AttLayer(Layer):
    def __init__(self, attention_dim,**kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__(** kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # if self.return_attention:
        #     return [mask, None]
        return None

    def call(self, x, mask=None):
        # [batch_size, sel_len, attention_dim]
        # [batch_size, attention_dim]
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)  # 对应公式(6)
        # 对应公式(6)
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)  # 对应公式(7)
        weighted_input = x * ait  # 对应公式(7)
        output = K.sum(weighted_input, axis=1)  # 对应公式(7)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

"""
    model1 : double-LSTM
"""
def build_model1():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(config.hid_size, return_sequences=False))(embedded_sequences)
    sentEncoder = Model(sentence_input, l_lstm)
    
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(config.hid_size, return_sequences=False))(review_encoder)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(l_lstm_sent)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(l_lstm_sent)
    model = Model(inputs=[review_input], outputs=[accu_preds, law_preds])
                                
    return model

"""
    model2 : word-level-attention+LSTM
"""
def build_model2():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(config.hid_size, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(config.hid_size))(review_encoder)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(l_lstm_sent)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(l_lstm_sent)
    model = Model(inputs=[review_input], outputs=[accu_preds, law_preds])
                                
    return model

"""
    model3 : double-attention+LSTM
"""
def build_model3():

    embedding_matrix = np.load(config.embedding_path)

    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)

    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(config.hid_size, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(LSTM(config.hid_size, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(l_att_sent)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(l_att_sent)
    model = Model(inputs=[review_input], outputs=[accu_preds, law_preds])

    return model

'''
    single-RNN
'''
def build_model4():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero=True)
        
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(config.hid_size))(embedded_sequences)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(l_lstm)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(l_lstm)
    model = Model(inputs=[sentence_input], outputs=[accu_preds, law_preds])
                                
    return model

'''
    single_rnn_word_level_attention
'''
def build_model5():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = True
                                )
        
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(LSTM(config.hid_size, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(l_att)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(l_att)
    model = Model(inputs=[sentence_input], outputs=[accu_preds, law_preds])
                                
    return model

'''
    CNN-multi
'''

def build_model6():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
        
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    model = Model(inputs=[sentence_input], outputs=[accu_preds, law_preds])
                                
    return model
'''
embedding without init
'''

def build_model7():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
#                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    model = Model(inputs=[sentence_input], outputs=[accu_preds, law_preds])
                                
    return model

def normalize(x):
    x /= (K.sum(x, axis = -1, keepdims = True) + K.epsilon())
    return x
'''
    cnn+double-check
'''

def build_model8():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
#                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
#                                    config.num_accu_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)

    accu_embedding_layer = Embedding(config.num_accu_liu,
#                                     config.num_law_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)

    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
#    law_preds2 = Dropout(0.5)(law_preds2)

    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
#    accu_preds2 = Dropout(0.5)(accu_preds2)


    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')

    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)

    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
    
    return model


def build_model9():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
        
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    accu_preds = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    term_preds = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(feature)
    model = Model(inputs=[sentence_input], outputs=[accu_preds, law_preds, term_preds])
                                
    return model

def build_model11():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
#                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    law_fix = concatenate([feature, law_merge2])
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3])
                                
    return model

def build_model10():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    fix_merge = concatenate([accu_merge2, law_merge2])
    
    final = concatenate([feature, fix_merge])
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(final)
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1])
                                
    return model

def getlog(x):
    x = K.log(x + K.epsilon())
    x *= -1.0
    return x

def build_model12():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    law_fix = concatenate([feature, law_merge2])
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    term_log = Lambda(getlog)(term_preds1)
    cross = merge([term_log, term_preds2], mode = 'dot', name = 'cross')
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model13():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    law_fix = concatenate([feature, law_merge2])
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = merge([term_log1, term_preds2], mode = 'dot')
    cross2 = merge([term_log2, term_preds1], mode = 'dot')
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def dis(x):
    M = [30, 15, 8.5, 6, 4, 2.5, 1.5, 0.875, 0.625, 0.25, 0]
    M = np.asarray(M, dtype = 'float32')
    M = K.expand_dims(M, 1)
    x = K.dot(x, M)
    return x * x

def build_model14():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    law_fix = concatenate([feature, law_merge2])
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])


    cross = subtract([term_preds1, term_preds2])
    cross2 = Lambda(dis, name = 'cross')(cross)
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross2])
                                
    return model

def dis2(x):
    x *= 0
    return x

def build_model15():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    law_fix = concatenate([feature, law_merge2])
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    
    
    cross = subtract([term_preds1, term_preds2])
    cross2 = Lambda(dis2, name = 'cross')(cross)
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross2])
                                
    return model

def dis3(x):
    M = [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    M = np.asarray(M, dtype = 'float32')
    M = K.expand_dims(M, 1)
    x = K.dot(x, M)
    return x * x

def build_model16():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    law_fix = concatenate([feature, law_merge2])
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    
    
    cross = subtract([term_preds1, term_preds2])
    cross2 = Lambda(dis3, name = 'cross')(cross)
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross2])
                                
    return model

def build_model17():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    term_accu = Dense(config.num_term_liu, activation ='sigmoid')(accu_merge2)
    term_law = Dense(config.num_term_liu, activation ='sigmoid')(law_merge2)
    
    fix_term = merge([term_accu,term_law], mode = 'mul')
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(feature)
    fix_term = merge([fix_term,term_preds1], mode = 'mul')
    
    term_preds2 = Lambda(normalize, name = 'term_preds2')(fix_term)
    
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2])
                                
    return model

def build_model18():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    accu_fix = Dense(config.word_embed_size, activation = 'elu')(accu_fix)
    law_fix = concatenate([feature, law_merge2])
    law_fix = Dense(config.word_embed_size, activation = 'elu')(law_fix)
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = merge([term_log1, term_preds2], mode = 'dot')
    cross2 = merge([term_log2, term_preds1], mode = 'dot')
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model19():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = concatenate([feature, accu_merge2])
    accu_fix = Dense(config.word_embed_size, activation = 'elu')(accu_fix)
    accu_fix = Dropout(0.5)(accu_fix)
    law_fix = concatenate([feature, law_merge2])
    law_fix = Dense(config.word_embed_size, activation = 'elu')(law_fix)
    law_fix = Dropout(0.5)(law_fix)
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = merge([term_log1, term_preds2], mode = 'dot')
    cross2 = merge([term_log2, term_preds1], mode = 'dot')
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model20():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
#                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = multiply([feature, accu_merge2])
    
    law_fix = multiply([feature, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = merge([term_log1, term_preds2], mode = 'dot')
    cross2 = merge([term_log2, term_preds1], mode = 'dot')
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model21():
    
    embedding_matrix = np.load(config.embedding_path)
    law2accu = np.load(config.law2accu)
    accu2law = np.load(config.accu2law)
    law2term = np.load(config.law2term)
    accu2term = np.load(config.accu2term)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
#                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_accu = Embedding(config.num_law_liu,
                        config.num_accu_liu,
                        weights = [law2accu],
                        input_length=config.num_law_liu,
                        trainable=False,
                        mask_zero=False)
    
    accu_law = Embedding(config.num_accu_liu,
                         config.num_law_liu,
                         weights = [accu2law],
                         input_length=config.num_accu_liu,
                         trainable=False,
                         mask_zero=False)
                         
                         
    law_term = Embedding(config.num_law_liu,
                          config.num_term_liu,
                          weights = [law2term],
                          input_length=config.num_law_liu,
                          trainable=False,
                          mask_zero=False)
     
    accu_term = Embedding(config.num_accu_liu,
                          config.num_term_liu,
                          weights = [accu2term],
                          input_length=config.num_accu_liu,
                          trainable=False,
                          mask_zero=False)
                         
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_law(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Activation('sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_accu(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Activation('sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_embedded_sequences2 = accu_term(accu_input)
    law_embedded_sequences2 = law_term(law_input)
    
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences2], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences2], mode = 'dot', dot_axes = (1, 1))
    
    fix_merge = multiply([accu_merge2, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(feature)
    
    term_preds2 = multiply([term_preds1, fix_merge])
    
    term_preds3 = Activation('softmax', name = "term_preds3")(term_preds2)
    
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds3])
                                
    return model

def build_model22():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
#                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_term = Dense(config.num_term_liu, activation = 'sigmoid')(accu_merge2)
    law_term = Dense(config.num_term_liu, activation = 'sigmoid')(law_merge2)
    
    accu_fix = multiply([feature, accu_merge2])
    
    law_fix = multiply([feature, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax')(accu_fix)
    term_preds1 = multiply([term_preds1, accu_term])
    term_preds1 = Lambda(normalize, name = "term_preds1")(term_preds1)
    
    term_preds2 = Dense(config.num_term_liu, activation ='softmax')(law_fix)
    tern_preds2 = multiply([term_preds2, law_term])
    term_preds2 = Lambda(normalize, name = "term_preds2")(term_preds2)
    
    term_preds3 = Average(name = 'term_preds3')([term_preds1, term_preds2])
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = merge([term_log1, term_preds2], mode = 'dot')
    cross2 = merge([term_log2, term_preds1], mode = 'dot')
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model23():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
#                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_term = Dense(config.num_term_liu, activation = 'sigmoid')(accu_merge2)
    law_term = Dense(config.num_term_liu, activation = 'sigmoid')(law_merge2)
    
    accu_fix = multiply([feature, accu_merge2])
    
    law_fix = multiply([feature, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax')(accu_fix)
    term_preds1 = multiply([term_preds1, accu_term])
    term_preds1 = Lambda(normalize, name = "term_preds1")(term_preds1)
    
    term_preds2 = Dense(config.num_term_liu, activation ='softmax')(law_fix)
    tern_preds2 = multiply([term_preds2, law_term])
    term_preds2 = Lambda(normalize, name = "term_preds2")(term_preds2)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = merge([term_log1, term_preds2], mode = 'dot')
    cross2 = merge([term_log2, term_preds1], mode = 'dot')
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model


def build_model24():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = multiply([feature, accu_merge2])
    
    law_fix = multiply([feature, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = merge([term_log1, term_preds2], mode = 'dot')
    cross2 = merge([term_log2, term_preds1], mode = 'dot')
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model25():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    accu_fix = multiply([feature, accu_merge2])
    
    law_fix = multiply([feature, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3])
                                
    return model

def build_model26():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = merge([accu_preds3,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_merge2 = merge([law_preds3,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    
    fix = multiply([feature, accu_merge2, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(fix)

    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1])
    return model



def build_model27():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)

    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)

    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)

    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)

    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)


    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)


    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])

    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)

    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2 = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2 = Dense(config.hid_size, activation = 'elu')(law_merge2)

    accu_fix = multiply([feature, accu_merge2])

    law_fix = multiply([feature, law_merge2])

    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)


    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model28():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.word_embed_size,
                                     input_length=config.num_term_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.word_embed_size, activation = 'elu')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.word_embed_size, activation = 'elu')(law_merge2)
    
    accu_fix = multiply([feature, accu_merge2_d])
    
    law_fix = multiply([feature, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    
    
    term_input = Input(shape = (config.num_term_liu, ), dtype = 'int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3, term_embedded_sequences], axes = (1, 1))
    
    accu_preds4 = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    law_preds4 = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    
    accu_preds5 = multiply([accu_preds3, accu_preds4])
    law_preds5 = multiply([law_preds3, law_preds4])
    
    accu_preds5 = Lambda(normalize, name = 'accu_preds5')(accu_preds5)
    law_preds5 = Lambda(normalize, name = 'law_preds5')(law_preds5)
    
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross, accu_preds5, law_preds5])
                                
    return model

def build_model29():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.word_embed_size,
                                     input_length=config.num_term_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2 = Dense(config.word_embed_size, activation = 'elu')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2 = Dense(config.word_embed_size, activation = 'elu')(law_merge2)
    
    accu_fix = multiply([feature, accu_merge2])
    
    law_fix = multiply([feature, law_merge2])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    
    
    term_input = Input(shape = (config.num_term_liu, ), dtype = 'int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3, term_embedded_sequences], axes = (1, 1))
    term_merge = Dense(config.word_embed_size, activation = 'elu')(term_merge)
    
    accu_preds4 = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    law_preds4 = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    
    accu_preds5 = multiply([accu_preds3, accu_preds4])
    law_preds5 = multiply([law_preds3, law_preds4])
    
    accu_preds5 = Lambda(normalize, name = 'accu_preds5')(accu_preds5)
    law_preds5 = Lambda(normalize, name = 'law_preds5')(law_preds5)
    
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross, accu_preds5, law_preds5])
                                
    return model

def build_model30():
    
    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.word_embed_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.word_embed_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.word_embed_size,
                                     input_length=config.num_term_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = 75, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((75,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((75,))(pooling3)
    cnn4 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((75,))(pooling4)
    cnn5 = Conv2D(filters = 75, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((75,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
#    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge = Dense(config.word_embed_size, activation = 'elu')(law_merge)
    
    law_fix = multiply([feature, law_merge])
    
    accu_preds1 = Dense(config.num_accu_liu, activation = 'softmax', name = "accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    
    
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    law_pred_merge = multiply([law_preds1, law_preds2])
    law_preds3 = Lambda(normalize, name = "law_preds3")(law_pred_merge)
    
    
    
    
    accu_merge = Dense(config.word_embed_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge])
    
    
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape = (config.num_term_liu, ), dtype = 'int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3, term_embedded_sequences], axes = (1, 1))
    
    accu_preds4 = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    law_preds4 = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    
    accu_preds5 = multiply([accu_preds1, accu_preds4])
    law_preds5 = multiply([law_preds1, law_preds4])
    
    accu_preds5 = Lambda(normalize, name = 'accu_preds5')(accu_preds5)
    law_preds5 = Lambda(normalize, name = 'law_preds5')(law_preds5)
    
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input], outputs=[accu_preds1, law_preds1, law_preds3, term_preds1, term_preds2, term_preds3, cross, accu_preds5, law_preds5])
                                
    return model


def build_model31():
    
#    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
#                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2 = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2 = Dense(config.hid_size, activation = 'elu')(law_merge2)
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
    pair_front = Dense(config.hid_size, activation = 'tanh')(pair_front)
    pair_last = embedding_layer(pair_last_word_input)
    pair_last = Dense(config.hid_size, activation = 'tanh')(pair_last)
    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    
    weight_law = dot([pair, law_merge2], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair, accu_merge2], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    
    accu_fix = multiply([feature, accu_merge2, feature_pair_accu])
    
    law_fix = multiply([feature, law_merge2, feature_pair_law])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model32_special():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid')(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2 = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2 = Dense(config.hid_size, activation = 'elu')(law_merge2)
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
#    pair_front = Dense(config.hid_size, activation = 'tanh', use_bias=True)(pair_front)

    pair_last = embedding_layer(pair_last_word_input)
#    pair_last = Dense(config.hid_size, activation = 'tanh', )(pair_last)

    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
#    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    
    accu_fix = multiply([feature, accu_merge2, feature_pair_accu])
    
    law_fix = multiply([feature, law_merge2, feature_pair_law])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model32_std_mul():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid', use_bias = False)(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid', use_bias = False)(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    accu_merge2_att = Dense(config.hid_size, activation = 'tanh')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.hid_size, activation = 'elu')(law_merge2)
    law_merge2_att = Dense(config.hid_size, activation = 'tanh')(law_merge2)
    
    
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
    #    pair_front = Dense(config.hid_size, activation = 'tanh', use_bias=True)(pair_front)
    
    pair_last = embedding_layer(pair_last_word_input)
    #    pair_last = Dense(config.hid_size, activation = 'tanh', )(pair_last)
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = multiply([feature, feature_pair_accu])
    feature_law = multiply([feature, feature_pair_law])
    
    accu_fix = multiply([feature_accu, accu_merge2_d])
    
    law_fix = multiply([feature, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model32_std_add():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid', use_bias = False)(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid', use_bias = False)(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    accu_merge2_att = Dense(config.hid_size, activation = 'tanh')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.hid_size, activation = 'elu')(law_merge2)
    law_merge2_att = Dense(config.hid_size, activation = 'tanh')(law_merge2)
    
    
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
    #    pair_front = Dense(config.hid_size, activation = 'tanh', use_bias=True)(pair_front)
    
    pair_last = embedding_layer(pair_last_word_input)
    #    pair_last = Dense(config.hid_size, activation = 'tanh', )(pair_last)
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = add([feature, feature_pair_accu])
    feature_law = add([feature, feature_pair_law])
    
    accu_fix = multiply([feature_accu, accu_merge2_d])
    
    law_fix = multiply([feature, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])

    return model

def build_model32_std_max():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid', use_bias = False)(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid', use_bias = False)(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    accu_merge2_att = Dense(config.hid_size, activation = 'tanh')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.hid_size, activation = 'elu')(law_merge2)
    law_merge2_att = Dense(config.hid_size, activation = 'tanh')(law_merge2)
    
    
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
    #    pair_front = Dense(config.hid_size, activation = 'tanh', use_bias=True)(pair_front)
    
    pair_last = embedding_layer(pair_last_word_input)
    #    pair_last = Dense(config.hid_size, activation = 'tanh', )(pair_last)
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = maximum([feature, feature_pair_accu])
    feature_law = maximum([feature, feature_pair_law])
    
    accu_fix = multiply([feature_accu, accu_merge2_d])
    
    law_fix = multiply([feature, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model32_std_cat():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid', use_bias = False)(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid', use_bias = False)(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    accu_merge2_att = Dense(config.hid_size, activation = 'tanh')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.hid_size, activation = 'elu')(law_merge2)
    law_merge2_att = Dense(config.hid_size, activation = 'tanh')(law_merge2)
    
    
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
    #    pair_front = Dense(config.hid_size, activation = 'tanh', use_bias=True)(pair_front)
    
    pair_last = embedding_layer(pair_last_word_input)
    #    pair_last = Dense(config.hid_size, activation = 'tanh', )(pair_last)
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    accu_fix = multiply([feature_accu, accu_merge2_d])
    
    law_fix = multiply([feature, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model


def build_model32_std_mul_test():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid', use_bias = False)(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid', use_bias = False)(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.hid_size, activation = 'elu')(accu_merge2)
#    accu_merge2_att = Dense(config.hid_size, activation = 'tanh')(accu_merge2)

    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.hid_size, activation = 'elu')(law_merge2)
#    law_merge2_att = Dense(config.hid_size, activation = 'tanh')(law_merge2)

    
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
    #    pair_front = Dense(config.hid_size, activation = 'tanh', use_bias=True)(pair_front)
    
    pair_last = embedding_layer(pair_last_word_input)
    #    pair_last = Dense(config.hid_size, activation = 'tanh', )(pair_last)
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2_d], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2_d], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = multiply([feature, feature_pair_accu])
    feature_law = multiply([feature, feature_pair_law])
    
    accu_fix = multiply([feature_accu, accu_merge2_d])
    
    law_fix = multiply([feature, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_model32():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid', use_bias = False)(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid', use_bias = False)(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    #    accu_merge2_att = Dense(config.hid_size, activation = 'tanh')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.hid_size, activation = 'elu')(law_merge2)
    #    law_merge2_att = Dense(config.hid_size, activation = 'tanh')(law_merge2)
    
    
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    
    pair_front = embedding_layer(pair_front_word_input)
    #    pair_front = Dense(config.hid_size, activation = 'tanh', use_bias=True)(pair_front)
    
    pair_last = embedding_layer(pair_last_word_input)
    #    pair_last = Dense(config.hid_size, activation = 'tanh', )(pair_last)
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2_d], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2_d], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)

#    feature_accu = multiply([feature, feature_pair_accu])
#    feature_law = multiply([feature, feature_pair_law])



    accu_fix = multiply([feature, accu_merge2_d])
    
    law_fix = multiply([feature_law, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

#final:
def build_model33():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
#                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
#                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                     config.word_embed_size / 8,
#                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    law_preds2 = Dense(config.num_law_liu, activation = 'sigmoid', use_bias = False)(accu_merge)
    #    law_preds2 = Dropout(0.5)(law_preds2)
    
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_liu, activation = 'sigmoid', use_bias = False)(law_merge)
    #    accu_preds2 = Dropout(0.5)(accu_preds2)
    
    
    accu_pred_merge = multiply([accu_preds1, accu_preds2])
    law_pred_merge = multiply([law_preds1, law_preds2])
    
    accu_preds3 = Lambda(normalize, name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Lambda(normalize, name = 'law_preds3')(law_pred_merge)
    
    
    
    accu_merge2 = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge2_d = Dense(config.hid_size, activation = 'elu')(accu_merge2)
    #    accu_merge2_att = Dense(config.hid_size, activation = 'tanh')(accu_merge2)
    
    law_merge2 = dot([law_preds3,law_embedded_sequences], axes = (1, 1))
    law_merge2_d = Dense(config.hid_size, activation = 'elu')(law_merge2)
    #    law_merge2_att = Dense(config.hid_size, activation = 'tanh')(law_merge2)
    
    
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge2_d], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge2_d], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
#    feature_accu = concatenate([feature, feature_pair_accu])
#    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    #    feature_accu = multiply([feature, feature_pair_accu])
    #    feature_law = multiply([feature, feature_pair_law])
    
    
    
    accu_fix = multiply([feature, accu_merge2_d])
    
    law_fix = multiply([feature_law, law_merge2_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = 'term_preds3')(term_preds3)
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    cross1 = dot([term_log1, term_preds2], axes = -1)
    cross2 = dot([term_log2, term_preds1], axes = -1)
    cross = Average(name = "cross")([cross1, cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
           outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3, term_preds1, term_preds2, term_preds3, cross])
                                
    return model

def build_final_test1():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds,law_embedded_sequences], axes = (1, 1))
    law_merge_d = Dense(config.hid_size, activation = 'elu')(law_merge)
    law_fix = multiply([feature, law_merge_d])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    accu_preds2 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds2")(law_fix)
    
    accu_preds3 = multiply([accu_preds1, accu_preds2])
    accu_preds3 = Lambda(normalize, name = "accu_preds3")(accu_preds3)
    
    accu_log1 = Lambda(getlog)(accu_preds1)
    accu_log2 = Lambda(getlog)(accu_preds2)
    accu_cross1 = dot([accu_log1, accu_preds2], axes = -1)
    accu_cross2 = dot([accu_log2, accu_preds1], axes = -1)
    accu_cross = Average(name = "accu_cross")([accu_cross1, accu_cross2])
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge_d = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge_d])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    term_cross1 = dot([term_log1, term_preds2], axes = -1)
    term_cross2 = dot([term_log2, term_preds1], axes = -1)
    term_cross = Average(name = "term_cross")([term_cross1, term_cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input],
                  outputs=[law_preds, accu_preds1, accu_preds2, accu_preds3, accu_cross, term_preds1, term_preds2, term_preds3, term_cross])
    
    return model

def build_final_test2():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
                                    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds = Dense(config.num_law_liu, activation='softmax', name="law_preds")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds,law_embedded_sequences], axes = (1, 1))
    law_merge_d = Dense(config.hid_size, activation = 'elu')(law_merge)
    law_fix = multiply([feature, law_merge_d])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(feature)
    accu_preds2 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds2")(law_fix)
    
    accu_preds3 = multiply([accu_preds1, accu_preds2])
    accu_preds3 = Lambda(normalize, name = "accu_preds3")(accu_preds3)
    
    accu_log1 = Lambda(getlog)(accu_preds1)
    accu_log2 = Lambda(getlog)(accu_preds2)
    accu_cross1 = dot([accu_log1, accu_preds2], axes = -1)
    accu_cross2 = dot([accu_log2, accu_preds1], axes = -1)
    accu_cross = Average(name = "accu_cross")([accu_cross1, accu_cross2])
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds3,accu_embedded_sequences], axes = (1, 1))
    accu_merge_d = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge_d])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_d], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_d], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))


    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    term_log1 = Lambda(getlog)(term_preds1)
    term_log2 = Lambda(getlog)(term_preds2)
    term_cross1 = dot([term_log1, term_preds2], axes = -1)
    term_cross2 = dot([term_log2, term_preds1], axes = -1)
    term_cross = Average(name = "term_cross")([term_cross1, term_cross2])
    model = Model(inputs=[sentence_input, accu_input, law_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds, accu_preds1, accu_preds2, accu_preds3, accu_cross, term_preds1, term_preds2, term_preds3, term_cross])
                                
    return model

def build_final_test3():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
                                     
    term_embedding_layer = Embedding(config.num_term_liu,
                                      config.hid_size,
                                      #                                     input_length=config.num_accu_liu,
                                      trainable=True,
                                      mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_d = Dense(config.hid_size, activation = 'elu')(law_merge)
    law_fix = multiply([feature, law_merge_d])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_d = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge_d])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_d], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_d], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds1, law_preds2,accu_preds1, accu_preds2, term_preds1, term_preds2, term_preds3])
                                
    return model


def build_final_test4():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'elu')(law_merge)
    law_merge_att = Dense(config.hid_size, activation = 'tanh')(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'tanh')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])

    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model

#withatt

def build_final_test5():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'tanh', use_bias = False)(law_merge)
    law_merge_att = Dense(config.hid_size, activation = 'elu')(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'tanh', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model




#without att
def build_final_test6():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'tanh', use_bias = False)(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'tanh', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_semantic], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_semantic], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model

#all tanh
def build_final_test7():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'tanh', use_bias = False)(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'tanh', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'tanh')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'tanh')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'tanh')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_semantic], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_semantic], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'tanh')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'tanh')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model

def build_final_test8():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'elu', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_semantic], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_semantic], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model

def build_final_test9():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(law_merge)
    law_merge_att = Dense(config.hid_size, activation = 'elu')(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'elu')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model


def build_final_test10():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last', activation = 'relu')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'relu')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'relu')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'relu')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(law_merge)
    law_merge_att = Dense(config.hid_size, activation = 'tanh')(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'tanh')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model

def build_final_test11():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(law_merge)
    law_merge_att = Dense(config.hid_size, activation = 'tanh')(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'tanh')(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model

def build_final_test12():
    
    #    embedding_matrix = np.load(config.embedding_path)
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                #                                weights=[embedding_matrix],
                                #                                input_length=config.total_num,
                                trainable=True,
                                mask_zero = False
                                )
    law_embedding_layer = Embedding(config.num_law_liu,
                                    config.hid_size,
                                    #                                    input_length=config.num_law_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    term_embedding_layer = Embedding(config.num_term_liu,
                                     config.hid_size,
                                     #                                     input_length=config.num_accu_liu,
                                     trainable=True,
                                     mask_zero=False)
    
    num_embedding_layer = Embedding(10,
                                    config.word_embed_size / 8,
                                    #                                     input_length=config.num_accu_liu,
                                    trainable=True,
                                    mask_zero=False)
    
    sentence_input = Input(shape=(config.total_num,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    cnninput = Reshape((config.total_num, config.word_embed_size, 1))(embedded_sequences)
    print(cnninput.shape)
    cnn2 = Conv2D(filters = config.hid_size / 4, kernel_size = (2, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    print(cnn2.shape)
    pooling2 = MaxPooling2D(pool_size=(config.total_num + 1 - 2, 1))(cnn2)
    print (pooling2.shape)
    pooling2 = Reshape((config.hid_size / 4,))(pooling2)
    print (pooling2.shape)
    cnn3 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    pooling3 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn3)
    pooling3 = Reshape((config.hid_size / 4,))(pooling3)
    cnn4 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    pooling4 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn4)
    pooling4 = Reshape((config.hid_size / 4,))(pooling4)
    cnn5 = Conv2D(filters = config.hid_size / 4, kernel_size = (3, config.word_embed_size), data_format='channels_last', activation = 'elu')(cnninput)
    pooling5 = MaxPooling2D(pool_size=(config.total_num + 1 - 3, 1))(cnn5)
    pooling5 = Reshape((config.hid_size / 4,))(pooling5)
    feature = concatenate([pooling2, pooling3, pooling4, pooling5])
    print(feature.shape)
    feature = Dropout(0.5)(feature)
    
    
    law_preds1 = Dense(config.num_law_liu, activation='softmax', name="law_preds1")(feature)
    
    law_input = Input(shape=(config.num_law_liu,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = dot([law_preds1,law_embedded_sequences], axes = (1, 1))
    law_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(law_merge)
    law_merge_att = Dense(config.hid_size, activation = 'tanh', use_bias = False)(law_merge)
    law_fix = multiply([feature, law_merge_semantic])
    
    accu_preds1 = Dense(config.num_accu_liu, activation='softmax', name="accu_preds1")(law_fix)
    
    
    accu_input = Input(shape=(config.num_accu_liu,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = dot([accu_preds1,accu_embedded_sequences], axes = (1, 1))
    accu_merge_semantic = Dense(config.hid_size, activation = 'elu', use_bias = False)(accu_merge)
    accu_merge_att = Dense(config.hid_size, activation = 'tanh', use_bias = False)(accu_merge)
    accu_fix = multiply([feature, accu_merge_semantic])
    
    pair_front_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_last_word_input = Input(shape = (config.total_pair, ), dtype = 'int32')
    pair_front_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isword_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    pair_front_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_last_num_input = Input(shape = (config.total_pair, 8), dtype = 'int32')
    pair_front_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    pair_last_isnum_input = Input(shape = (config.total_pair, ), dtype = 'float32')
    
    
    
    
    pair_front_word = embedding_layer(pair_front_word_input)
    pair_front_isword = Reshape((config.total_pair, 1))(pair_front_isword_input)
    pair_front_word = multiply([pair_front_isword, pair_front_word])
    
    pair_last_word = embedding_layer(pair_last_word_input)
    pair_last_isword = Reshape((config.total_pair, 1))(pair_last_isword_input)
    pair_last_word = multiply([pair_last_isword, pair_last_word])
    
    
    
    pair_front_num = num_embedding_layer(pair_front_num_input)
    pair_front_num = Reshape((config.total_pair, config.word_embed_size))(pair_front_num)
    pair_front_num = Dense(config.word_embed_size, activation = 'elu')(pair_front_num)
    pair_front_isnum = Reshape((config.total_pair, 1))(pair_front_isnum_input)
    pair_front_num = multiply([pair_front_isnum, pair_front_num])
    
    pair_last_num = num_embedding_layer(pair_last_num_input)
    pair_last_num = Reshape((config.total_pair, config.word_embed_size))(pair_last_num)
    pair_last_num = Dense(config.word_embed_size, activation = 'elu')(pair_last_num)
    pair_last_isnum = Reshape((config.total_pair, 1))(pair_last_isnum_input)
    pair_last_num = multiply([pair_last_isnum, pair_last_num])
    
    
    pair_front = add([pair_front_num, pair_front_word])
    pair_last = add([pair_last_num, pair_last_word])
    
    pair = concatenate([pair_front, pair_last])
    
    print ('pair', pair.shape)
    
    pair = Reshape((config.total_pair, 2, config.word_embed_size))(pair)
    
    pair = TimeDistributed(LSTM(config.hid_size, return_sequences=False))(pair)
    
    #    pair = add([pair_front, pair_last])
    print ('pair', pair.shape)
    pair_d = Dense(config.hid_size, activation = 'tanh', use_bias = False)(pair)
    
    weight_law = dot([pair_d, law_merge_att], axes = (-1, -1))
    weight_law = Activation('softmax')(weight_law)
    weight_accu = dot([pair_d, accu_merge_att], axes = (-1, -1))
    weight_accu = Activation('softmax')(weight_accu)
    
    feature_pair_law = dot([weight_law, pair], axes = (1, 1))
    feature_pair_accu = dot([weight_accu, pair], axes = (1, 1))
    
    
    feature_accu = concatenate([feature, feature_pair_accu])
    feature_accu = Dense(config.hid_size , activation = 'elu')(feature_accu)
    feature_law = concatenate([feature, feature_pair_law])
    feature_law = Dense(config.hid_size , activation = 'elu')(feature_law)
    
    
    accu_fix = multiply([feature_accu, accu_fix])
    
    law_fix = multiply([feature_law, law_fix])
    
    term_preds1 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds1")(accu_fix)
    term_preds2 = Dense(config.num_term_liu, activation ='softmax', name = "term_preds2")(law_fix)
    
    term_preds3 = multiply([term_preds1, term_preds2])
    term_preds3 = Lambda(normalize, name = "term_preds3")(term_preds3)
    
    
    term_input = Input(shape=(config.num_term_liu,),dtype='int32')
    term_embedded_sequences = term_embedding_layer(term_input)
    term_merge = dot([term_preds3,term_embedded_sequences], axes = (1, 1))
    
    
    
    term2law = Dense(config.num_law_liu, activation = 'sigmoid')(term_merge)
    term2accu = Dense(config.num_accu_liu, activation = 'sigmoid')(term_merge)
    
    accu2law = Dense(config.num_law_liu, activation = 'sigmoid')(accu_merge)
    
    law_preds2 = multiply([law_preds1, term2law, accu2law])
    law_preds2 = Lambda(normalize, name = "law_preds2")(law_preds2)
    
    accu_preds2 = multiply([accu_preds1, term2accu])
    accu_preds2 = Lambda(normalize, name = "accu_preds2")(accu_preds2)
    
    
    model = Model(inputs=[sentence_input, accu_input, law_input, term_input, pair_front_word_input, pair_last_word_input, pair_front_isword_input, pair_last_isword_input, pair_front_num_input, pair_last_num_input, pair_front_isnum_input, pair_last_isnum_input],
                  outputs=[law_preds2,accu_preds2,term_preds3])
                                
    return model
