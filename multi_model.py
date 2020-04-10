
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


class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()
    
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
        return mask
    
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
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=False))(embedded_sequences)
    sentEncoder = Model(sentence_input, l_lstm)

    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=False))(review_encoder)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_lstm_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_lstm_sent)
    model = Model(inputs=[review_input], outputs=[accu_preds1, law_preds1])
                                
    return model

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
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    model = Model(inputs=[review_input], outputs=[accu_preds1, law_preds1])
                                
    return model


def build_model3():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_clear, activation = 'tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_clear, activation = 'tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Activation('sigmoid',name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Activation('sigmoid',name = 'law_preds3')(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model


def build_model4():
    
    embedding_matrix = np.load(config.embedding_path)
    law_n_embedding = np.load(config.law_n_embedding)
    accu_n_embedding = np.load(config.accu_n_embedding)
    
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    weights=[law_n_embedding],
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     weights=[accu_n_embedding],
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_clear, activation = 'tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_clear, activation = 'tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Activation('sigmoid',name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Activation('sigmoid',name = 'law_preds3')(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model

def build_model5():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Activation('tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Activation('tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Activation('sigmoid',name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Activation('sigmoid',name = 'law_preds3')(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model


def build_model6():
    
    embedding_matrix = np.load(config.embedding_path)
    law_n_embedding = np.load(config.law_n_embedding)
    accu_n_embedding = np.load(config.accu_n_embedding)
    
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    weights=[law_n_embedding],
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     weights=[accu_n_embedding],
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Activation('tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Activation('tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Activation('sigmoid',name = 'accu_preds3')(accu_pred_merge)
    law_preds3 = Activation('sigmoid',name = 'law_preds3')(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model

def build_model7():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_clear, activation = 'tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_clear, activation = 'tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Dense(config.num_accu_clear, activation = 'sigmoid', name="accu_preds3")(accu_pred_merge)
    law_preds3 = Dense(config.num_law_clear, activation = 'sigmoid', name="law_preds3")(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model


def build_model8():
    
    embedding_matrix = np.load(config.embedding_path)
    law_n_embedding = np.load(config.law_n_embedding)
    accu_n_embedding = np.load(config.accu_n_embedding)
    
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    weights=[law_n_embedding],
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     weights=[accu_n_embedding],
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Dense(config.num_law_clear, activation = 'tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Dense(config.num_accu_clear, activation = 'tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Dense(config.num_accu_clear, activation = 'sigmoid', name="accu_preds3")(accu_pred_merge)
    law_preds3 = Dense(config.num_law_clear, activation = 'sigmoid', name="law_preds3")(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model

def build_model9():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Activation('tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Activation('tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Dense(config.num_accu_clear, activation = 'sigmoid', name="accu_preds3")(accu_pred_merge)
    law_preds3 = Dense(config.num_law_clear, activation = 'sigmoid', name="law_preds3")(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model


def build_model10():
    
    embedding_matrix = np.load(config.embedding_path)
    law_n_embedding = np.load(config.law_n_embedding)
    accu_n_embedding = np.load(config.accu_n_embedding)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    weights=[law_n_embedding],
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     weights=[accu_n_embedding],
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    print (l_att.shape)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    print (review_input.shape)
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    print (review_encoder.shape)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Activation('tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Activation('tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Dense(config.num_accu_clear, activation = 'sigmoid', name="accu_preds3")(accu_pred_merge)
    law_preds3 = Dense(config.num_law_clear, activation = 'sigmoid', name="law_preds3")(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model

def build_model11():
    
    embedding_matrix = np.load(config.embedding_path)
    
    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)
        
    law_embedding_layer = Embedding(config.num_law_clear,
                                    config.num_accu_clear,
                                    input_length=config.num_law_clear,
                                    trainable=True,
                                    mask_zero=False)
    
    accu_embedding_layer = Embedding(config.num_accu_clear,
                                     config.num_law_clear,
                                     input_length=config.num_accu_clear,
                                     trainable=True,
                                     mask_zero=False)
    
    sentence_input = Input(shape=(config.num_words,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(config.hid_size, return_sequences=True))(embedded_sequences)
    #    l_dense = TimeDistributed(Dense(config.hid_size * 2))(l_lstm)
    l_att = AttLayer(config.attention_dim)(l_lstm)
    sentEncoder = Model(sentence_input, l_att)
    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(config.hid_size, return_sequences=True))(review_encoder)
    #    l_dense_sent = TimeDistributed(Dense(config.hid_size * 2))(l_lstm_sent)
    l_att_sent = AttLayer(config.attention_dim)(l_lstm_sent)
    accu_preds1 = Dense(config.num_accu_clear, activation='sigmoid', name="accu_preds1")(l_att_sent)
    law_preds1 = Dense(config.num_law_clear, activation='sigmoid', name="law_preds1")(l_att_sent)
    
    accu_input = Input(shape=(config.num_accu_clear,),dtype='int32')
    accu_embedded_sequences = accu_embedding_layer(accu_input)
    accu_merge = merge([accu_preds1,accu_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    law_preds2 = Activation('tanh')(accu_merge)
    
    law_input = Input(shape=(config.num_law_clear,),dtype='int32')
    law_embedded_sequences = law_embedding_layer(law_input)
    law_merge = merge([law_preds1,law_embedded_sequences], mode = 'dot', dot_axes = (1, 1))
    accu_preds2 = Activation('tanh')(law_merge)
    
    accu_pred_merge = merge([accu_preds1, accu_preds2], mode = 'mul')
    law_pred_merge = merge([law_preds1, law_preds2], mode = 'mul')
    
    accu_preds3 = Dense(config.num_accu_clear, activation = 'sigmoid', name="accu_preds3")(accu_pred_merge)
    law_preds3 = Dense(config.num_law_clear, activation = 'sigmoid', name="law_preds3")(law_pred_merge)
    
    model = Model(inputs=[review_input, accu_input, law_input], outputs=[accu_preds1, law_preds1, accu_preds3, law_preds3])
                                
    return model
