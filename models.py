# -*--coding:utf-8*-
config = Config()


class AttLayer(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        self.supports_masking = True
        super(AttLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], 1)))
        self.b = K.variable(self.init((input_shape[-2], )))
        self.u = K.variable(self.init((input_shape[-2], )))
        self.trainable_weights = [self.W]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        # if self.return_attention:
        #     return [mask, None]
        return mask

    def call(self, x, mask=None):
        uit = K.dot(x, self.W)  # 对应公式(5)
        uit = K.squeeze(uit, -1)  # 对应公式(5)
        uit = uit + self.b  # 对应公式(5)
        uit = K.tanh(uit)  # 对应公式(5)
        ait = uit * self.u  # 对应公式(6)
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


def build_model():

    embedding_matrix = np.load(config.embedding_path)

    embedding_layer = Embedding(config.word_vocab_size,
                                config.word_embed_size,
                                weights=[embedding_matrix],
                                input_length=config.num_words,
                                trainable=True,
                                mask_zero=True)

    sentence_input = Input(shape=(config.num_sents,), dtype='int32')
    embedded_sequences = embedding_layer(sentence_input)
    l_lstm = Bidirectional(GRU(150, return_sequences=True))(embedded_sequences)
    l_dense = TimeDistributed(Dense(300))(l_lstm)
    l_att = AttLayer()(l_dense)
    sentEncoder = Model(sentence_input, l_att)

    review_input = Input(shape=(config.num_sents, config.num_words), dtype='int32')
    review_encoder = TimeDistributed(sentEncoder)(review_input)
    l_lstm_sent = Bidirectional(GRU(150, return_sequences=True))(review_encoder)
    l_dense_sent = TimeDistributed(Dense(300))(l_lstm_sent)
    l_att_sent = AttLayer()(l_dense_sent)
    preds = Dense(config.num_accu, activation='softmax')(l_att_sent)
    model = Model(review_input, preds)

    return model
