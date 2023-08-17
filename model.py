
# import keras
# from keras.layers import *
# from keras.models import *
# from keras.optimizers import Adam
# from keras.regularizers import l1, l2
# import numpy as np
# from keras import backend as K
# from keras.engine.topology import Layer, InputSpec
# from keras import initializers
# # from tensorflow_core.contrib.eager.python.examples.rnn_ptb.rnn_ptb import Embedding
# from keras.models import Model
# from keras.layers import *
# from keras.activations import *
# from keras.regularizers import *
# from keras.initializers import *


from layers import SelfAttention


import keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.optimizers import Adam
from keras.regularizers import l1, l2
# from keras_self_attention import SeqSelfAttention
import numpy as np
from tensorflow.keras import backend as K
# from tensorflow.keras.engine.topology import Layer, InputSpec
from tensorflow.keras import initializers
from tensorflow.keras import regularizers

# from tensorflow_core.contrib.eager.python.examples.rnn_ptb.rnn_ptb import Embedding


size=60
num_hops=30

kernal_size = 4
num_filters = 20
muti_layer_Conv_filters = [20, 40, 60, 80]
kernal_size_2 = 10
pool_size = 2
strides = 2
MAX_LEN_en = 3000
MAX_LEN_pr = 2000
NB_WORDS = 4097
EMBEDDING_DIM = 100
embedding_matrix = np.load('embedding_matrix.npy')
regularization_lambda = 0.0001
attention_filters1 = 50
attention_filters2 = 10

class AttLayer(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
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
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        # output = K.sum(weighted_input, axis=1)
        output = weighted_input
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape)
        # return (input_shape[0], input_shape[-1])

class AttLayer_last(Layer):
    def __init__(self, attention_dim):
        # self.init = initializers.get('normal')
        self.init = initializers.RandomNormal(seed=10)
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer_last, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainableWeights = [self.W, self.b, self.u]
        super(AttLayer_last, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

class attention(Layer):
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(attention,self).__init__()

    def build(self, input_shape):
        #assert len(input_shape) == 3
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")

        super(attention,self).build(input_shape)


    def call(self, x, mask=None):
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        if self.return_sequences:
            return output
        return K.sum(output, axis=1)

def get_model_OneHot_SelfAttention_SoftAttention():
# def get_model():
    enhancers=Input(shape=(MAX_LEN_en, 4,))
    promoters=Input(shape=(MAX_LEN_pr, 4,))

    enhancer_conv_layer_1 = Conv1D(filters = num_filters,kernel_size = kernal_size,padding = "valid",activation='relu')(enhancers)
    enhancer_conv_layer_2 = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(enhancer_conv_layer_1)
    enhancer_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(enhancer_conv_layer_2)

    promoter_conv_layer_1 = Conv1D(filters = num_filters,kernel_size = kernal_size,padding = "valid",activation='relu')(promoters)
    promoter_conv_layer_2 = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(promoter_conv_layer_1)
    promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer_2)

    merge_layer=Concatenate(axis=1)([enhancer_max_pool_layer, promoter_max_pool_layer])
    bn=BatchNormalization()(merge_layer)
    dt=Dropout(0.5)(bn)

    l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(dt)
    l_SelfAtt = SelfAttention(size=size, num_hops=num_hops, use_penalization=False, model_api='matrix')(l_gru_1)

    l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(l_SelfAtt)
    l_SoftAtt = AttLayer_last(20)(l_gru_2)

    preds = Dense(1, activation='sigmoid')(l_SoftAtt)

    model=Model([enhancers,promoters],preds)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model

def get_model_dna2vec_SelfAttention_SoftAttention():
# def get_model():

    enhancers=Input(shape=(MAX_LEN_en,))
    promoters=Input(shape=(MAX_LEN_pr,))

    emb_en=Embedding(NB_WORDS,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(enhancers)
    emb_pr=Embedding(NB_WORDS,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(promoters)

    # emb_en=Embedding(NB_WORDS, EMBEDDING_DIM, trainable=True)(enhancers)
    # emb_pr=Embedding(NB_WORDS, EMBEDDING_DIM, trainable=True)(promoters)

    enhancer_conv_layer = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(emb_en)
    enhancer_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(enhancer_conv_layer)

    promoter_conv_layer = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(emb_pr)
    promoter_max_pool_layer = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer)

    merge_layer=Concatenate(axis=1)([enhancer_max_pool_layer, promoter_max_pool_layer])
    bn=BatchNormalization()(merge_layer)
    dt=Dropout(0.5)(bn)

    l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(dt)
    l_SelfAtt = SelfAttention(size=size, num_hops=num_hops, use_penalization=False, model_api='matrix')(l_gru_1)

    l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(l_SelfAtt)
    l_SoftAtt = AttLayer_last(20)(l_gru_2)

    preds = Dense(1, activation='sigmoid')(l_SoftAtt)

    model=Model([enhancers,promoters],preds)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model

#def get_model_dna2vec_OneHot_SelfAttention_SoftAttention():
def get_model():
    enhancers_dna2vec=Input(shape=(MAX_LEN_en,))
    promoters_dna2vec=Input(shape=(MAX_LEN_pr,))
    enhancers_oneHot = Input(shape=(MAX_LEN_en, 4,))
    promoters_oneHot = Input(shape=(MAX_LEN_pr, 4,))

    emb_en_dna2vec=Embedding(NB_WORDS,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(enhancers_dna2vec)
    emb_pr_dna2vec=Embedding(NB_WORDS,EMBEDDING_DIM,weights=[embedding_matrix],trainable=True)(promoters_dna2vec)

    enhancer_conv_layer_dna2vec = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(emb_en_dna2vec)
    enhancer_max_pool_layer_dna2vec = MaxPooling1D(pool_size = 20, strides = 20)(enhancer_conv_layer_dna2vec)

    promoter_conv_layer_dna2vec = Conv1D(filters = 64,kernel_size = 40,padding = "valid",activation='relu')(emb_pr_dna2vec)
    promoter_max_pool_layer_dna2vec = MaxPooling1D(pool_size = 20, strides = 20)(promoter_conv_layer_dna2vec)

    merge_layer_dna2vec=Concatenate(axis=1)([enhancer_max_pool_layer_dna2vec, promoter_max_pool_layer_dna2vec])


    enhancer_conv_layer_oneHot_1 = Conv1D(filters=num_filters, kernel_size=kernal_size, padding="valid", activation='relu')(
        enhancers_oneHot)
    enhancer_conv_layer_oneHot_2 = Conv1D(filters=64, kernel_size=40, padding="valid", activation='relu')(enhancer_conv_layer_oneHot_1)
    enhancer_max_pool_layer_oneHot = MaxPooling1D(pool_size=20, strides=20)(enhancer_conv_layer_oneHot_2)

    promoter_conv_layer_oneHot_1 = Conv1D(filters=num_filters, kernel_size=kernal_size, padding="valid", activation='relu')(
        promoters_oneHot)
    promoter_conv_layer_oneHot_2 = Conv1D(filters=64, kernel_size=40, padding="valid", activation='relu')(promoter_conv_layer_oneHot_1)
    promoter_max_pool_layer_oneHot = MaxPooling1D(pool_size=20, strides=20)(promoter_conv_layer_oneHot_2)

    merge_layer_oneHot = Concatenate(axis=1)([enhancer_max_pool_layer_oneHot, promoter_max_pool_layer_oneHot])

    merge_layer = Concatenate(axis=1)([merge_layer_dna2vec, merge_layer_oneHot])

    bn=BatchNormalization()(merge_layer)

    dt=Dropout(0.5)(bn)

    l_gru_1 = Bidirectional(GRU(50, return_sequences=True))(dt)
    l_SelfAtt = SelfAttention(size=size, num_hops=num_hops, use_penalization=False, model_api='matrix')(l_gru_1)

    l_gru_2 = Bidirectional(GRU(50, return_sequences=True))(l_SelfAtt)
    l_SoftAtt = AttLayer_last(20)(l_gru_2)

    preds = Dense(1, activation='sigmoid')(l_SoftAtt)

    model=Model([enhancers_dna2vec,promoters_dna2vec,enhancers_oneHot, promoters_oneHot],preds)
    model.compile(loss='binary_crossentropy',optimizer='adam')
    return model
