
try:
    import cPickle as pickle
except:
    import pickle

from keras.callbacks import Callback
import warnings
import matplotlib
matplotlib.use('Agg') # Change matplotlib backend, in case we have no X server running..
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, SGD
from keras.models import Model
import theano
from keras import backend as K





class EarlyStopping(Callback):
    def __init__(self, monitor='val_loss', patience=3, verbose=0, save_weigths='models/encoder_adam_400.dat'):
        super(Callback, self).__init__()
        self.save_weights = save_weigths
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.best = np.Inf
        self.wait = 0
        self.best_weights = None

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % (self.monitor), RuntimeWarning)

        if current < self.best:
            self.best = current
            self.best_weights = self.model.get_weights()
            self.model.save_weights(self.save_weights,overwrite=True)
            self.wait = 0
        else:
            if self.wait >= self.patience:
                if self.verbose > 0:
                    print("Epoch %05d: early stopping" % (epoch))
                self.model.set_weights
                self.model.stop_training = True
            self.wait += 1

def auto_encoder(in_out_dim,width,encode_size):
    model_enc = Sequential()
    model_enc.add(Dense(input_dim=in_out_dim,output_dim=width,name='input'))
    model_enc.add(Activation('tanh'))
    model_enc.add(BatchNormalization())
    model_enc.add(Dense(output_dim=width/2))
    model_enc.add(Activation('tanh'))
    model_enc.add(BatchNormalization())
    model_enc.add(Dense(output_dim=width/4))
    model_enc.add(Activation('tanh'))
    model_enc.add(BatchNormalization())
    model_enc.add(Dense(output_dim=encode_size,name='encoded'))
    model_enc.add(Activation('tanh',name='encoded_tanh'))
    model_enc.add(BatchNormalization())
    model_enc.add(Dense(output_dim=width/4))
    model_enc.add(Activation('tanh'))
    model_enc.add(BatchNormalization())
    model_enc.add(Dense(output_dim=width/2))
    model_enc.add(Activation('tanh'))
    model_enc.add(BatchNormalization())
    model_enc.add(Dense(output_dim=width))
    model_enc.add(Activation('tanh'))
    model_enc.add(BatchNormalization())
    model_enc.add(Dense(output_dim=in_out_dim))
    model_enc.add(Activation('linear',name='output'))
    # optimizer= SGD(lr=0.001,momentum=0.9,nesterov=True)
    optimizer = Adam(lr=0.003)
    loss='mse'
    metrics=['accuracy']
    model_enc.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    return model_enc
def decode(model, layer, X_batch):

    get_activations = K.function([model.layers[layer].input, K.learning_phase()], [model.layers[-1].output,])
    activations = get_activations([X_batch,0])

    return np.array(activations[0])
    # decoder = Model(input=model.layers[layer].input, output=model.layers[-1].output)
    #
    # return decoder

def encode(model, layer, X_batch):

    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([X_batch,0])

    return np.array(activations[0])
    # decoder = Model(input=model.layers[0].input, output=model.layers[layer].output)
    #
    # return decoder
def lstm_enc_dec(timesteps,in_dim):
    m = Sequential()
    m.add(LSTM(timesteps, input_dim=in_dim, return_sequences=True))
    m.add(LSTM(in_dim, return_sequences=True))
    m.add(Activation('linear'))
    optimizer = Adam(lr=0.001)
    m.compile(loss='mse', optimizer=optimizer)
    return m
