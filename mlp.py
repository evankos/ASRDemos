
from tools.autoencoder_model import *
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import Adam, SGD
import theano
import lasagne
from nolearn.lasagne import NeuralNet, TrainSplit
from IPython.display import clear_output
from tqdm import *

import sys

# sys.path.append('../python')

from tools.data import Corpus, History
from sklearn.metrics import log_loss
train=Corpus('data/TIMIT_train.hdf5',load_normalized=True)
dev=Corpus('data/TIMIT_dev.hdf5',load_normalized=True)
# test=Corpus('../data/TIMIT_test.hdf5',load_normalized=True)

tr_in,tr_out_dec=train.get()
dev_in,dev_out_dec=dev.get()

# tr_in,tr_out_dec=train.get_enc()
# dev_in,dev_out_dec=dev.get_enc()

# tst_in,tst_out_dec=test.get()

print tr_in.shape
print tr_in[0].shape
print tr_out_dec.shape
print tr_out_dec[0].shape




# for u in range(tr_in.shape[0]):
#     tr_in[u]=tr_in[u][:,:26]
# for u in range(dev_in.shape[0]):
#     dev_in[u]=dev_in[u][:,:26]
# for u in range(tst_in.shape[0]):
#     tst_in[u]=tst_in[u][:,:26]


input_dim=tr_in[0].shape[1]
output_dim=61
hidden_num=512
epoch_num=3


def dec2onehot(dec):
    ret=[]
    for u in dec:
        assert np.all(u<output_dim)
        num=u.shape[0]
        r=np.zeros((num,output_dim))
        r[range(0,num),u]=1
        ret.append(r)
    return np.array(ret)



tr_out=dec2onehot(tr_out_dec)
dev_out=dec2onehot(dev_out_dec)
# tst_out=dec2onehot(tst_out_dec)

model = Sequential()

model.add(Dense(input_dim=input_dim,output_dim=hidden_num))
model.add(Activation('tanh'))
model.add(Dropout(0.3))
model.add(Dense(output_dim=hidden_num))
model.add(Activation('tanh'))
model.add(Dense(output_dim=output_dim))
model.add(Activation('softmax'))

# optimizer= SGD(lr=0.003,momentum=0.9,nesterov=True)
optimizer = Adam()
loss='categorical_crossentropy'
metrics=['accuracy']

model.compile(loss=loss, optimizer=optimizer, metrics=metrics)


from random import shuffle

tr_hist=History('Train')
dev_hist=History('Dev')
tst_hist=History('Test')


tr_it=range(tr_in.shape[0])

for e in range(epoch_num):

    print 'Epoch #{}/{}'.format(e+1,epoch_num)
    sys.stdout.flush()

    shuffle(tr_it)


    for u in tqdm(tr_it):
        l,a=model.train_on_batch(tr_in[u],tr_out[u])
        tr_hist.r.addLA(l,a,tr_out[u].shape[0])
    clear_output()
    tr_hist.log()

    for u in range(dev_in.shape[0]):
        l,a=model.test_on_batch(dev_in[u],dev_out[u])
        dev_hist.r.addLA(l,a,dev_out[u].shape[0])
    dev_hist.log()


    # for u in range(tst_in.shape[0]):
    #     l,a=model.test_on_batch(tst_in[u],tst_out[u])
    #     tst_hist.r.addLA(l,a,tst_out[u].shape[0])
    # tst_hist.log()


pickle.dump(model, open('models/classifier_enc.pkl','wb'))
pickle.dump(dev_hist, open('models/testHist_enc.pkl','wb'))
pickle.dump(tr_hist, open('models/trainHist_enc.pkl','wb'))


import matplotlib.pyplot as P

fig,ax=P.subplots(2,sharex=True,figsize=(12,10))

ax[0].set_title('Loss')
ax[0].plot(tr_hist.loss,label='Train')
ax[0].plot(dev_hist.loss,label='Dev')
# ax[0].plot(tst_hist.loss,label='Test')
ax[0].legend()
ax[0].set_ylim((1.4,2.0))


ax[1].set_title('PER %')
ax[1].plot(100*(1-np.array(tr_hist.acc)),label='Train')
ax[1].plot(100*(1-np.array(dev_hist.acc)),label='Dev')
# ax[1].plot(100*(1-np.array(tst_hist.acc)),label='Test')
ax[1].legend()
ax[1].set_ylim((45,55))
fig.savefig('train_enc.png')

print 'Min train PER: {:%}'.format(1-np.max(tr_hist.acc))
# print 'Min test PER: {:%}'.format(1-np.max(tst_hist.acc))
print 'Min dev PER epoch: #{}'.format((np.argmax(dev_hist.acc)+1))
# print 'Test PER on min dev: {:%}'.format(1-tst_hist.acc[np.argmax(dev_hist.acc)])





######################################################################################