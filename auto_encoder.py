import sys

from tools.autoencoder_model import *


import h5py



h5 = h5py.File('data/TIMIT_enc_dec_train.hdf5','r')
# X_p = h5['data']['train_normalized'][()]
X_p = h5['data']['train_original'][()]
X_p = X_p / 10000
print'X_p type and shape:', X_p.dtype, X_p.shape
print'X_p.min():', X_p.min()
print'X_p.max():', X_p.max()

# X = np.zeros((X_p.shape[0], 5*X_p.shape[1]), dtype=np.float32)
# for index in tqdm(X_p.shape[0]):
#     X[index,:] = X_p[index-2,index-1,index,index+1,index+2,:].ravel()
# print'X type and shape:', X.dtype, X.shape
# print'X.min():', X.min()
# print'X.max():', X.max()
# exit()


encode_size = 39
in_out_dim = 400
width = 800
timesteps = 5



model_enc=auto_encoder(in_out_dim,width,encode_size)
# model_enc.fit(X_p, X_p,nb_epoch=10, batch_size=128, callbacks=[EarlyStopping(patience=2, save_weigths='models/encoder_adam_400_original.dat')], validation_split=0.1)
# model_enc.load_weights('models/encoder_adam_400.dat')
model_enc.load_weights('models/encoder_adam_400_original.dat')

X_pred = encode(model_enc,10,X_p[0:2,:])
print X_pred.shape

Y_pred = decode(model_enc,10,X_pred)
print Y_pred.shape


plt.figure(1)
plt.title('Signal Wave...')
plt.plot(X_p[0,:])
plt.plot(Y_pred[0,:])
plt.savefig('predicted.png')

