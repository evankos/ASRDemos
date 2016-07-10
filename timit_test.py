import sys

# sys.path.append('../python')

from tools.timit import *


train_corp=prepare_corp_dir('data/TIMIT_train.list','C:/Users/Vaggelis/PycharmProjects/TIMIT/train')
dev_corp=prepare_corp_dir('data/TIMIT_dev.list','C:/Users/Vaggelis/PycharmProjects/TIMIT/test')


encoeder_decoder_dataset(train_corp,'data/TIMIT_enc_dec_train.hdf5')
exit()
extract_features(train_corp, 'data/TIMIT_train.hdf5')
extract_features(dev_corp, 'data/TIMIT_dev.hdf5')


normalize('data/TIMIT_train.hdf5')
normalize('data/TIMIT_dev.hdf5')
