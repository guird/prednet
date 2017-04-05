'''
Evaluate trained PredNet on KITTI sequences.
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
from six.moves import cPickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import h5py as h
import hickle as hkl

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from theano.tensor.basic import zeros as thzero
#from data_utils import SequenceGenerator 

WEIGHTS_DIR = "model_data"
DATA_DIR = "../vim2/preprocessed/"
RESULTS_SAVE_DIR = "../vim2/results"
WEIGHTS_OUT_DIR = "vim2_weights"


n_plot = 40
sample_size = 10


weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
training_file = os.path.join(DATA_DIR, 'train')
out_file = os.path.join(WEIGHTS_OUT_DIR, "vim2_weights")
#test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

num_epochs = 1
num_samples = 108000#let's just say for now
batch_size = 5400
num_batches = int(num_samples/batch_size)
nt = batch_size/4
#load model


#train from scratch


input_shape = (3, 128, 160)
stack_sizes = (input_shape[0], 48, 96, 192)
R_stack_sizes = stack_sizes
A_filt_sizes = (3, 3, 3)
Ahat_filt_sizes = (3, 3, 3, 3)
R_filt_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0., 0., 0.])
layer_loss_weights = np.expand_dims(layer_loss_weights, 1)
time_loss_weights = 1./ (nt - 1) * np.ones((nt,1))
time_loss_weights[0] = 0


prednet = PredNet(stack_sizes, R_stack_sizes,
                  A_filt_sizes, Ahat_filt_sizes, R_filt_sizes,
                  output_mode='error', return_sequences=True, dim_ordering='th')

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)

model = Model(input=inputs, output=errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

#test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', dim_ordering=dim_ordering)
#X_test = test_generator.create_all()
#[int(vim2_stim2.shape[0] / batch_size# )







errors_shape = (batch_size/nt,nt,4)

target_zero = np.zeros(errors_shape);

batches = range(num_batches) #a list of the indices all samples of 10 frames


for e in range(num_epochs): #execute some epochs
    print "start epoch " + str(e)
    np.random.shuffle(batches) #randomize sample order


    for batch_num in batches: #for each minibatch
        #batch = samples[batch_size*batch_num:batch_size*(batch_num + 1)] #construct a list of indices of samples for this batch
        print " batch: " + str(batch_num)
        fname = DATA_DIR + "train"+str(batch_num * batch_size) +"_" + str((batch_num + 1) * batch_size) + ".hkl"
        print fname
        X_train =  np.zeros((batch_size/nt,nt, 128, 160,    3))
        f = hkl.load(fname)

        for i in range(batch_size/nt):
            X_train[i,:,:,:,:] = f[i*nt:(i+1)*nt]

        #X_train[0,:,:,:,:] = f[0:batch_size/2]
        #X_train[1,:,:,:,:] = f[batch_size/2:]#initialize X_test before we load values

        print target_zero.shape
        ind = 0 #index in X_train

        X_train = np.transpose(X_train, (0, 1, 4, 2, 3)) #permute dimensions so prednet can accept it
        print X_train.shape
        l = model.train_on_batch(X_train, target_zero)
        print l
        

json_string = model.to_json()
with open(out_file, "w") as f:
    f.write(json_string)

#X_hat = np.transpose(X_hat, (0, 1, 4, 2, 3))

