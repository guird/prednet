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
#from data_utils import SequenceGenerator 

WEIGHTS_DIR = "model_data"
DATA_DIR = "data"
RESULTS_SAVE_DIR = "../vim2/results"
WEIGHTS_OUT_DIR = "vim2_weights"


n_plot = 40
sample_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_kitti_model.json')
training_file = os.path.join(DATA_DIR, 'vim2_train')
out_file = os.path.join(WEIGHTS_OUT_DIR, "vim2_weights")
#test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')

num_epochs = 1
num_samples = 2#let's just say for now
batch_size = 2
num_batches = int(num_samples/batch_size)

#load model


#train from scratch

nt = 10
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
                  output_mode='all_ error', return_sequences=True)

inputs = Input(shape=(nt,) + input_shape)
errors = prednet(inputs)  # errors will be (batch_size, nt, nb_layers)

model = Model(input=inputs, output=errors)
model.compile(loss='mean_absolute_error', optimizer='adam')

#test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', dim_ordering=dim_ordering)
#X_test = test_generator.create_all()
#[int(vim2_stim2.shape[0] / batch_size# )
X_train = np.zeros([num_samples, sample_size, 128, 160,3])


for i in (range(num_samples)):
    X_train[i,:,:,:,:] = hkl.load(training_file + str(i) +".hkl")
X_train = np.transpose(X_train, (0, 1, 4, 2, 3))



errors_shape = (batch_size,10,4, 128, 160, 3)

target_zero = np.zeros(errors_shape);

for e in range(num_epochs):
    samples = range(num_samples)
    np.random.shuffle(samples)

    for i in range(num_batches):

        model.train_on_batch(X_train[i*batch_size:(i+1)*batch_size], target_zero)


json_string = model.to_json()
with open(out_file, "w") as f:
    f.write(json_string)

#X_hat = np.transpose(X_hat, (0, 1, 4, 2, 3))

